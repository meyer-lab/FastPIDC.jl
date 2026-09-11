"""
    FastPIDCCUDAExt

Package extension providing CUDA-accelerated implementations of
[`FastPIDC.compute_puc_full_cuda`](@ref) and
[`FastPIDC.solve_bayesian_blocks_cuda`](@ref), loaded automatically once
`using CUDA` makes the `CUDA` package available alongside `FastPIDC`.
Selected by passing `config.backend = :cuda` and `config.bb_backend = :cuda`
(both the default).

Rather than maintaining a second, CUDA.jl-native copy of the kernels, this
extension compiles and drives the single canonical kernel source shared with
the Python package, `python/src/fastpidc/kernels/pidc_kernels.cu` (see that
file's header comment for both algorithms). It is plain CUDA C, so it is
compiled here with `nvcc` (required on `PATH` in addition to a functional
GPU) into PTX for the active device's compute capability, then loaded with
`CUDA.CuModule` and driven with `CUDA.cudacall` - the same kernels Python's
`cuda` backend loads via `cupy.RawModule`.
"""
module FastPIDCCUDAExt

using FastPIDC
using CUDA

# --- Shared kernel source: compile once per (session, GPU architecture) ---

const _MODULE_CACHE = Dict{String,CuModule}()

"""
    _kernel_source_path() -> String

Path to the canonical CUDA kernel source, shared with the Python package.
Resolved relative to this Julia package's root, since a git-based install
(`Pkg.add(url = ...)`) clones the whole repository, `python/` included.
"""
function _kernel_source_path()
    path = joinpath(pkgdir(FastPIDC), "python", "src", "fastpidc", "kernels", "pidc_kernels.cu")
    isfile(path) || error(
        "Shared CUDA kernel source not found at $path. FastPIDC.jl's CUDA " *
        "extension expects to find it relative to the package root (see " *
        "the FastPIDCCUDAExt module docstring); if FastPIDC.jl was " *
        "installed without the `python/` subdirectory, this backend is " *
        "unavailable.",
    )
    return path
end

"""
    _compile_ptx(arch::String) -> String

Compile the shared kernel source to PTX text targeting virtual architecture
`arch` (e.g. `"compute_89"`), using `nvcc`. Requires a CUDA toolkit
installation with `nvcc` on `PATH`.
"""
function _compile_ptx(arch::String)
    nvcc = Sys.which("nvcc")
    nvcc === nothing && error(
        "The CUDA backend needs `nvcc` (from a CUDA toolkit installation) " *
        "on PATH to compile the shared kernel source " *
        "(python/src/fastpidc/kernels/pidc_kernels.cu); none was found. " *
        "Install the CUDA toolkit, or use `config.backend = :cpu`.",
    )

    src = _kernel_source_path()
    mktempdir() do dir
        ptx_path = joinpath(dir, "pidc_kernels.ptx")
        cmd = `$nvcc --ptx -arch=$arch $src -o $ptx_path`
        out = IOBuffer()
        try
            run(pipeline(cmd; stdout=out, stderr=out))
        catch e
            error("nvcc failed to compile $src:\n$(String(take!(out)))")
        end
        return read(ptx_path, String)
    end
end

"""
    _get_module() -> CuModule

The compiled kernel module for the current device's compute capability,
compiling (and caching, per architecture, for the life of the Julia
session) on first use.
"""
function _get_module()
    cap = CUDA.capability(CUDA.device())
    arch = "compute_$(cap.major)$(cap.minor)"
    get!(_MODULE_CACHE, arch) do
        CuModule(_compile_ptx(arch))
    end
end

# --- Host implementation ---

"""
    _MAX_K_BINS

Largest bins-per-gene the shared kernels can index. They compute every flat
buffer offset in 64-bit, with one exception: `joint_counts_kernel` forms the
bin-pair index `u * k_bins + v` in Int32, which stays exact only while
`k_bins^2` fits in Int32 (isqrt(typemax(Int32)) == 46340).
"""
const _MAX_K_BINS = 46340

function _check_kernel_index_limits(k_bins::Integer)
    k_bins <= _MAX_K_BINS || error(
        "compute_puc_full_cuda: the discretizer selected k_bins=$k_bins bins " *
        "per gene, but the CUDA kernels index bin pairs in 32-bit and support " *
        "at most $(_MAX_K_BINS). Use discretizer=\"uniform_width\" with a " *
        "fixed, small number_of_bins, or config.backend = :cpu.",
    )
    return nothing
end

function _smallest_unsigned_type(max_value::Integer)
    max_value >= 0 || throw(ArgumentError("max_value must be nonnegative"))

    if max_value <= 255
        return UInt8
    elseif max_value <= 65_535
        return UInt16
    elseif max_value <= 4_294_967_295
        return UInt32
    else
        return UInt64
    end
end

"""
    FastPIDC.compute_puc_full_cuda(nodes, config, base) -> (mi_scores, puc_scores)

GPU implementation of [`FastPIDC.compute_puc_full`](@ref): computes the full
pairwise MI matrix and pre-context PUC matrix for `nodes` on the GPU,
processing genes along the target (`z`) axis in chunks (up to 256 genes at
a time) sized to fit the GPU memory currently free, to bound device memory
use even when the discretizer has picked a large number of bins (see the
chunk-sizing comment in the implementation). Moves discretized data and
marginal probabilities to the GPU once, then for each chunk launches the
shared `joint_counts_kernel`, `mi_si_kernel` and `puc_accumulation_kernel`
(see the `FastPIDCCUDAExt` module docstring) in sequence, symmetrizing the
resulting PUC matrix before returning both matrices to the CPU.
`config.verbose` enables progress printouts; `base` is currently unused
(mutual information is always computed in base 2 on the GPU, matching the
kernel source). Raises an `ErrorException` with a suggested remedy if even
a single-gene chunk would not fit in the currently-free GPU memory, or if
the discretizer selected more than `_MAX_K_BINS` bins per gene.

The chunked intermediates legitimately exceed 2^31 elements on large gene
sets - `counts` alone is `k_bins^2 * num_nodes * chunk_size` - so the shared
kernels index them in 64-bit (see the indexing contract in the kernel source).

Device buffers use Julia's column-major layout with dimensions reversed
relative to the kernel source's documented (row-major) shapes - e.g. a
kernel-documented `(k_bins, n)` array is allocated here with Julia size
`(n, k_bins)` - so that the flat in-memory layout the kernels index into
with manual pointer arithmetic is identical in both languages, with no
transposition needed at the call boundary.
"""
function FastPIDC.compute_puc_full_cuda(nodes, config, base)
    md = _get_module()
    joint_counts_kernel = CuFunction(md, "joint_counts_kernel")
    mi_si_kernel = CuFunction(md, "mi_si_kernel")
    puc_accumulation_kernel = CuFunction(md, "puc_accumulation_kernel")

    num_nodes = length(nodes)
    num_samples = length(nodes[1].binned_values)
    k_bins = maximum(n -> n.number_of_bins, nodes)
    _check_kernel_index_limits(k_bins)

    # Prepare static data on CPU and move to GPU. The shared CUDA C kernels use
    # 0-indexed Int32 bin ids; FastPIDC.jl's bin ids are 1-indexed, so shift
    # them down at this boundary.
    data_cpu = zeros(Int32, num_nodes, num_samples)          # kernel shape (m, n), reversed
    marginals_cpu = zeros(Float64, num_nodes, k_bins)       # kernel shape (k_bins, n), reversed
    for i = 1:num_nodes
        data_cpu[i, :] .= Int32.(nodes[i].binned_values) .- Int32(1)
        p = nodes[i].probabilities
        marginals_cpu[i, 1:length(p)] .= Float64.(p)
    end

    data_gpu = CuArray(data_cpu)
    marginals_gpu = CuArray(marginals_cpu)

    # Global output matrices (kernel shape (n, n); square, so no reversal needed).
    puc_scores_gpu = CUDA.zeros(Float64, num_nodes, num_nodes)
    mi_matrix_gpu = CUDA.zeros(Float64, num_nodes, num_nodes)

    # Chunked intermediates scale with k_bins^2 * num_nodes * chunk_size for
    # joint counts and k_bins * num_nodes * chunk_size for specific information.
    # Size the target-gene chunk from currently-free memory rather than always
    # allocating a fixed 256-gene chunk.
    #
    # This bounds MEMORY AVAILABILITY only - it is not, and must not be turned
    # back into, a bound on the flat element index. The kernels index these
    # buffers in 64-bit precisely so the chunk can be sized from free memory
    # without an int32 element-count cap.
    bytes_per_chunk_col =
        k_bins^2 * num_nodes * sizeof(Int32) +  # counts_chunk_gpu
        k_bins * num_nodes * sizeof(Float64)    # si_chunk_gpu
    free_bytes = Int(CUDA.free_memory())
    safety_factor = 0.8  # headroom for fixed buffers + allocator overhead/fragmentation
    max_chunk_size = floor(Int, free_bytes * safety_factor / bytes_per_chunk_col)
    chunk_size = clamp(max_chunk_size, 1, min(256, num_nodes))

    if max_chunk_size < 1
        error(
            "compute_puc_full_cuda: even a single-gene chunk would require " *
            "$(round(bytes_per_chunk_col / 2^30, digits = 2)) GiB of GPU memory " *
            "(only $(round(free_bytes * safety_factor / 2^30, digits = 2)) GiB " *
            "usable), because the discretizer selected k_bins=$k_bins bins per " *
            "gene. This is usually caused by an adaptive discretizer (e.g. " *
            "\"bayesian_blocks\", the default) picking an unbounded number of " *
            "bins on a dataset with many samples. Try discretizer=\"uniform_width\" " *
            "with a fixed, small number_of_bins (e.g. 10-20), or config.backend = :cpu.",
        )
    end

    # Chunked intermediate buffers (pre-allocated once), with dimensions
    # reversed to preserve the row-major flat layout expected by the CUDA C kernels.
    counts_chunk_gpu = CUDA.zeros(Int32, chunk_size, num_nodes, k_bins, k_bins)
    si_chunk_gpu = CUDA.zeros(Float64, chunk_size, num_nodes, k_bins)

    if config.verbose
        println(
            "[FastPIDC] GPU Chunked PUC: Processing $num_nodes x $num_nodes pairs " *
            "(k_bins=$k_bins)...",
        )
        println(
            "[FastPIDC] Using chunk size of $chunk_size " *
            "(approx. $(ceil(Int, num_nodes / chunk_size)) iterations), " *
            "sized to fit $(round(free_bytes / 2^30, digits = 2)) GiB free GPU memory",
        )
    end

    threads = (16, 16)

    # Iterate over the Z-axis in chunks. The shared kernels use 0-based target
    # indices, so convert z_start at the call boundary.
    for z_start_1 in 1:chunk_size:num_nodes
        z_start = z_start_1 - 1
        z_end = min(z_start_1 + chunk_size - 1, num_nodes)
        z_curr_chunk_size = z_end - z_start_1 + 1

        CUDA.fill!(counts_chunk_gpu, Int32(0))
        CUDA.fill!(si_chunk_gpu, Float64(0))

        blocks = (cld(num_nodes, 16), cld(z_curr_chunk_size, 16))

        cudacall(
            joint_counts_kernel,
            (CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, Cint, Cint, Cint),
            data_gpu, counts_chunk_gpu,
            Cint(num_nodes), Cint(num_samples), Cint(k_bins),
            Cint(z_start), Cint(z_curr_chunk_size);
            blocks=blocks, threads=threads,
        )

        cudacall(
            mi_si_kernel,
            (
                CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble},
                Cint, Cint, Cint, Cint, Cint,
            ),
            counts_chunk_gpu, marginals_gpu, mi_matrix_gpu, si_chunk_gpu,
            Cint(num_nodes), Cint(num_samples), Cint(k_bins),
            Cint(z_start), Cint(z_curr_chunk_size);
            blocks=blocks, threads=threads,
        )

        cudacall(
            puc_accumulation_kernel,
            (
                CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble},
                Cint, Cint, Cint, Cint,
            ),
            si_chunk_gpu, mi_matrix_gpu, puc_scores_gpu, marginals_gpu,
            Cint(num_nodes), Cint(k_bins),
            Cint(z_start), Cint(z_curr_chunk_size);
            blocks=blocks, threads=threads,
        )
    end

    # Kernels write row-major (x, z) into a Julia array whose dimensions were
    # reversed above, so transpose the square outputs back to Julia's convention.
    mi_matrix_cpu = permutedims(Array(mi_matrix_gpu))
    puc_scores_cpu = permutedims(Array(puc_scores_gpu))

    # Symmetrize PUC scores: each ordered pair contains one directional
    # contribution from the shared kernel.
    for i = 1:num_nodes
        for j = (i+1):num_nodes
            val = puc_scores_cpu[i, j] + puc_scores_cpu[j, i]
            puc_scores_cpu[i, j] = val
            puc_scores_cpu[j, i] = val
        end
    end

    return mi_matrix_cpu, puc_scores_cpu
end

# --- Bayesian-block CUDA backend -------------------------------------------

FastPIDC.bayesian_blocks_cuda_available() = CUDA.functional()

"""
    _bb_kernel_name(CountT, IndexT) -> String

Entry point in the shared kernel source for the given prefix-count and
back-pointer element types. CUDA C has no generics, so `pidc_kernels.cu`
macro-generates one `extern "C"` kernel per valid type pair and the host
selects by name.
"""
function _bb_kernel_name(
    ::Type{CountT},
    ::Type{IndexT},
) where {CountT<:Integer,IndexT<:Integer}
    suffixes = Dict{DataType,String}(
        UInt8 => "u8",
        UInt16 => "u16",
        UInt32 => "u32",
        UInt64 => "u64",
    )

    haskey(suffixes, CountT) || throw(
        ArgumentError(
            "Bayesian-block prefix counts must be an unsigned type the shared " *
            "kernel provides (UInt8, UInt16, UInt32 or UInt64); got $CountT",
        ),
    )
    haskey(suffixes, IndexT) || throw(
        ArgumentError(
            "Bayesian-block back-pointers must be an unsigned type the shared " *
            "kernel provides (UInt8, UInt16, UInt32 or UInt64); got $IndexT",
        ),
    )
    # U_g never exceeds the observation count, so only these pairs exist.
    sizeof(IndexT) <= sizeof(CountT) || throw(
        ArgumentError(
            "Bayesian-block back-pointer type $IndexT is wider than the " *
            "prefix-count type $CountT, which the shared kernel does not " *
            "instantiate (U_g never exceeds the observation count)",
        ),
    )

    return "bayesian_blocks_dp_$(suffixes[CountT])_$(suffixes[IndexT])"
end

function _bb_threads_for_max_u(max_u::Integer)
    if max_u <= 32
        return 32
    elseif max_u <= 512
        return 64
    elseif max_u <= 4_096
        return 128
    else
        return 256
    end
end

function _bb_quantile_buckets(problems::Vector{FastPIDC.BayesianBlocksProblem})
    n = length(problems)
    n == 0 && return Vector{Vector{Int}}()

    # U_g is already available from required preprocessing. Sorting only these
    # gene indices is a lightweight O(G log G) operation and avoids a second
    # scan of the expression matrix merely to choose GPU workload buckets.
    order = sortperm(eachindex(problems); by = i -> length(problems[i].prefix_counts))
    n_buckets = min(4, n)
    buckets = Vector{Vector{Int}}()
    for bucket = 1:n_buckets
        lo = fld((bucket - 1) * n, n_buckets) + 1
        hi = fld(bucket * n, n_buckets)
        lo <= hi && push!(buckets, collect(order[lo:hi]))
    end
    return buckets
end

function _bb_prior_values(max_u::Integer)
    # The prior depends only on endpoint K, not on the gene. Compute it once on
    # the CPU with the reference expression, avoiding one pow/log pair per gene
    # per endpoint and eliminating that source of CPU/CUDA numeric variation.
    return [4 - log(73.53 * 0.05 * ((K)^-0.478)) for K = 1:max_u]
end

function _bb_problem_bytes(
    problem::FastPIDC.BayesianBlocksProblem,
    ::Type{CountT},
    ::Type{IndexT},
) where {CountT<:Integer,IndexT<:Integer}
    u = length(problem.prefix_counts)
    return (
        sizeof(Float64) * (u + 1) + # block lengths
        sizeof(CountT) * u +        # prefix counts
        sizeof(Float64) * u +       # best scores
        sizeof(IndexT) * u          # back-pointers
    )
end

function _bb_memory_batches(
    bucket::Vector{Int},
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    budget_bytes::Integer,
    ::Type{CountT},
    ::Type{IndexT},
) where {CountT<:Integer,IndexT<:Integer}
    batches = Vector{Vector{Int}}()
    current = Int[]
    current_bytes = 0

    for problem_index in bucket
        problem_bytes = _bb_problem_bytes(problems[problem_index], CountT, IndexT)
        problem_bytes <= budget_bytes || throw(
            ArgumentError(
                "One Bayesian-block problem requires $(problem_bytes) bytes, " *
                "which exceeds the CUDA batch budget of $(budget_bytes) bytes. " *
                "Reduce the number of unique input values for that gene.",
            ),
        )

        if !isempty(current) && current_bytes + problem_bytes > budget_bytes
            push!(batches, current)
            current = Int[]
            current_bytes = 0
        end
        push!(current, problem_index)
        current_bytes += problem_bytes
    end

    !isempty(current) && push!(batches, current)
    return batches
end

function _flatten_bb_batch(
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    problem_indices::Vector{Int},
    ::Type{CountT},
) where {CountT<:Integer}
    n_genes = length(problem_indices)
    total_states = sum(i -> length(problems[i].prefix_counts), problem_indices)
    total_blocks = total_states + n_genes

    prefix_counts = Vector{CountT}(undef, total_states)
    block_lengths = Vector{Float64}(undef, total_blocks)
    state_offsets = Vector{Int64}(undef, n_genes)
    block_offsets = Vector{Int64}(undef, n_genes)
    unique_counts = Vector{Int32}(undef, n_genes)

    state_cursor = 1
    block_cursor = 1
    for (local_gene, problem_index) in enumerate(problem_indices)
        problem = problems[problem_index]
        u = length(problem.prefix_counts)
        u <= typemax(Int32) || throw(
            ArgumentError("Bayesian blocks CUDA backend supports at most $(typemax(Int32)) unique values per gene"),
        )

        state_offsets[local_gene] = state_cursor
        block_offsets[local_gene] = block_cursor
        unique_counts[local_gene] = Int32(u)

        @inbounds for j = 1:u
            prefix_counts[state_cursor+j-1] = CountT(problem.prefix_counts[j])
        end
        edge_end = problem.edges[end]
        @inbounds for j = 1:(u+1)
            block_lengths[block_cursor+j-1] = edge_end - problem.edges[j]
        end

        state_cursor += u
        block_cursor += u + 1
    end

    return prefix_counts, block_lengths, state_offsets, block_offsets, unique_counts
end

function _change_points_from_last(last_values, offset::Int, n::Int)
    n >= 1 || throw(ArgumentError("Bayesian-block backtracking requires n >= 1"))

    # A valid partition may place every unique value in its own block. In that
    # case the returned edge-index path contains U_g + 1 entries, so allocating
    # only U_g slots can underflow to Julia index zero during backtracking.
    change_points = Vector{Int64}(undef, n + 1)
    i_cp = n + 2
    ind = n + 1
    while true
        i_cp -= 1
        change_points[i_cp] = ind
        ind == 1 && break

        state = ind - 1
        1 <= state <= n || throw(
            ArgumentError("invalid Bayesian-block state $state while backtracking"),
        )
        # The shared kernel writes 0-based predecessors (see pidc_kernels.cu);
        # shift back to Julia's 1-based candidate indices here.
        next_ind = Int(last_values[offset + state - 1]) + 1
        1 <= next_ind <= state || throw(
            ArgumentError(
                "invalid Bayesian-block back-pointer $next_ind for state $state",
            ),
        )
        ind = next_ind
    end
    return change_points[i_cp:end]
end

function _solve_bb_cuda_batch_with_priors(
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    problem_indices::Vector{Int},
    threads::Int,
    ::Type{CountT},
    ::Type{IndexT},
    priors_gpu,
) where {CountT<:Integer,IndexT<:Integer}
    threads in (32, 64, 128, 256) || throw(
        ArgumentError(
            "CUDA Bayesian blocks requires a power-of-two thread count " *
            "from 32, 64, 128, or 256; got $threads",
        ),
    )

    prefix_counts, block_lengths, state_offsets, block_offsets, unique_counts =
        _flatten_bb_batch(problems, problem_indices, CountT)

    prefix_gpu = CuArray(prefix_counts)
    block_gpu = CuArray(block_lengths)
    # `state_offsets`/`block_offsets` are 1-based cursors for host-side slicing;
    # the shared kernels index 0-based, so convert at the call boundary (as the
    # PUC path does for `z_start`).
    state_offsets_gpu = CuArray(state_offsets .- 1)
    block_offsets_gpu = CuArray(block_offsets .- 1)
    unique_counts_gpu = CuArray(unique_counts)
    best_gpu = CUDA.zeros(Float64, length(prefix_counts))
    last_gpu = CUDA.zeros(IndexT, length(prefix_counts))
    final_scores_gpu = CUDA.zeros(Float64, length(problem_indices))

    bb_kernel = CuFunction(_get_module(), _bb_kernel_name(CountT, IndexT))

    try
        cudacall(
            bb_kernel,
            (
                CuPtr{CountT}, CuPtr{Cdouble}, CuPtr{Int64}, CuPtr{Int64},
                CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{IndexT}, CuPtr{Cdouble},
                CuPtr{Cdouble},
            ),
            prefix_gpu, block_gpu, state_offsets_gpu, block_offsets_gpu,
            unique_counts_gpu, best_gpu, last_gpu, final_scores_gpu, priors_gpu;
            blocks=length(problem_indices), threads=threads,
        )

        last_values = Array(last_gpu)
        final_scores = Array(final_scores_gpu)

        solutions =
            Vector{FastPIDC.BayesianBlocksSolution}(undef, length(problem_indices))
        for local_gene = eachindex(problem_indices)
            offset = state_offsets[local_gene]
            n = Int(unique_counts[local_gene])
            change_points = _change_points_from_last(last_values, offset, n)
            solutions[local_gene] = FastPIDC.BayesianBlocksSolution(
                change_points,
                final_scores[local_gene],
            )
        end
        return solutions
    finally
        # Explicitly return batch allocations to CUDA's pool. The CUDA backend
        # may process many U_g buckets, so relying on a later GC cycle can retain
        # unnecessary pressure between batches or after an exception.
        for array in (
            prefix_gpu,
            block_gpu,
            state_offsets_gpu,
            block_offsets_gpu,
            unique_counts_gpu,
            best_gpu,
            last_gpu,
            final_scores_gpu,
        )
            CUDA.unsafe_free!(array)
        end
    end
end

function _solve_bb_cuda_batch(
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    problem_indices::Vector{Int},
    threads::Int,
    ::Type{CountT},
    ::Type{IndexT},
) where {CountT<:Integer,IndexT<:Integer}
    max_u = maximum(i -> length(problems[i].prefix_counts), problem_indices)
    priors_gpu = CuArray(_bb_prior_values(max_u))
    try
        return _solve_bb_cuda_batch_with_priors(
            problems,
            problem_indices,
            threads,
            CountT,
            IndexT,
            priors_gpu,
        )
    finally
        CUDA.unsafe_free!(priors_gpu)
    end
end

function FastPIDC.solve_bayesian_blocks_cuda(
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    verbose::Bool,
)
    CUDA.functional() || return nothing
    isempty(problems) && return FastPIDC.BayesianBlocksSolution[]

    sample_count = maximum(p -> Int(round(p.prefix_counts[end])), problems)
    max_u = maximum(p -> length(p.prefix_counts), problems)

    # A cumulative prefix count can reach the number of cells, so select the
    # smallest exact unsigned type that guards against overflow for this input.
    CountT = _smallest_unsigned_type(sample_count)
    # Back-pointers only need to represent candidate indices up to U_g.
    IndexT = _smallest_unsigned_type(max_u)

    free_bytes = Int(CUDA.free_memory())
    # Keep headroom for the CUDA context, allocator bookkeeping, and other
    # active package allocations while still using most of the currently free
    # device memory.
    budget_bytes = max(1, floor(Int, 0.65 * Float64(free_bytes)))

    buckets = _bb_quantile_buckets(problems)
    solutions = Vector{FastPIDC.BayesianBlocksSolution}(undef, length(problems))
    priors_gpu = CuArray(_bb_prior_values(max_u))

    if verbose
        unique_counts = sort!(collect(length(p.prefix_counts) for p in problems))
        median_u = unique_counts[cld(length(unique_counts), 2)]
        println(
            "[FastPIDC] CUDA Bayesian blocks: $(length(problems)) genes, " *
            "U_g median=$median_u, max=$(unique_counts[end]), " *
            "prefix counts=$(CountT), back-pointers=$(IndexT)",
        )
        println(
            "[FastPIDC] CUDA Bayesian blocks memory budget: " *
            "$(round(budget_bytes / 2.0^30; digits = 2)) GiB",
        )
    end

    try
        for (bucket_number, bucket) in enumerate(buckets)
            bucket_max_u = maximum(i -> length(problems[i].prefix_counts), bucket)
            threads = _bb_threads_for_max_u(bucket_max_u)
            batches = _bb_memory_batches(
                bucket,
                problems,
                budget_bytes,
                CountT,
                IndexT,
            )

            if verbose
                bucket_min_u = minimum(i -> length(problems[i].prefix_counts), bucket)
                println(
                    "[FastPIDC] CUDA BB bucket $bucket_number/$(length(buckets)): " *
                    "$(length(bucket)) genes, U_g=$bucket_min_u:$bucket_max_u, " *
                    "threads=$threads, batches=$(length(batches))",
                )
            end

            for batch in batches
                batch_solutions = _solve_bb_cuda_batch_with_priors(
                    problems,
                    batch,
                    threads,
                    CountT,
                    IndexT,
                    priors_gpu,
                )
                for (problem_index, solution) in zip(batch, batch_solutions)
                    solutions[problem_index] = solution
                end
            end
        end
        return solutions
    finally
        CUDA.unsafe_free!(priors_gpu)
    end
end

end # module
