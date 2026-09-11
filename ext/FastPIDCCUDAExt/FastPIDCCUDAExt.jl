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

const _KERNEL_INT_MAX = typemax(Int32)
const _GPU_MEMORY_BUDGET_NUMERATOR = 65
const _GPU_MEMORY_BUDGET_DENOMINATOR = 100
const _MAX_CHUNK_SIZE = 256

function _gpu_memory_budget_percent_label(
    numerator_value::Integer = _GPU_MEMORY_BUDGET_NUMERATOR,
    denominator_value::Integer = _GPU_MEMORY_BUDGET_DENOMINATOR,
)
    denominator_value > 0 || throw(ArgumentError("denominator_value must be positive"))
    percent = 100 * numerator_value // denominator_value
    return denominator(percent) == 1 ?
           "$(numerator(percent))%" : "$(round(Float64(percent), digits = 2))%"
end

function _gpu_memory_budget_bytes(free_bytes::Integer)
    free_bytes > 0 || throw(ArgumentError("free_bytes must be positive"))
    return Int(
        div(
            big(free_bytes) * _GPU_MEMORY_BUDGET_NUMERATOR,
            _GPU_MEMORY_BUDGET_DENOMINATOR,
        ),
    )
end

function _reusable_gpu_memory_bytes(
    driver_free_bytes::Integer,
    pool_cached_bytes::Integer = 0,
    pool_used_bytes::Integer = 0,
)
    driver_free_bytes >= 0 || throw(ArgumentError("driver_free_bytes must be nonnegative"))
    pool_cached_bytes >= 0 || throw(ArgumentError("pool_cached_bytes must be nonnegative"))
    pool_used_bytes >= 0 || throw(ArgumentError("pool_used_bytes must be nonnegative"))
    pool_used_bytes <= pool_cached_bytes || throw(
        ArgumentError("pool_used_bytes cannot exceed pool_cached_bytes"),
    )

    # CUDA.free_memory() reports memory still free at the driver level. CUDA.jl's
    # stream-ordered pool may additionally hold reserved-but-unused bytes that are
    # immediately reusable by this process, so include those in the planning view.
    return Int(big(driver_free_bytes) + big(pool_cached_bytes - pool_used_bytes))
end

function _available_gpu_memory_bytes()
    driver_free_bytes = Int(CUDA.free_memory())

    # CUDA.jl 5.4+ exposes pool accounting helpers. On allocators/devices that do
    # not use the stream-ordered pool they return `missing`, in which case driver
    # free memory is already the only reusable pool we can account for.
    cached = CUDA.cached_memory()
    used = CUDA.used_memory()
    if cached isa Integer && used isa Integer
        return _reusable_gpu_memory_bytes(driver_free_bytes, Int(cached), Int(used))
    end
    return driver_free_bytes
end

"""
    _check_kernel_scalar_limits(num_nodes, num_samples, k_bins)

The shared CUDA kernels receive launch dimensions as signed 32-bit integers.
All composed flat-buffer offsets are 64-bit, but these scalar dimensions must
still fit exactly in `Int32` before crossing the host/device boundary.
"""
function _check_kernel_scalar_limits(
    num_nodes::Integer,
    num_samples::Integer,
    k_bins::Integer,
)
    for (name, value) in (
        ("num_nodes", num_nodes),
        ("num_samples", num_samples),
        ("k_bins", k_bins),
    )
        value >= 1 || throw(ArgumentError("$name must be positive; got $value"))
        value <= _KERNEL_INT_MAX || throw(
            ArgumentError(
                "compute_puc_full_cuda: $name=$value exceeds the CUDA kernel's " *
                "signed 32-bit scalar limit of $(_KERNEL_INT_MAX).",
            ),
        )
    end
    return nothing
end

function _puc_memory_plan(
    num_nodes::Integer,
    num_samples::Integer,
    k_bins::Integer,
    free_bytes::Integer,
)
    # Use BigInt for the planning arithmetic so an impossible input cannot
    # overflow on the host while we are trying to decide whether it fits.
    n = big(num_nodes)
    m = big(num_samples)
    k = big(k_bins)
    budget_bytes = big(_gpu_memory_budget_bytes(free_bytes))

    fixed_bytes =
        n * m * sizeof(Int32) +       # discretized data
        n * k * sizeof(Float64) +     # marginals
        2 * n * n * sizeof(Float64)  # MI + PUC output matrices
    bytes_per_chunk_col =
        k * k * n * sizeof(Int32) +  # joint counts
        k * n * sizeof(Float64)      # specific information

    minimum_bytes = fixed_bytes + bytes_per_chunk_col
    minimum_bytes <= budget_bytes || error(
        "compute_puc_full_cuda: the fixed GPU buffers plus a one-gene chunk " *
        "would require $(round(Float64(minimum_bytes) / 2^30, digits = 2)) GiB, " *
        "which exceeds the configured $(_gpu_memory_budget_percent_label()) memory budget " *
        "($(round(Float64(budget_bytes) / 2^30, digits = 2)) GiB of " *
        "$(round(free_bytes / 2^30, digits = 2)) GiB currently reusable). " *
        "Reduce the number of genes/samples/bins, use a fixed small-bin " *
        "discretizer, or use config.backend = :cpu.",
    )

    max_chunk_size = div(budget_bytes - fixed_bytes, bytes_per_chunk_col)
    chunk_size = min(max_chunk_size, big(_MAX_CHUNK_SIZE), n)

    return (
        chunk_size = Int(chunk_size),
        fixed_bytes = Int(fixed_bytes),
        bytes_per_chunk_col = Int(bytes_per_chunk_col),
        budget_bytes = Int(budget_bytes),
    )
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
pairwise MI matrix and pre-context PUC matrix for `nodes` on the GPU.
Before allocating device buffers it plans the complete footprint - fixed data,
marginals and output matrices plus chunked intermediates - against the configured
fraction of memory currently reusable by the process (driver-free plus unused
CUDA.jl pool blocks). The target (`z`) chunk is capped at 256 genes and shrunk
as needed, leaving the remaining headroom for allocator fragmentation, CUDA/runtime
workspaces and concurrent users. If the fixed buffers plus a one-gene chunk do
not fit that budget, the function raises a descriptive error before allocating
the large device arrays.

All composed flat-buffer offsets in the shared CUDA kernels are 64-bit,
including the bin-pair term. The scalar launch dimensions remain signed
32-bit for ABI compatibility and are validated on the host before conversion.
`config.verbose` prints the memory plan; `base` is currently unused (mutual
information is always computed in base 2 on the GPU, matching the kernel
source).

Device buffers use Julia's column-major layout with dimensions reversed
relative to the kernel source's documented (row-major) shapes - e.g. a
kernel-documented `(k_bins, n)` array is allocated here with Julia size
`(n, k_bins)` - so that the flat in-memory layout the kernels index into
with manual pointer arithmetic is identical in both languages, with no
transposition needed at the call boundary.
"""
function FastPIDC.compute_puc_full_cuda(nodes, config, base)
    isempty(nodes) && throw(ArgumentError("compute_puc_full_cuda requires at least one node"))

    # Compile/load the module before measuring free memory so its device-side
    # footprint is already reflected in the runtime memory budget.
    md = _get_module()
    joint_counts_kernel = CuFunction(md, "joint_counts_kernel")
    mi_si_kernel = CuFunction(md, "mi_si_kernel")
    puc_accumulation_kernel = CuFunction(md, "puc_accumulation_kernel")

    num_nodes = length(nodes)
    num_samples = length(nodes[1].binned_values)
    all(n -> length(n.binned_values) == num_samples, nodes) || throw(
        ArgumentError("all nodes must contain the same number of discretized samples"),
    )
    k_bins = maximum(n -> n.number_of_bins, nodes)
    _check_kernel_scalar_limits(num_nodes, num_samples, k_bins)

    # Plan the *entire* device footprint before allocating anything. The budget
    # is the configured fraction of currently reusable VRAM (driver-free plus unused
    # CUDA.jl pool blocks), leaving the remainder for allocator fragmentation,
    # runtime/library workspaces, and other users/processes on a shared GPU.
    free_bytes = _available_gpu_memory_bytes()
    memory_plan = _puc_memory_plan(num_nodes, num_samples, k_bins, free_bytes)
    chunk_size = memory_plan.chunk_size

    # Prepare static data on CPU. The shared CUDA C kernels use 0-indexed Int32
    # bin ids; FastPIDC.jl's bin ids are 1-indexed, so shift them down at this
    # boundary.
    data_cpu = zeros(Int32, num_nodes, num_samples)          # kernel shape (m, n), reversed
    marginals_cpu = zeros(Float64, num_nodes, k_bins)       # kernel shape (k_bins, n), reversed
    for i = 1:num_nodes
        node = nodes[i]
        node.number_of_bins >= 1 || throw(
            ArgumentError("node $(node.label) has non-positive number_of_bins"),
        )
        maximum(node.binned_values) <= node.number_of_bins || throw(
            ArgumentError("node $(node.label) contains a bin id above number_of_bins"),
        )
        minimum(node.binned_values) >= 1 || throw(
            ArgumentError("node $(node.label) contains a bin id below 1"),
        )

        data_cpu[i, :] .= Int32.(node.binned_values) .- Int32(1)
        p = node.probabilities
        length(p) <= k_bins || throw(
            ArgumentError("node $(node.label) has more probabilities than k_bins"),
        )
        marginals_cpu[i, 1:length(p)] .= Float64.(p)
    end

    data_gpu = nothing
    marginals_gpu = nothing
    puc_scores_gpu = nothing
    mi_matrix_gpu = nothing
    counts_chunk_gpu = nothing
    si_chunk_gpu = nothing

    try
        data_gpu = CuArray(data_cpu)
        marginals_gpu = CuArray(marginals_cpu)

        # Global output matrices (kernel shape (n, n); square, so no reversal needed).
        puc_scores_gpu = CUDA.zeros(Float64, num_nodes, num_nodes)
        mi_matrix_gpu = CUDA.zeros(Float64, num_nodes, num_nodes)

        # Chunked intermediate buffers, pre-allocated once. Dimensions are
        # reversed to preserve the row-major flat layout expected by CUDA C.
        counts_chunk_gpu = CUDA.zeros(Int32, chunk_size, num_nodes, k_bins, k_bins)
        si_chunk_gpu = CUDA.zeros(Float64, chunk_size, num_nodes, k_bins)

        if config.verbose
            println(
                "[FastPIDC] GPU Chunked PUC: Processing $num_nodes x $num_nodes pairs " *
                "(k_bins=$k_bins)...",
            )
            println(
                "[FastPIDC] GPU memory: $(round(free_bytes / 2^30, digits = 2)) GiB reusable; " *
                "$(_gpu_memory_budget_percent_label()) budget=$(round(memory_plan.budget_bytes / 2^30, digits = 2)) GiB; " *
                "fixed=$(round(memory_plan.fixed_bytes / 2^30, digits = 2)) GiB",
            )
            println(
                "[FastPIDC] Using chunk size of $chunk_size " *
                "(approx. $(ceil(Int, num_nodes / chunk_size)) iterations)",
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

        # Kernels write row-major (x, z) into Julia arrays whose dimensions are
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
    finally
        # Return allocations to CUDA.jl's pool immediately, including on a kernel
        # error. This prevents a failed or repeated run from retaining pressure
        # until Julia's GC notices the arrays.
        for array in (
            counts_chunk_gpu,
            si_chunk_gpu,
            puc_scores_gpu,
            mi_matrix_gpu,
            marginals_gpu,
            data_gpu,
        )
            array === nothing || CUDA.unsafe_free!(array)
        end
    end
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
        sizeof(IndexT) * u +        # back-pointers
        sizeof(Int64) +             # state offset
        sizeof(Int64) +             # block offset
        sizeof(Int32) +             # unique count
        sizeof(Float64)             # final score
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

function _bb_memory_plan(
    bucket::Vector{Int},
    problems::Vector{FastPIDC.BayesianBlocksProblem},
    free_bytes::Integer,
    ::Type{CountT},
    ::Type{IndexT},
) where {CountT<:Integer,IndexT<:Integer}
    budget_bytes = _gpu_memory_budget_bytes(free_bytes)
    batches = _bb_memory_batches(
        bucket,
        problems,
        budget_bytes,
        CountT,
        IndexT,
    )
    return (batches = batches, budget_bytes = budget_bytes)
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
    length(problem_indices) <= _KERNEL_INT_MAX || throw(
        ArgumentError(
            "CUDA Bayesian blocks batch has $(length(problem_indices)) genes, " *
            "which exceeds the signed 32-bit block-index limit.",
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

    # Load the module before measuring reusable memory so kernel/module residency
    # is already reflected in the driver's and CUDA.jl pool's accounting.
    _get_module()

    sample_count = maximum(p -> Int(round(p.prefix_counts[end])), problems)
    max_u = maximum(p -> length(p.prefix_counts), problems)
    max_u <= _KERNEL_INT_MAX || throw(
        ArgumentError(
            "Bayesian blocks CUDA backend supports at most $(_KERNEL_INT_MAX) " *
            "unique values per gene; got $max_u",
        ),
    )

    # A cumulative prefix count can reach the number of cells, so select the
    # smallest exact unsigned type that guards against overflow for this input.
    CountT = _smallest_unsigned_type(sample_count)
    # Back-pointers only need to represent candidate indices up to U_g.
    IndexT = _smallest_unsigned_type(max_u)

    buckets = _bb_quantile_buckets(problems)
    solutions = Vector{FastPIDC.BayesianBlocksSolution}(undef, length(problems))

    # The priors remain live across all batches. Allocate them first, then size
    # each bucket from the *remaining* reusable memory at runtime. Every batch is
    # kept within the configured fraction of what is free at that moment, and _bb_problem_bytes
    # includes all per-gene device metadata, not just the large state arrays.
    priors_gpu = CuArray(_bb_prior_values(max_u))

    if verbose
        unique_counts = sort!(collect(length(p.prefix_counts) for p in problems))
        median_u = unique_counts[cld(length(unique_counts), 2)]
        println(
            "[FastPIDC] CUDA Bayesian blocks: $(length(problems)) genes, " *
            "U_g median=$median_u, max=$(unique_counts[end]), " *
            "prefix counts=$(CountT), back-pointers=$(IndexT)",
        )
    end

    try
        for (bucket_number, bucket) in enumerate(buckets)
            bucket_max_u = maximum(i -> length(problems[i].prefix_counts), bucket)
            threads = _bb_threads_for_max_u(bucket_max_u)

            free_bytes = _available_gpu_memory_bytes()
            memory_plan = _bb_memory_plan(
                bucket,
                problems,
                free_bytes,
                CountT,
                IndexT,
            )
            budget_bytes = memory_plan.budget_bytes
            batches = memory_plan.batches

            if verbose
                bucket_min_u = minimum(i -> length(problems[i].prefix_counts), bucket)
                println(
                    "[FastPIDC] CUDA BB bucket $bucket_number/$(length(buckets)): " *
                    "$(length(bucket)) genes, U_g=$bucket_min_u:$bucket_max_u, " *
                    "threads=$threads, batches=$(length(batches)), " *
                    "reusable=$(round(free_bytes / 2.0^30; digits = 2)) GiB, " *
                    "$(_gpu_memory_budget_percent_label()) budget=$(round(budget_bytes / 2.0^30; digits = 2)) GiB",
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
