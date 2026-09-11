using FastPIDC
using Test
using CUDA

# Regression test for the 32-bit flat-index overflow that crashed a 12,071-gene
# x 38,176-cell run with CUDA_ERROR_ILLEGAL_ADDRESS. The shared kernels compute
# the joint-count index as k_bins^2 * n * chunk_size; at 33 bins, 12,071 genes
# and a 256-gene chunk that is 3,365,201,664 elements, whose top index wraps
# past typemax(Int32). All flat offsets are 64-bit now.
#
# The overflow cannot be reproduced at small sizes - the buffer must genuinely
# hold more than 2^31 Int32 elements, i.e. at least ~8 GiB of device memory - so
# this file is opt-in and skipped unless explicitly requested:
#
#     FASTPIDC_LARGEMEM_TESTS=1 julia --project=. -e 'using Pkg; Pkg.test()'
#
# Do not enable it on a shared GPU: it allocates ~8 GiB.

const _LARGEMEM_ENABLED = get(ENV, "FASTPIDC_LARGEMEM_TESTS", "0") == "1"

# 64 genes at 725 bins with a 64-gene chunk is the cheapest configuration that
# clears 2^31 counts elements: 8.02 GiB, versus ~23 MiB of everything else.
const _LARGEMEM_NODES = 64
const _LARGEMEM_SAMPLES = 2000
const _LARGEMEM_BINS = 725
# The host derives chunk_size from free memory; it only reaches 64 (rather than
# 63, which stays inside Int32) with roughly 11 GiB free, so require headroom.
const _LARGEMEM_FREE_FLOOR = 12 * 2^30

_counts_elements(k_bins, n, chunk) = big(k_bins)^2 * n * chunk

if _LARGEMEM_ENABLED && CUDA.functional()
    @testset "PUC flat indexing past 2^31" begin
        free_bytes = Int(CUDA.free_memory())

        if free_bytes < _LARGEMEM_FREE_FLOOR
            @warn "Skipping large-memory PUC index test" required_gib =
                _LARGEMEM_FREE_FLOOR / 2^30 free_gib = free_bytes / 2^30
        else
            values = [
                (sin(sample * 0.37 + gene) + 1) / 2 for sample = 1:_LARGEMEM_SAMPLES,
                gene = 1:_LARGEMEM_NODES
            ]
            nodes = [
                Node(
                    "N$gene",
                    values[:, gene],
                    "uniform_width",
                    "maximum_likelihood",
                    _LARGEMEM_BINS,
                ) for gene = 1:_LARGEMEM_NODES
            ]
            k_bins = maximum(node -> node.number_of_bins, nodes)

            # The host picks the chunk itself, so fail loudly rather than pass
            # vacuously if this configuration drifts below the threshold the
            # test exists to cross.
            cuda_ext = Base.get_extension(FastPIDC, :FastPIDCCUDAExt)
            @test cuda_ext !== nothing
            bytes_per_chunk_col =
                k_bins^2 * length(nodes) * sizeof(Int32) +
                k_bins * length(nodes) * sizeof(Float64)
            chunk_size = clamp(
                floor(Int, free_bytes * 0.8 / bytes_per_chunk_col),
                1,
                min(256, length(nodes)),
            )
            @test _counts_elements(k_bins, length(nodes), chunk_size) - 1 >
                  big(typemax(Int32))

            gpu_mi, gpu_puc = FastPIDC.compute_puc_full(
                nodes;
                estimator = "maximum_likelihood",
                base = 2,
                config = PIDCConfig(backend = :cuda),
            )
            cpu_mi, cpu_puc = FastPIDC.compute_puc_full(
                nodes;
                estimator = "maximum_likelihood",
                base = 2,
                config = PIDCConfig(backend = :cpu),
            )

            @test all(isfinite, gpu_mi)
            @test all(>=(0), gpu_puc)
            @test isapprox(gpu_mi, cpu_mi; atol = 1e-9, rtol = 1e-12)
            @test isapprox(gpu_puc, cpu_puc; atol = 1e-6, rtol = 1e-9)
        end
    end
end

# Cheap, always-run companion: the one kernel limit 64-bit indexing does not
# lift. joint_counts_kernel forms the bin-pair index u * k_bins + v in Int32.
if CUDA.functional()
    @testset "Kernel bin-pair index limit" begin
        cuda_ext = Base.get_extension(FastPIDC, :FastPIDCCUDAExt)
        @test cuda_ext !== nothing
        @test cuda_ext._MAX_K_BINS^2 <= typemax(Int32)
        @test big(cuda_ext._MAX_K_BINS + 1)^2 > big(typemax(Int32))
        @test cuda_ext._check_kernel_index_limits(cuda_ext._MAX_K_BINS) === nothing
        @test_throws ErrorException cuda_ext._check_kernel_index_limits(
            cuda_ext._MAX_K_BINS + 1,
        )
    end
end
