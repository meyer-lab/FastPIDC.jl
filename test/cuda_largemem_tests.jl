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
# The host plans the entire PUC device footprint against 65% of currently
# reusable memory. Require enough headroom for the >8 GiB counts buffer plus
# fixed arrays.
const _LARGEMEM_FREE_FLOOR = 14 * 2^30

_counts_elements(k_bins, n, chunk) = big(k_bins)^2 * n * chunk

if _LARGEMEM_ENABLED && CUDA.functional()
    @testset "PUC flat indexing past 2^31" begin
        cuda_ext = Base.get_extension(FastPIDC, :FastPIDCCUDAExt)
        @test cuda_ext !== nothing
        free_bytes = cuda_ext._available_gpu_memory_bytes()

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
            memory_plan = cuda_ext._puc_memory_plan(
                length(nodes),
                length(nodes[1].binned_values),
                k_bins,
                free_bytes,
            )
            chunk_size = memory_plan.chunk_size
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

# Cheap, always-run companion: all composed flat offsets are 64-bit now. The
# remaining ABI limit is that launch scalars themselves must fit signed Int32.
if CUDA.functional()
    @testset "Kernel scalar index limits" begin
        cuda_ext = Base.get_extension(FastPIDC, :FastPIDCCUDAExt)
        @test cuda_ext !== nothing

        # k_bins^2 may exceed Int32 because the bin-pair term is widened before
        # multiplication in joint_counts_kernel.
        @test cuda_ext._check_kernel_scalar_limits(12_071, 38_176, 50_000) === nothing
        @test big(50_000)^2 > big(typemax(Int32))

        @test_throws ArgumentError cuda_ext._check_kernel_scalar_limits(
            Int(typemax(Int32)) + 1,
            1,
            1,
        )
        @test_throws ArgumentError cuda_ext._check_kernel_scalar_limits(
            1,
            Int(typemax(Int32)) + 1,
            1,
        )
        @test_throws ArgumentError cuda_ext._check_kernel_scalar_limits(
            1,
            1,
            Int(typemax(Int32)) + 1,
        )
    end
end
