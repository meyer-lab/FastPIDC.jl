using FastPIDC
using Test
using CUDA

function _bb_test_values()
    values = Vector{Float64}[
        [
            0.0, 0.0, 0.0, 0.0,
            0.1, 0.1, 0.2, 0.2,
            3.0, 3.0, 3.1, 3.1,
            8.0, 8.1, 8.1, 8.2,
        ],
        [
            -5.0, -5.0, -4.9, -4.8,
            -0.2, -0.1, 0.0, 0.1,
            4.8, 4.9, 5.0, 5.0,
            12.0, 12.0, 12.1, 12.2,
        ],
        vcat(fill(0.0, 12), collect(0.5:0.5:8.0), fill(15.0, 12)),
        [Float64(i^2) / 17 for i = 0:39],
        fill(2.5, 32),
    ]

    # Add deterministic zero-inflated and clustered cases with varying U_g so
    # conformance exercises more than a few hand-selected partitions.
    for case = 1:16
        n = 48 + 3 * case
        x = [
            sin((sample + case) / 5) +
            0.2 * cos((2 * sample + case) / 7) +
            (sample % (5 + case % 4) == 0 ? 3.0 : 0.0) for sample = 1:n
        ]
        x[case:(4 + case % 5):end] .= 0.0
        push!(values, x)
    end
    return values
end

const _CUDA_EXT = Base.get_extension(FastPIDC, :FastPIDCCUDAExt)

@testset "CUDA memory planning (device-independent)" begin
    @test _CUDA_EXT !== nothing
    cuda_ext = _CUDA_EXT

    @testset "65% budget and reusable-memory arithmetic" begin
        @test cuda_ext._gpu_memory_budget_percent_label() ==
              "$(100 * cuda_ext._GPU_MEMORY_BUDGET_NUMERATOR ÷ cuda_ext._GPU_MEMORY_BUDGET_DENOMINATOR)%"
        @test cuda_ext._gpu_memory_budget_percent_label(80, 100) == "80%"
        @test cuda_ext._gpu_memory_budget_bytes(1000) == 650
        @test cuda_ext._gpu_memory_budget_bytes(1001) == 650
        @test_throws ArgumentError cuda_ext._gpu_memory_budget_bytes(0)
        @test_throws ArgumentError cuda_ext._gpu_memory_budget_bytes(-1)

        @test cuda_ext._reusable_gpu_memory_bytes(1000, 400, 250) == 1150
        @test_throws ArgumentError cuda_ext._reusable_gpu_memory_bytes(100, 50, 51)
    end

    @testset "Bayesian-block 65% batch planner" begin
        values = Float64.(1:50)
        problem = FastPIDC.prepare_bayesian_blocks(values)
        problems = [problem for _ = 1:6]
        bucket = collect(eachindex(problems))
        per_problem = cuda_ext._bb_problem_bytes(problem, UInt16, UInt16)
        u = length(problem.prefix_counts)
        @test per_problem ==
              sizeof(Float64) * (u + 1) +
              sizeof(UInt16) * u +
              sizeof(Float64) * u +
              sizeof(UInt16) * u +
              sizeof(Int64) +
              sizeof(Int64) +
              sizeof(Int32) +
              sizeof(Float64)

        # Prefix counts and back-pointers must widen before their values overflow.
        @test cuda_ext._smallest_unsigned_type(255) == UInt8
        @test cuda_ext._smallest_unsigned_type(256) == UInt16
        @test cuda_ext._smallest_unsigned_type(65_535) == UInt16
        @test cuda_ext._smallest_unsigned_type(65_536) == UInt32
        @test cuda_ext._smallest_unsigned_type(big(typemax(UInt32)) + 1) == UInt64

        # Pick reusable memory whose 65% budget is exactly two problems. The
        # resulting batches must therefore contain exactly two genes each.
        target_budget = 2 * per_problem
        free_bytes = cld(target_budget * 100, 65)
        plan = cuda_ext._bb_memory_plan(
            bucket,
            problems,
            free_bytes,
            UInt16,
            UInt16,
        )

        @test plan.budget_bytes == target_budget
        @test plan.batches == [[1, 2], [3, 4], [5, 6]]
        for batch in plan.batches
            batch_bytes = sum(
                i -> cuda_ext._bb_problem_bytes(problems[i], UInt16, UInt16),
                batch,
            )
            @test batch_bytes <= plan.budget_bytes
        end

        # One problem requiring more than the 65% budget is rejected before any
        # device allocation is attempted.
        @test_throws ArgumentError cuda_ext._bb_memory_plan(
            [1],
            [problem],
            per_problem,
            UInt16,
            UInt16,
        )
    end

end

if CUDA.functional()
    @testset "CUDA Bayesian blocks equivalence and determinism" begin
        cuda_ext = Base.get_extension(FastPIDC, :FastPIDCCUDAExt)
        @test cuda_ext !== nothing

        values_by_gene = _bb_test_values()
        problems = FastPIDC.prepare_bayesian_blocks.(values_by_gene)
        cpu_solutions = FastPIDC.solve_bayesian_blocks_cpu.(problems)
        gpu_solutions_1 = FastPIDC.solve_bayesian_blocks_cuda(problems, false)
        gpu_solutions_2 = FastPIDC.solve_bayesian_blocks_cuda(problems, false)

        @test gpu_solutions_1 !== nothing
        @test gpu_solutions_2 !== nothing

        @testset "CPU vs GPU Bayesian-block result" begin
            for i = eachindex(problems)
                cpu_solution = cpu_solutions[i]
                gpu_solution = gpu_solutions_1[i]

                # CUDA and CPU libdevice/libm logarithms may differ by a few
                # bits, but robust test cases should retain the same optimum.
                @test isapprox(
                    gpu_solution.score,
                    cpu_solution.score;
                    atol = 1e-9,
                    rtol = 1e-12,
                )
                @test gpu_solution.change_points == cpu_solution.change_points

                cpu_edges = problems[i].edges[cpu_solution.change_points]
                gpu_edges = problems[i].edges[gpu_solution.change_points]
                @test gpu_edges == cpu_edges

                # A constant gene has one unique value, so its two outer
                # Bayesian-block edges are equal. The CPU and CUDA solvers can
                # still be compared above, but a zero-width interval is not a
                # valid LinearDiscretizer and therefore has no encoding step.
                if length(problems[i].prefix_counts) == 1
                    @test all(==(only(unique(values_by_gene[i]))), values_by_gene[i])
                    continue
                end

                cpu_ids = FastPIDC.encode(
                    FastPIDC.LinearDiscretizer(cpu_edges),
                    values_by_gene[i],
                )
                gpu_ids = FastPIDC.encode(
                    FastPIDC.LinearDiscretizer(gpu_edges),
                    values_by_gene[i],
                )
                @test gpu_ids == cpu_ids
            end
        end

        @testset "CUDA Bayesian blocks determinism" begin
            for i = eachindex(gpu_solutions_1)
                @test gpu_solutions_1[i].change_points ==
                      gpu_solutions_2[i].change_points
                @test reinterpret(UInt64, [gpu_solutions_1[i].score]) ==
                      reinterpret(UInt64, [gpu_solutions_2[i].score])
            end

            # Candidate assignment changes with block size, but the ordered
            # first-maximum reduction must produce the same deterministic DP.
            all_indices = collect(eachindex(problems))
            solutions_32 = cuda_ext._solve_bb_cuda_batch(
                problems,
                all_indices,
                32,
                UInt8,
                UInt8,
            )
            solutions_64 = cuda_ext._solve_bb_cuda_batch(
                problems,
                all_indices,
                64,
                UInt8,
                UInt8,
            )
            for i = eachindex(problems)
                @test solutions_32[i].change_points == solutions_64[i].change_points
                @test reinterpret(UInt64, [solutions_32[i].score]) ==
                      reinterpret(UInt64, [solutions_64[i].score])
            end

            # Compile and exercise the UInt16 prefix-count/back-pointer kernel
            # variant used when either cells or U_g exceed UInt8 capacity.
            wide_values = [Float64(i^2) / 101 for i = 0:299]
            wide_problem = FastPIDC.prepare_bayesian_blocks(wide_values)
            wide_cpu = FastPIDC.solve_bayesian_blocks_cpu(wide_problem)
            wide_gpu = only(
                cuda_ext._solve_bb_cuda_batch(
                    [wide_problem],
                    [1],
                    64,
                    UInt16,
                    UInt16,
                ),
            )
            @test wide_gpu.change_points == wide_cpu.change_points
            @test isapprox(wide_gpu.score, wide_cpu.score; atol = 1e-9, rtol = 1e-12)

            @test_throws ArgumentError cuda_ext._solve_bb_cuda_batch(
                problems,
                all_indices,
                48,
                UInt8,
                UInt8,
            )

            reversed_solutions = FastPIDC.solve_bayesian_blocks_cuda(
                reverse(problems),
                false,
            )
            @test reverse([s.change_points for s in reversed_solutions]) ==
                  [s.change_points for s in gpu_solutions_1]
        end

        @testset "Bayesian-block backtracking edge cases" begin
            # The shared CUDA C kernel writes 0-based predecessor indices, so
            # these raw back-pointer arrays use the device encoding: a valid
            # entry for state `s` lies in 0:(s-1). The resulting 1-based edge
            # paths are unchanged.
            @test cuda_ext._change_points_from_last(UInt8[0], 1, 1) ==
                  Int64[1, 2]
            @test cuda_ext._change_points_from_last(UInt8[0, 1], 1, 2) ==
                  Int64[1, 2, 3]
            @test cuda_ext._change_points_from_last(UInt8[0, 0], 1, 2) ==
                  Int64[1, 3]
            # A back-pointer that points forward past its own state.
            @test_throws ArgumentError cuda_ext._change_points_from_last(
                UInt8[0, 2],
                1,
                2,
            )
        end

        @testset "Shared kernel entry point selection" begin
            @test cuda_ext._bb_kernel_name(UInt8, UInt8) ==
                  "bayesian_blocks_dp_u8_u8"
            @test cuda_ext._bb_kernel_name(UInt32, UInt16) ==
                  "bayesian_blocks_dp_u32_u16"
            @test cuda_ext._bb_kernel_name(UInt64, UInt64) ==
                  "bayesian_blocks_dp_u64_u64"
            # U_g never exceeds the observation count, so a back-pointer type
            # wider than the prefix-count type is not instantiated.
            @test_throws ArgumentError cuda_ext._bb_kernel_name(UInt8, UInt16)
            @test_throws ArgumentError cuda_ext._bb_kernel_name(Int32, UInt8)

            # Every advertised entry point must exist in the compiled module.
            md = cuda_ext._get_module()
            for count_type in (UInt8, UInt16, UInt32, UInt64),
                index_type in (UInt8, UInt16, UInt32, UInt64)

                sizeof(index_type) <= sizeof(count_type) || continue
                name = cuda_ext._bb_kernel_name(count_type, index_type)
                @test CUDA.CuFunction(md, name) isa CUDA.CuFunction
            end
        end

        @testset "GPU memory accounting" begin
            @test cuda_ext._gpu_memory_budget_bytes(1000) == 650
            @test cuda_ext._reusable_gpu_memory_bytes(1000, 400, 250) == 1150
            @test_throws ArgumentError cuda_ext._reusable_gpu_memory_bytes(100, 50, 51)
        end

        @testset "Lightweight U_g bucketing and batching" begin
            buckets = cuda_ext._bb_quantile_buckets(problems)
            @test sort(vcat(buckets...)) == collect(eachindex(problems))
            @test cuda_ext._bb_threads_for_max_u(32) == 32
            @test cuda_ext._bb_threads_for_max_u(33) == 64
            @test cuda_ext._bb_threads_for_max_u(513) == 128
            @test cuda_ext._bb_threads_for_max_u(4_097) == 256

            # The batching helper requires a concrete byte count. Keep this
            # focused check so an accidental bare `return` cannot silently
            # turn the estimate into `nothing` again.
            u = length(problems[1].prefix_counts)
            expected_bytes =
                sizeof(Float64) * (u + 1) +
                sizeof(UInt8) * u +
                sizeof(Float64) * u +
                sizeof(UInt8) * u +
                sizeof(Int64) +
                sizeof(Int64) +
                sizeof(Int32) +
                sizeof(Float64)
            @test cuda_ext._bb_problem_bytes(problems[1], UInt8, UInt8) ==
                  expected_bytes

            batches = cuda_ext._bb_memory_batches(
                buckets[1],
                problems,
                typemax(Int),
                UInt8,
                UInt8,
            )
            @test batches == [buckets[1]]
        end

        @testset "End-to-end nodes and PUC integrity" begin
            # Use equal-length columns so both backends receive identical
            # observations through the same batched node-construction path.
            matrix = hcat(
                repeat(values_by_gene[1], 3),
                repeat(values_by_gene[2], 3),
                vcat(fill(0.0, 16), collect(0.5:0.5:8.0), fill(15.0, 16)),
                vcat(values_by_gene[4], values_by_gene[4][1:8]),
            )
            labels = ["G1", "G2", "G3", "G4"]
            value_at = i -> (@view matrix[:, i])

            nodes_cpu = FastPIDC._build_nodes(
                labels,
                value_at;
                discretizer = "bayesian_blocks",
                estimator = "maximum_likelihood",
                number_of_bins = 10,
                bb_backend = :cpu,
                verbose = false,
            )
            nodes_gpu = FastPIDC._build_nodes(
                labels,
                value_at;
                discretizer = "bayesian_blocks",
                estimator = "maximum_likelihood",
                number_of_bins = 10,
                bb_backend = :cuda,
                verbose = false,
            )

            for i = eachindex(nodes_cpu)
                @test nodes_gpu[i].number_of_bins == nodes_cpu[i].number_of_bins
                @test nodes_gpu[i].binned_values == nodes_cpu[i].binned_values
                @test nodes_gpu[i].probabilities == nodes_cpu[i].probabilities
            end

            # Exercise the public loader and its independent bb_backend option.
            mktemp() do path, io
                println(io, join(vcat("gene", ["S$i" for i = 1:size(matrix, 1)]), " "))
                for gene = 1:size(matrix, 2)
                    println(io, join(vcat(labels[gene], string.(matrix[:, gene])), " "))
                end
                close(io)

                public_cpu = get_nodes(path; bb_backend = :cpu)
                public_gpu = get_nodes(path; bb_backend = :cuda)
                for i = eachindex(public_cpu)
                    @test public_gpu[i].number_of_bins == public_cpu[i].number_of_bins
                    @test public_gpu[i].binned_values == public_cpu[i].binned_values
                end
            end

            config_cuda = PIDCConfig(backend = :cuda, bb_backend = :cuda)
            mi_cpu_bins, puc_cpu_bins =
                FastPIDC.compute_puc_full_cuda(nodes_cpu, config_cuda, 2)
            mi_gpu_bins, puc_gpu_bins =
                FastPIDC.compute_puc_full_cuda(nodes_gpu, config_cuda, 2)
            @test mi_gpu_bins == mi_cpu_bins
            @test puc_gpu_bins == puc_cpu_bins
        end
    end
else
    @warn "CUDA unavailable. Skipping CUDA Bayesian-block tests."
end
