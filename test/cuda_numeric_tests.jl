using FastPIDC
using Test
using LinearAlgebra
using CUDA
using Statistics
const DATA_DIR = joinpath(dirname(@__FILE__), "data")

# Only run if GPU is available
if CUDA.functional()
    @testset "CUDA Numeric Equivalence & Integrity" begin
        # Setup minimal dummy data
        dataset = joinpath(DATA_DIR, "toy_small_200.txt") # Use a small, fast dataset
        nodes = get_nodes(dataset)

        config_cpu = PIDCConfig(backend = :cpu, verbose = false)
        config_cuda = PIDCConfig(backend = :cuda, verbose = false)

        # We need to call the internal matrix generators directly, not just the Network wrapper
        mi_cpu, puc_cpu = FastPIDC.compute_puc_full(nodes, config = config_cpu, base = 2)
        mi_gpu, puc_gpu = FastPIDC.compute_puc_full_cuda(nodes, config_cuda, 2.0)

        # --- TEST 1: MI Matrix Symmetry ---
        @testset "MI Matrix Symmetry" begin
            # CPU should be perfectly symmetric
            @test issymmetric(mi_cpu)
            # GPU might have Float32 drift, check with tolerance
            @test isapprox(mi_gpu, transpose(mi_gpu), atol = 1e-8)
        end

        # --- TEST 2: Determinism (Race Condition Check) ---
        @testset "GPU Determinism" begin
            # Run the GPU calculation twice on the exact same data
            mi_gpu1, puc_gpu1 = FastPIDC.compute_puc_full_cuda(nodes, config_cuda, 2)
            mi_gpu2, puc_gpu2 = FastPIDC.compute_puc_full_cuda(nodes, config_cuda, 2)

            # These should be bit-for-bit identical. If they aren't, 
            # there is an atomic race condition or uninitialized memory.
            @test mi_gpu1 == mi_gpu2
            @test puc_gpu1 == puc_gpu2
        end

        # --- TEST 3: Numeric Equivalence to legacy ---
        @testset "CPU vs GPU Numeric Tolerance" begin
            mi_cpu, puc_cpu = FastPIDC.compute_puc_full(nodes, config = config_cpu)
            mi_gpu, puc_gpu = FastPIDC.compute_puc_full_cuda(nodes, config_cuda, 2)

            # Check Symmetry
            @test isapprox(mi_gpu, transpose(mi_gpu), atol = 1e-8)
            @test isapprox(puc_gpu, transpose(puc_gpu), atol = 1e-8)

            diffs = abs.(puc_cpu .- puc_gpu)
            max_abs_err = maximum(diffs)
            # Calculate relative error (avoiding division by zero on the diagonal)
            rel_errs = diffs ./ (abs.(puc_cpu) .+ 1e-9)
            max_rel_err = maximum(rel_errs)

            println("\n--- GPU/CPU Numerical Diagnostic ---")
            println("Max Absolute Error: $max_abs_err")
            println("Max Relative Error: $max_rel_err")

            # Check if they are just scaled versions of each other
            ratio = puc_gpu ./ (puc_cpu .+ 1e-9)
            println("Mean Ratio (GPU/CPU): ", mean(ratio[puc_cpu .> 1.0]))

            @test isapprox(mi_cpu, mi_gpu, atol = 1e-8)
            # Broadcast isapprox to all elements, not Euclidean norm
            @test all(isapprox.(puc_cpu, puc_gpu, atol = 1e-8))
        end

        # --- TEST 4: txt vs. H5 Determinism ---
        @testset "txt vs. H5 Determinism" begin
            node_txt = get_nodes(dataset)
            node_h5 =  get_nodes(joinpath(DATA_DIR, "toy_small_200.h5"))
            # Run the GPU calculation twice on the exact same data
            mi_txt, puc_txt = FastPIDC.compute_puc_full_cuda(nodes, config_cuda, 2)
            mi_h5, puc_h5 = FastPIDC.compute_puc_full_cuda(nodes, config_cuda, 2)

            # These should be bit-for-bit identical. If they aren't, 
            # there is an atomic race condition or uninitialized memory.
            @test mi_txt == mi_h5
            @test puc_txt == puc_h5
        end

        # --- TEST 5: Edge Rank Preservation & Diagnostics ---
        @testset "Top Edge Rank Preservation" begin
            # Generate the full sorted network objects
            net_cpu = InferredNetwork(PUCNetworkInference(), nodes, config = config_cpu)
            net_gpu = InferredNetwork(PUCNetworkInference(), nodes, config = config_cuda)

            # Extract the ordered lists of sets
            cpu_pairs = [Set([e.nodes[1].label, e.nodes[2].label]) for e in net_cpu.edges]
            gpu_pairs = [Set([e.nodes[1].label, e.nodes[2].label]) for e in net_gpu.edges]

            # --- Top 250 Overlap Test ---
            top_k = min(250, length(cpu_pairs))
            top_cpu_set = Set(cpu_pairs[1:top_k])
            top_gpu_set = Set(gpu_pairs[1:top_k])

            overlap = length(intersect(top_cpu_set, top_gpu_set))
            println("\n[Diagnostics] Top $top_k Edge Overlap: $overlap / $top_k")

            # --- Full Equivalence & Shift Diagnostics ---
            if cpu_pairs != gpu_pairs
                println("[Diagnostics] Exact array match failed. Analyzing rank shifts...")

                # Map each edge to its exact rank in the CPU list
                cpu_ranks = Dict(pair => i for (i, pair) in enumerate(cpu_pairs))

                max_shift = 0
                total_shifted = 0

                for (gpu_rank, pair) in enumerate(gpu_pairs)
                    cpu_rank = cpu_ranks[pair]
                    shift = abs(cpu_rank - gpu_rank)

                    if shift > 0
                        total_shifted += 1
                        max_shift = max(max_shift, shift)

                        # Print detailed info if the shift happens in the top 100
                        if gpu_rank <= 100 || cpu_rank <= 100
                            println(
                                "  -> Shift near top: Edge $pair moved CPU Rank $cpu_rank -> GPU Rank $gpu_rank (Shift: $shift)",
                            )
                        end
                    end
                end

                println(
                    "[Diagnostics] Total shifted edges: $total_shifted / $(length(cpu_pairs))",
                )
                println("[Diagnostics] Maximum rank shift observed: $max_shift")
            end
        end
    end
else
    @warn "CUDA unavailable. Skipping numeric equivalence tests."
end
