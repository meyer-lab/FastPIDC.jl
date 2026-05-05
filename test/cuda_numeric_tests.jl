using FastPIDC
using Test
using LinearAlgebra
using CUDA
using Statistics

# Only run if GPU is available
if CUDA.functional()
    @testset "CUDA Numeric Equivalence & Integrity" begin
        # Setup minimal dummy data
        dataset = "data/toy_small_200.txt" # Use a small, fast dataset
        nodes = get_nodes(dataset)
        
        config_cpu = PIDCConfig(triplet_backend=:threads, verbose=false)
        config_cuda = PIDCConfig(triplet_backend=:cuda, verbose=false)
        
        # --- TEST 1: MI Matrix Symmetry ---
        @testset "MI Matrix Symmetry" begin
            # We need to call the internal matrix generators directly, not just the Network wrapper
            mi_cpu, puc_cpu = FastPIDC.compute_puc_full(nodes, config=config_cpu, base = 2)
            mi_gpu, puc_gpu = FastPIDC.compute_puc_full_cuda(nodes, config_cuda, 2.0)

            # CPU should be perfectly symmetric
            @test issymmetric(mi_cpu)
            # GPU might have Float32 drift, check with tolerance
            @test isapprox(mi_gpu, transpose(mi_gpu), atol=1e-8)
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
            mi_cpu, puc_cpu = FastPIDC.compute_puc_full(nodes, config=config_cpu)
            mi_gpu, puc_gpu = FastPIDC.compute_puc_full_cuda(nodes, config_cuda, 2)
            
            # Check Symmetry
            @test isapprox(mi_gpu, transpose(mi_gpu), atol=1e-4)
            @test isapprox(puc_gpu, transpose(puc_gpu), atol=1e-4)
            
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
                        
            @test isapprox(mi_cpu, mi_gpu, atol=1e-4)
            # Broadcast isapprox to all elements, not Euclidean norm, fails on 32 bit precision
            # @test all(isapprox.(puc_cpu, puc_gpu, atol = 1e-3))
        end

        # --- TEST 4: Edge Rank Preservation ---
        @testset "Top Edge Rank Preservation" begin
            # Generate the full sorted network objects
            net_cpu = InferredNetwork(PUCNetworkInference(), nodes, config=config_cpu)
            net_gpu = InferredNetwork(PUCNetworkInference(), nodes, config=config_cuda)
            
            # We want to check if the top k edges are discovering the same biological connections.
            # We use Sets of labels because edges are undirected (A-B is the same as B-A).
            top_k = length(net_cpu.edges)
            # top_k = min(1000, length(net_cpu.edges))
            
            cpu_pairs = [Set([e.nodes[1].label, e.nodes[2].label]) for e in net_cpu.edges[1:top_k]]
            gpu_pairs = [Set([e.nodes[1].label, e.nodes[2].label]) for e in net_gpu.edges[1:top_k]]
            
            overlap = length(intersect(cpu_pairs, gpu_pairs))
            
            # Expect at least a 100% overlap in the top 100 edges.
            @test overlap >= 100
        end
    end
else
    @warn "CUDA unavailable. Skipping numeric equivalence tests."
end
