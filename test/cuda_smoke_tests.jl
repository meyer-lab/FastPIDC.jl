using FastPIDC
using Test

@testset "CUDA Backend Hook" begin
    # Test that requesting :cuda backend when CUDA.jl is NOT loaded throws the expected error.
    
    # Setup dummy data
    data = [
        "G1" 1 2 1 2;
        "G2" 2 1 2 1;
        "G3" 1 1 2 2
    ]
    nodes = [FastPIDC.Node(data[i, :], "bayesian_blocks", "maximum_likelihood", 10) for i in 1:3]
    config = PIDCConfig(backend=:cuda)
    
    # Should throw ErrorException because CUDA.jl is not loaded in this test environment
    try
        FastPIDC.compute_puc_full(nodes, config=config)
        @test false # Should not reach here
    catch e
        @test e isa ErrorException
        @test contains(e.msg, "CUDA.jl is not loaded")
    end
end
