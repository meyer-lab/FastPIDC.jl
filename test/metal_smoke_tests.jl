using Test
using FastPIDC

@testset "Metal Backend Hook" begin
    # Create dummy nodes
    nodes = [
        Node(["A", 1, 2, 1, 2], "uniform_width", "maximum_likelihood", 2),
        Node(["B", 2, 1, 2, 1], "uniform_width", "maximum_likelihood", 2),
        Node(["C", 1, 1, 2, 2], "uniform_width", "maximum_likelihood", 2)
    ]
    
    config = PIDCConfig(triplet_backend=:metal)
    
    # Should throw ErrorException because Metal.jl is not loaded in this test environment
    @test_throws ErrorException InferredNetwork(PUCNetworkInference(), nodes, config=config)
    
    # Verify the error message
    try
        InferredNetwork(PUCNetworkInference(), nodes, config=config)
    catch e
        @test contains(e.msg, "Metal.jl is not loaded")
    end
end
