using FastPIDC
using CUDA
using Distributed
using SharedArrays

# Ensure 10 workers for CPU implementation
if nprocs() < 11
    needed = 11 - nprocs()
    addprocs(needed)
end
@everywhere using FastPIDC

# Check for CUDA
if !CUDA.functional()
    println("CUDA is not functional on this system. Cannot run GPU benchmark.")
    exit(0)
end

dataset = "test/data/toy_large_1k.txt"
if !isfile(dataset)
    # Try alternate path if running from within test/
    dataset = "data/toy_large_1k.txt"
end

println("Loading nodes from $dataset...")
nodes = get_nodes(dataset)
num_nodes = length(nodes)
println("Loaded $num_nodes nodes.")

# Warmup and Baseline (CPU - Distributed with 10 workers)
println("\n--- CPU Distributed ($(nprocs()-1) workers) ---")
config_cpu = PIDCConfig(triplet_backend=:distributed, verbose=false)
# Warmup
InferredNetwork(PUCNetworkInference(), nodes[1:min(100, num_nodes)], config=config_cpu)
t_cpu = @elapsed InferredNetwork(PUCNetworkInference(), nodes, config=config_cpu)
println("CPU Distributed time: $t_cpu seconds")

# CUDA Backend
println("\n--- CUDA GPU ---")
config_cuda = PIDCConfig(triplet_backend=:cuda, verbose=false)
# Warmup
InferredNetwork(PUCNetworkInference(), nodes[1:min(100, num_nodes)], config=config_cuda)
# Benchmark
t_cuda = @elapsed InferredNetwork(PUCNetworkInference(), nodes, config=config_cuda)
println("CUDA GPU time: $t_cuda seconds")

# Correctness check
println("\n--- Correctness Check ---")
res_cpu = InferredNetwork(PUCNetworkInference(), nodes[1:min(100, num_nodes)], config=config_cpu)
res_cuda = InferredNetwork(PUCNetworkInference(), nodes[1:min(100, num_nodes)], config=config_cuda)

w_cpu = [e.weight for e in res_cpu.edges[1:10]]
w_cuda = [e.weight for e in res_cuda.edges[1:10]]
println("Top 10 weights (CPU):  $w_cpu")
println("Top 10 weights (CUDA): $w_cuda")

max_diff = 0.0
for i in 1:min(100, length(res_cpu.edges))
    global max_diff = max(max_diff, abs(res_cpu.edges[i].weight - res_cuda.edges[i].weight))
end
println("Max difference in top 100 edges: $max_diff")

speedup = t_cpu / t_cuda
println("\nSpeedup vs CPU ($(nprocs()-1) workers): $(round(speedup, digits=2))x")
