using FastPIDC
using Metal
using Distributed
using SharedArrays

# Check for Metal
if !Metal.functional()
    println("Metal is not functional on this system. Cannot run GPU benchmark.")
    exit(0)
end

dataset = "test/data/toy_large_1k.txt"
println("Loading nodes from $dataset...")
nodes = get_nodes(dataset)
num_nodes = length(nodes)
println("Loaded $num_nodes nodes.")

# Warmup and Baseline (CPU - Distributed)
# We start with 1 worker to see single-core performance if procs not added yet
println("\n--- CPU Distributed (1 worker) ---")
config_cpu = PIDCConfig(triplet_backend=:distributed, verbose=false)
# Note: we use InferredNetwork to trigger the full pipeline
t_cpu = @elapsed InferredNetwork(PUCNetworkInference(), nodes, config=config_cpu)
println("CPU Distributed (1 worker) time: $t_cpu seconds")

# Metal Backend
println("\n--- Metal GPU ---")
config_metal = PIDCConfig(triplet_backend=:metal, verbose=false)
# Warmup and get result
res_metal = InferredNetwork(PUCNetworkInference(), nodes, config=config_metal)
# Benchmark
t_metal = @elapsed InferredNetwork(PUCNetworkInference(), nodes, config=config_metal)
println("Metal GPU time: $t_metal seconds")

# Correctness check
println("\n--- Correctness Check ---")
res_cpu = InferredNetwork(PUCNetworkInference(), nodes, config=config_cpu)

w_cpu = [e.weight for e in res_cpu.edges[1:10]]
w_metal = [e.weight for e in res_metal.edges[1:10]]
println("Top 10 weights (CPU):   $w_cpu")
println("Top 10 weights (Metal): $w_metal")

# Compare some weights directly
# Since edges are sorted, let's just compare the weight matrices if we could access them.
# But we can check if they are approximately equal.
max_diff = 0.0
for i in 1:min(100, length(res_cpu.edges))
    global max_diff = max(max_diff, abs(res_cpu.edges[i].weight - res_metal.edges[i].weight))
end
println("Max difference in top 100 edges: $max_diff")

speedup = t_cpu / t_metal
println("\nSpeedup: $(round(speedup, digits=2))x")

if nprocs() > 1
    println("\n--- CPU Distributed ($(nprocs()) workers) ---")
    t_cpu_multi = @elapsed InferredNetwork(PUCNetworkInference(), nodes, config=config_cpu)
    println("CPU Distributed ($(nprocs()) workers) time: $t_cpu_multi seconds")
    println("Speedup vs Multi-core CPU: $(round(t_cpu_multi / t_metal, digits=2))x")
end
