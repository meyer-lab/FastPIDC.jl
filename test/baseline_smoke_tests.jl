using Test, DelimitedFiles
using .BaselineHelpers
using FastPIDC
using Distributed
using MatrixMarket
using SparseArrays

# Paths
const DATA_DIR = joinpath(dirname(@__FILE__), "data")
const OUT_DIR  = joinpath(dirname(@__FILE__), "baseline_outputs")
isdir(OUT_DIR) || mkpath(OUT_DIR)

const TIMINGS_PATH = joinpath(OUT_DIR, "timings.tsv")

# --- Small yeast dataset: mirror current tests and save snapshots ---
@testset "Yeast10 baseline snapshots" begin
    data_file = joinpath(DATA_DIR, "yeast1_10_data.txt")
    mi_net, clr_net, puc_net, pidc_net = run_all_networks(data_file)

    # Save matrices and edge lists (for future diffs)
    # NOTE: these are optional; you already have reference txt files for MI/PUC/CLRs.
    # We still snapshot PIDC edges for end-to-end ranking diffs.
    BaselineHelpers.save_edges_tsv(joinpath(OUT_DIR, "pidc_yeast_edges.tsv"), pidc_net)
end

# --- Toy 1kx200 dataset: determinism + timings/allocations ---
@testset "Toy 1kx200 determinism + timings" begin
    data_file = joinpath(DATA_DIR, "toy_small_200.txt")

    # First run
    t1 = @timed begin
        mi1, clr1, puc1, pidc1 = run_all_networks(data_file)
        mi1, clr1, puc1, pidc1
    end
    (mi1, clr1, puc1, pidc1) = t1.value

    # Second run
    t2 = @timed begin
        mi2, clr2, puc2, pidc2 = run_all_networks(data_file)
        mi2, clr2, puc2, pidc2
    end
    (mi2, clr2, puc2, pidc2) = t2.value

    # Determinism: check a subset of weights and full edge order equality
    @test length(mi1.edges)   == length(mi2.edges)
    @test length(puc1.edges)  == length(puc2.edges)
    @test length(pidc1.edges) == length(pidc2.edges)

    # Exact equality should hold given deterministic pipeline; if CI noise appears,
    # switch to ≈ with tiny tolerance.
    for idx in (1, 5, 10, 50, length(pidc1.edges))
        @test mi1.edges[idx].weight   == mi2.edges[idx].weight
        @test clr1.edges[idx].weight  == clr2.edges[idx].weight
        @test puc1.edges[idx].weight  == puc2.edges[idx].weight
        @test pidc1.edges[idx].weight == pidc2.edges[idx].weight
        @test Set([n.label for n in pidc1.edges[idx].nodes]) ==
              Set([n.label for n in pidc2.edges[idx].nodes])
    end

    # Persist snapshots for later diffs (original vs. modernized)
    BaselineHelpers.save_edges_tsv(joinpath(OUT_DIR, "pidc_toy_edges.tsv"), pidc1)

    # Log timings/allocations to a simple TSV
    open(TIMINGS_PATH, "w") do io
        println(io, "phase\twall_seconds\talloc_bytes")
        println(io, "toy_200_first\t$(t1.time)\t$(t1.bytes)")
        println(io, "toy_200_second\t$(t2.time)\t$(t2.bytes)")
    end

    @info "Toy timings (s)" first=t1.time second=t2.time
    @info "Toy allocations (bytes)" first=t1.bytes second=t2.bytes

end

@testset "Config is backward compatible" begin
    data_file = joinpath(dirname(@__FILE__), "data", "toy_small_200.txt")
    nodes = get_nodes(data_file)
    net1 = InferredNetwork(PIDCNetworkInference(), nodes)
    net2 = InferredNetwork(PIDCNetworkInference(), nodes; config = PIDCConfig())
    @test net1.edges[1].weight == net2.edges[1].weight
end

@testset "Pruned context matches legacy when k >= n" begin
    # Essentially the exact same test as "Pruned PUC matches full when k >= n"
    data_file = joinpath(DATA_DIR, "toy_small_200.txt")
    nodes = get_nodes(data_file)
    n = length(nodes)

    cfg_legacy = PIDCConfig(triplet_block_k = n, context_mode=:legacy_dense)
    cfg_pruned = PIDCConfig(triplet_block_k = n, context_mode=:pruned)

    net_legacy = InferredNetwork(PIDCNetworkInference(), nodes; config=cfg_legacy)
    net_pruned = InferredNetwork(PIDCNetworkInference(), nodes; config=cfg_pruned)

    @test length(net_legacy.edges) == length(net_pruned.edges)

    function edge_map(net)
        d = Dict{Tuple{String,String},Float64}()
        for e in net.edges
            a,b = sort([e.nodes[1].label, e.nodes[2].label])
            d[(a,b)] = e.weight
        end
        d
    end

    d1 = edge_map(net_legacy)
    d2 = edge_map(net_pruned)

    @test keys(d1) == keys(d2)
    for k in keys(d1)
        @test d1[k] ≈ d2[k] atol=1e-10
    end
end

@testset "Pruned PUC matches full when k >= n" begin
    data_file = joinpath(DATA_DIR, "toy_small_200.txt")
    nodes = get_nodes(data_file)
    n = length(nodes)

    cfg_full = PIDCConfig(triplet_block_k = 0)  # full PUC
    cfg_tar  = PIDCConfig(triplet_block_k = n)

    net_full = InferredNetwork(PIDCNetworkInference(), nodes; config = cfg_full)
    net_tar  = InferredNetwork(PIDCNetworkInference(), nodes; config = cfg_tar)

    @test length(net_full.edges) == length(net_tar.edges)

    for idx in (1, 5, 10, 50, length(net_full.edges))
        @test net_full.edges[idx].weight ≈ net_tar.edges[idx].weight atol = 1e-8
        @test Set(n.label for n in net_full.edges[idx].nodes) ==
              Set(n.label for n in net_tar.edges[idx].nodes)
    end
end

@testset "Pruned PUC timing (target mode, toy 1kx200)" begin
    data_file = joinpath(DATA_DIR, "toy_small_200.txt")

    cfg_full = PIDCConfig(triplet_block_k = 0)
    cfg_tar  = PIDCConfig(triplet_block_k = 20)

    t_full = @timed begin
        _mi, _clr, _puc, pidc_full = run_all_networks(data_file; config = cfg_full)
        pidc_full
    end

    t_tar = @timed begin
        _mi, _clr, _puc, pidc_tar = run_all_networks(data_file; config = cfg_tar)
        pidc_tar
    end

    @info "PUC timing (target mode, toy)" full = t_full.time target = t_tar.time
    @info "PUC allocations (target mode, toy)" full = t_full.bytes target = t_tar.bytes

    # Edge counts ≈ TSV lengths
    full_edges   = length(t_full.value.edges)
    pruned_edges = length(t_tar.value.edges)

    @info "PIDC edge counts (target mode, toy)" full_edges = full_edges pruned_edges = pruned_edges

    # Optional soft sanity check: pruning really prunes
    @test pruned_edges < full_edges

    open(TIMINGS_PATH, "a") do io
        println(io, "toy_200_puc_full_target\t$(t_full.time)\t$(t_full.bytes)")
        println(io, "toy_200_puc_pruned_target\t$(t_tar.time)\t$(t_tar.bytes)")
        println(io, "toy_200_puc_pruned_target edges vs full\t$(pruned_edges)\t$(full_edges)")
    end

    # Soft assertion:
    # @test t_tar.time <= 1.2 * t_full.time
end

@testset "Pruned PUC distributed matches threaded when k >= n" begin
    data_file = joinpath(DATA_DIR, "toy_small_200.txt")
    nodes = get_nodes(data_file)
    n = length(nodes)

    cfg_threads = PIDCConfig(
        triplet_block_k = n,
        triplet_backend = :threads,
        verbose = true,
    )

    cfg_dist = PIDCConfig(
        triplet_block_k = n,
        triplet_backend = :distributed,
        verbose = true,
    )

    net_threads = InferredNetwork(PIDCNetworkInference(), nodes; config = cfg_threads)
    net_dist    = InferredNetwork(PIDCNetworkInference(), nodes; config = cfg_dist)

    @test length(net_threads.edges) == length(net_dist.edges)

    for idx in (1, 5, 10, 50, length(net_threads.edges))
        @test net_threads.edges[idx].weight ≈ net_dist.edges[idx].weight atol = 1e-8
        @test Set(n.label for n in net_threads.edges[idx].nodes) ==
              Set(n.label for n in net_dist.edges[idx].nodes)
    end
end

@testset "Toy dataset writes TSV and MTX consistently" begin
    function count_nonzero_entries_from_tsv(path::String)
        mat = readdlm(path, '\t')
        n_nonzero = 0
    
        for i in 1:size(mat, 1)
            w = Float64(mat[i, 3])
            w == 0.0 && continue
            n_nonzero += 1
        end
    
        return n_nonzero
    end

    # Swap this path if your actual toy 1k x 200 file has a different name
    data_file = joinpath(DATA_DIR, "toy_small_200.txt")

    cfg = PIDCConfig(
        triplet_block_k = 0,
        triplet_backend = :threads,
        verbose = false,
    )

    nodes = get_nodes(data_file)
    net = InferredNetwork(PIDCNetworkInference(), nodes; config = cfg)

    tsv_path   = joinpath(OUT_DIR, "toy_pidc_edges.tsv")
    mtx_path   = joinpath(OUT_DIR, "toy_pidc_edges.mtx")
    genes_path = joinpath(OUT_DIR, "toy_pidc_edges_genes.txt")

    # Write both formats
    write_network_file(tsv_path, net)
    write_network_mtx(mtx_path, net)

    # Smoke checks: files exist
    @test isfile(tsv_path)
    @test isfile(mtx_path)
    @test isfile(genes_path)

    # Read MTX back in Julia
    A = MatrixMarket.mmread(mtx_path)

    # Read gene list sidecar
    genes = readlines(genes_path)

    # Basic consistency checks
    n = length(net.nodes)
    @test size(A, 1) == n
    @test size(A, 2) == n
    @test length(genes) == n

    # If internal net.edges stores each undirected edge once,
    # the symmetric matrix should have 2*E nonzeros.
    n_unique_pairs = count_nonzero_entries_from_tsv(tsv_path)
    @test nnz(A) == n_unique_pairs

    # Diagonal should be zero
    # @test all(diag(Matrix(A)) .== 0.0)
    @test all(A[i, i] == 0.0 for i in 1:size(A, 1))
end

# --------- LARGE TESTS --------

@testset "Large toy PUC timing" begin
    large_file = joinpath(DATA_DIR, "toy_large_1k.txt")

    cfg_full      = PIDCConfig(triplet_block_k = 0)
    cfg_target_thr = PIDCConfig(triplet_block_k = 20, triplet_backend = :threads, verbose = true)
    cfg_target_dist = PIDCConfig(triplet_block_k = 20, triplet_backend = :distributed, verbose = true)

    t_full = @timed begin
        _mi, _clr, _puc, pidc_full = run_all_networks(large_file; config = cfg_full)
        pidc_full
    end

    t_target_thr = @timed begin
        _mi, _clr, _puc, pidc_target_thr = run_all_networks(large_file; config = cfg_target_thr)
        pidc_target_thr
    end
    
    t_target_dist = @timed begin
        _mi, _clr, _puc, pidc_target_dist = run_all_networks(large_file; config = cfg_target_dist)
        pidc_target_dist
    end

    @info "Large toy PUC timings (s)" full = t_full.time target_thr = t_target_thr.time target_dist = t_target_dist.time
    @info "Large toy PUC allocations (bytes)" full = t_full.bytes target_thr = t_target_thr.bytes target_dist = t_target_dist.bytes

      # Edge counts ≈ TSV lengths
      full_edges   = length(t_full.value.edges)
      target_pruned_edges = length(t_target_dist.value.edges)
  
      @info "PIDC edge counts (target mode, toy)" full_edges = full_edges pruned_edges = target_pruned_edges
  
      # Optional soft sanity check: pruning really prunes
      @test target_pruned_edges < full_edges

    open(TIMINGS_PATH, "a") do io
        println(io, "toy_large_1k_puc_full\t$(t_full.time)\t$(t_full.bytes)")
        println(io, "toy_large_1k_puc_pruned_target_threads\t$(t_target_thr.time)\t$(t_target_thr.bytes)")
        println(io, "toy_large_1k_puc_pruned_target_distributed\t$(t_target_dist.time)\t$(t_target_dist.bytes)")
        println(io, "toy_large_1k_puc_pruned_target_distributed edges vs full\t$(target_pruned_edges)\t$(full_edges)")
    end
end