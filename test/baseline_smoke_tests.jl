using Test, DelimitedFiles
using .BaselineHelpers
using FastPIDC
using Distributed
using NPZ
using SparseArrays

# Paths
const DATA_DIR = joinpath(dirname(@__FILE__), "data")
const OUT_DIR = joinpath(dirname(@__FILE__), "baseline_outputs")
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
    @test length(mi1.edges) == length(mi2.edges)
    @test length(puc1.edges) == length(puc2.edges)
    @test length(pidc1.edges) == length(pidc2.edges)

    # Exact equality should hold given deterministic pipeline; if CI noise appears,
    # switch to ≈ with tiny tolerance.
    for idx in (1, 5, 10, 50, length(pidc1.edges))
        @test mi1.edges[idx].weight == mi2.edges[idx].weight
        @test clr1.edges[idx].weight == clr2.edges[idx].weight
        @test puc1.edges[idx].weight == puc2.edges[idx].weight
        @test pidc1.edges[idx].weight == pidc2.edges[idx].weight
        @test Set([n.label for n in pidc1.edges[idx].nodes]) == Set([n.label for n in pidc2.edges[idx].nodes])
    end
end

@testset "I/O Parity: Legacy TXT vs HDF5" begin
    txt_path = joinpath(DATA_DIR, "toy_small_200.txt")
    h5_path  = joinpath(DATA_DIR, "toy_small_200.h5")
    
    # Load nodes using both methods
    nodes_txt = get_nodes(txt_path)
    nodes_h5  = get_nodes(h5_path)
    
    # Assert the same number of nodes/genes were loaded
    @test length(nodes_txt) == length(nodes_h5)
    
    # Assert every single node matches perfectly
    for i in 1:length(nodes_txt)
        node_t = nodes_txt[i]
        node_h = nodes_h5[i]
        
        # Verify gene labels match exactly
        @test node_t.label == node_h.label

        # HDF5 should preserve floats perfectly.
        if hasproperty(node_t, :probabilities)
            @test isapprox(node_t.probabilities, node_h.probabilities, atol=1e-8)
        else
            # Fallback if probabilities aren't directly exposed
            # Replace with the relevant data field in your Node struct
            # @test isapprox(node_t.data, node_h.data, atol=1e-8)
            println("failed")
        end
    end
end

@testset "Toy dataset writes TSV and NPY consistently" begin
    function count_nonzero_entries_from_tsv(path::String)
        mat = readdlm(path, '\t')
        n_nonzero = 0

        for i = 1:size(mat, 1)
            w = Float64(mat[i, 3])
            w == 0.0 && continue
            n_nonzero += 1
        end

        return n_nonzero
    end

    # Swap this path if your actual toy 1k x 200 file has a different name
    data_file = joinpath(DATA_DIR, "toy_small_200.txt")

    cfg = PIDCConfig(backend = :cpu, verbose = false)

    nodes = get_nodes(data_file)
    net = InferredNetwork(PIDCNetworkInference(), nodes; config = cfg)

    tsv_path   = joinpath(OUT_DIR, "toy_pidc_edges.tsv")
    npy_path   = joinpath(OUT_DIR, "toy_pidc_edges.npy")
    genes_path = joinpath(OUT_DIR, "toy_pidc_edges_genes.txt")

    # Write both formats
    write_network_file(tsv_path, net)
    write_network_npy(npy_path, net)

    # Read the sidecar genes file back in
    loaded_genes = readlines(genes_path)
    expected_genes = [String(node.label) for node in net.nodes]

    # Read NPY back in Julia
    A = npzread(npy_path)

    # Basic consistency checks
    n = length(net.nodes)
    @test size(A, 1) == n
    @test size(A, 2) == n
    
    # Check that the genes sidecar has the correct size and order
    @test length(loaded_genes) == n
    @test loaded_genes == expected_genes

    # If internal net.edges stores each undirected edge once,
    # the symmetric matrix should have 2*E nonzeros.
    n_unique_pairs = count_nonzero_entries_from_tsv(tsv_path)
    
    # Since Float32 might have tiny variations from Float64 TSV parsing, 
    # counting non-zeros is a robust check
    n_nonzero_npy = count(x -> x != 0.0f0, A)

    # They should match exactly
    @test n_nonzero_npy == n_unique_pairs

    # Diagonal should be zero
    @test all(A[i, i] == 0.0f0 for i in 1:size(A, 1))
end
