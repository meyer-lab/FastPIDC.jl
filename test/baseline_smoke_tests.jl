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

    # # Log timings/allocations to a simple TSV
    # open(TIMINGS_PATH, "w") do io
    #     println(io, "phase\twall_seconds\talloc_bytes")
    #     println(io, "toy_200_first\t$(t1.time)\t$(t1.bytes)")
    #     println(io, "toy_200_second\t$(t2.time)\t$(t2.bytes)")
    # end

    # @info "Toy timings (s)" first=t1.time second=t2.time
    # @info "Toy allocations (bytes)" first=t1.bytes second=t2.bytes

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
        backend = :cpu,
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
    @test all(A[i, i] == 0.0 for i in 1:size(A, 1))
end