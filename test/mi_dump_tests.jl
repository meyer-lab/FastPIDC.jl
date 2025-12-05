using Test
using DelimitedFiles
using FastPIDC
const DATA_DIR = joinpath(dirname(@__FILE__), "data")
const OUT_DIR  = joinpath(dirname(@__FILE__), "baseline_outputs")
isdir(OUT_DIR) || mkpath(OUT_DIR)

@testset "MI dump for PIDC" begin
    data_file = joinpath(DATA_DIR, "toy_small_200.txt")
    out_file  = joinpath(OUT_DIR, "toy_mi_dump.tsv")

    # Clean up from previous runs
    isfile(out_file) && rm(out_file)

    # Small k just to exercise pruned PIDC path
    cfg = PIDCConfig(
        triplet_block_k = 50,
        dump_mi_path = out_file,
        dump_mi_fraction = 0.1,   # top 10% MI pairs
    )

    nodes = get_nodes(data_file)
    net   = InferredNetwork(PIDCNetworkInference(), nodes; config = cfg)

    @test isfile(out_file)

    lines = readlines(out_file)
    @test length(lines) > 1
    @test startswith(lines[1], "gene_i\tgene_j\tmi")

    n = length(nodes)
    max_pairs = n * (n - 1) ÷ 2
    n_rows = length(lines) - 1

    @test 1 <= n_rows <= max_pairs
end
