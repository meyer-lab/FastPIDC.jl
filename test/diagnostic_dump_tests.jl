using Test
using DelimitedFiles
using FastPIDC
const DATA_DIR = joinpath(dirname(@__FILE__), "data")
const OUT_DIR = joinpath(dirname(@__FILE__), "baseline_outputs")
isdir(OUT_DIR) || mkpath(OUT_DIR)

@testset "Diagnostic dumps for PIDC" begin
    data_file = joinpath(DATA_DIR, "toy_small_200.txt")
    mi_file = joinpath(OUT_DIR, "toy_mi_dump.tsv")
    puc_file = joinpath(OUT_DIR, "toy_puc_dump.tsv")


    # Clean up from previous runs
    isfile(mi_file) && rm(mi_file)
    isfile(puc_file) && rm(puc_file)


    # Small k just to exercise pruned PIDC path
    cfg = PIDCConfig(
        backend = :cpu,
        dump_mi_path = mi_file,
        dump_mi_fraction = 0.1,   # top 10% MI pairs
        dump_puc_path = puc_file,
        dump_puc_fraction = 0.1,
    )

    nodes = get_nodes(data_file)
    net = InferredNetwork(PIDCNetworkInference(), nodes; config = cfg)

    @test isfile(mi_file)
    @test isfile(puc_file)

    mi_lines = readlines(mi_file)
    @test length(mi_lines) > 1
    @test startswith(mi_lines[1], "gene_i\tgene_j\tmi")

    puc_lines = readlines(puc_file)
    @test length(puc_lines) > 1
    @test startswith(puc_lines[1], "gene_i\tgene_j\tpuc")

end
