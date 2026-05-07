# src/mi_dump.jl
#
# Dump upper-triangular MI scores (i<j) to a long-format TSV.
# This is intended for diagnostics (KDE, ECDF, threshold tuning) outside
# the core package.

function dump_mi_scores(
    mi_scores::AbstractMatrix{Float64},
    nodes::Vector{Node},
    config::PIDCConfig,
)

    path = config.dump_mi_path
    path === nothing && return  # no-op if not requested

    frac = clamp(config.dump_mi_fraction, 0.0, 1.0)
    n = length(nodes)
    n == 0 && return

    # Collect (mi, i, j) for i<j
    pairs = Vector{Tuple{Float64,Int,Int}}()
    sizehint!(pairs, n * (n - 1) ÷ 2)

    for i = 1:n
        for j = (i+1):n
            push!(pairs, (mi_scores[i, j], i, j))
        end
    end

    # Optionally keep only top fraction by MI
    if frac < 1.0
        sort!(pairs; by = x -> x[1], rev = true)
        k = max(1, round(Int, frac * length(pairs)))
        pairs = pairs[1:k]
    end

    # Write TSV: gene_i, gene_j, mi
    open(path, "w") do io
        println(io, "gene_i\tgene_j\tmi")
        for (mi, i, j) in pairs
            println(io, "$(nodes[i].label)\t$(nodes[j].label)\t$mi")
        end
    end
end
