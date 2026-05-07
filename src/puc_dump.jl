# src/puc_dump.jl
# Dump pre-context PUC scores to TSV.

using Printf

@inline function _clamp01(x::Float64)
    x < 0.0 && return 0.0
    x > 1.0 && return 1.0
    return x
end

@inline function _nkeep(m::Int, fraction::Float64)
    f = _clamp01(fraction)
    return max(1, Int(ceil(f * m)))
end

# Robust node label getter
@inline function _label(node)
    # Your Node likely has .label
    return hasproperty(node, :label) ? getproperty(node, :label) : string(node)
end

"""
    dump_puc_scores(scores, nodes, config; mi_scores=nothing)

Dump pre-context PUC scores to TSV at config.dump_puc_path.

"""
function dump_puc_scores(
    scores::AbstractMatrix{<:Real},
    nodes::Vector,
    config::PIDCConfig;
    mi_scores::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
)

    path = config.dump_puc_path
    path === nothing && return nothing

    n = size(scores, 1)
    @assert size(scores, 2) == n
    @assert length(nodes) == n

    # Collect candidate (i,j,score)
    ei = Int[]
    ej = Int[]
    w = Float64[]

    # dump all unordered pairs
    sizehint!(ei, n*(n-1) ÷ 2)
    sizehint!(ej, n*(n-1) ÷ 2)
    sizehint!(w, n*(n-1) ÷ 2)
    for i = 1:n
        for j = (i+1):n
            push!(ei, i);
            push!(ej, j);
            push!(w, Float64(scores[i, j]))
        end
    end

    ord = sortperm(w; rev = true)
    keep_n = _nkeep(length(ord), config.dump_puc_fraction)

    open(path, "w") do io
        println(io, "gene_i\tgene_j\tpuc")
        @inbounds for u = 1:keep_n
            t = ord[u]
            i = ei[t];
            j = ej[t]
            @printf(io, "%s\t%s\t%.17g\n", _label(nodes[i]), _label(nodes[j]), w[t])
        end
    end

    return nothing
end
