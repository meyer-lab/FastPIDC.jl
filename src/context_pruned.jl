# src/context_pruned.jl
#
# Memory-efficient context weighting for CLR/PIDC that only applies context
# to the MI-kNN-pruned candidate edge set (same set as build_knn_mask / build_edges_with_mi_pruning).
#
# Key property:
#   - When k >= n-1, candidate pairs = all i<j, so this recovers the legacy dense behavior.

using Distributions

"""
    build_candidate_pairs(mi_scores, k)

Return candidate unordered pairs (ei, ej) with ei[t] < ej[t], following the SAME
MI kNN rule as build_knn_mask:

For each i:
  idxs = partialsortperm(mi_scores[i,:], 1:k; rev=true)
  for j in idxs (skipping j==i):
     keep (i,j) and symmetrize

If k <= 0 or k >= n-1 -> returns all pairs i<j.

This implementation avoids allocating a dense keep[n,n] Bool matrix by:
  - pushing encoded unordered pairs into a vector
  - sort! + unique! to deduplicate
"""
function build_candidate_pairs(mi_scores::AbstractMatrix{Float64}, k::Int)
    n = size(mi_scores, 1)
    @assert size(mi_scores, 2) == n "MI matrix must be square"

    # No pruning / dense: all unordered pairs
    if k <= 0 || k >= n - 1
        m = div(n * (n - 1), 2)
        ei = Vector{Int}(undef, m)
        ej = Vector{Int}(undef, m)
        t = 0
        @inbounds for i in 1:n-1
            for j in i+1:n
                t += 1
                ei[t] = i
                ej[t] = j
            end
        end
        return ei, ej
    end

    # Pruned: collect unordered pairs from kNN lists, then deduplicate.
    # Upper bound: each node contributes up to k neighbors => ~ n*k directed entries.
    # After symmetrization and dedup, unordered edges are typically O(2kn).
    pairs = Vector{UInt64}()
    sizehint!(pairs, n * k)

    @inbounds for i in 1:n
        row = view(mi_scores, i, :)
        idxs = partialsortperm(row, 1:k; rev = true)
        for j in idxs
            j == i && continue
            a = i < j ? i : j
            b = i < j ? j : i
            # Encode unordered pair (a,b) into UInt64: high 32 bits = a, low 32 bits = b
            push!(pairs, (UInt64(a) << 32) | UInt64(b))
        end
    end

    sort!(pairs)
    unique!(pairs)

    m = length(pairs)
    ei = Vector{Int}(undef, m)
    ej = Vector{Int}(undef, m)

    @inbounds for t in 1:m
        key = pairs[t]
        ei[t] = Int(key >> 32)
        ej[t] = Int(key & 0xFFFFFFFF)
    end

    return ei, ej
end



# ---------- Helper: build incident lists for each node ----------
function build_incident_lists(ei::Vector{Int}, ej::Vector{Int}, n::Int)
    deg = zeros(Int, n)
    @inbounds for t in eachindex(ei)
        deg[ei[t]] += 1
        deg[ej[t]] += 1
    end

    inc = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        inc[i] = Int[]
        sizehint!(inc[i], deg[i])
    end

    @inbounds for t in eachindex(ei)
        i = ei[t]; j = ej[t]
        push!(inc[i], t)
        push!(inc[j], t)
    end

    return inc
end


# ---------- CLR context on pruned candidate edges ----------
# ---------- CLR context on pruned candidate edges ----------
function clr_context_weights_pruned(scores::AbstractMatrix{Float64},
                                    mi_scores::AbstractMatrix{Float64},
                                    k::Int)
    n = size(scores, 1)
    @assert size(scores, 2) == n

    ei, ej = build_candidate_pairs(mi_scores, k)
    inc = build_incident_lists(ei, ej, n)

    mu = zeros(Float64, n)
    v  = zeros(Float64, n)

    # compute per-node mean/var (SAMPLE var, matching Julia var(x) default)
    @inbounds for i in 1:n
        idxs = inc[i]
        m = length(idxs)

        if m == 0
            mu[i] = 0.0
            v[i]  = 0.0
            continue
        end

        s = 0.0
        for t in idxs
            a = ei[t]; b = ej[t]
            s += scores[a, b]  # scores symmetric on candidate edges
        end
        mui = s / m
        mu[i] = mui

        ss = 0.0
        for t in idxs
            a = ei[t]; b = ej[t]
            d = scores[a, b] - mui
            ss += d * d
        end

        # sample variance to match var(scores_i) in legacy apply_clr_context
        v[i] = (m > 1) ? (ss / (m - 1)) : 0.0
    end

    w = Vector{Float64}(undef, length(ei))
    @inbounds for t in eachindex(ei)
        i = ei[t]; j = ej[t]
        sc = scores[i, j]

        di = sc - mu[i]
        dj = sc - mu[j]

        zi = (v[i] == 0.0 || di < 0.0) ? 0.0 : (di * di) / v[i]
        zj = (v[j] == 0.0 || dj < 0.0) ? 0.0 : (dj * dj) / v[j]

        w[t] = sqrt(zi + zj)
    end

    return w, ei, ej
end


# ---------- PIDC context on pruned candidate edges ----------
function pidc_context_weights_pruned(scores::AbstractMatrix{Float64},
                                     mi_scores::AbstractMatrix{Float64},
                                     k::Int)
    n = size(scores, 1)
    @assert size(scores, 2) == n

    ei, ej = build_candidate_pairs(mi_scores, k)
    inc = build_incident_lists(ei, ej, n)

    gamma_ok  = falses(n)
    gamma_fit = Vector{Union{Nothing,Gamma}}(undef, n)

    # Always compute CLR fallback stats for every node (needed for edge-level fallback)
    mu = zeros(Float64, n)
    v  = zeros(Float64, n)

    @inbounds for i in 1:n
        idxs = inc[i]
        m = length(idxs)

        if m == 0
            gamma_ok[i]  = false
            gamma_fit[i] = nothing
            mu[i] = 0.0
            v[i]  = 0.0
            continue
        end

        # Collect incident scores once (still O(total candidate edges))
        svec = Vector{Float64}(undef, m)
        s = 0.0
        for (u, t) in enumerate(idxs)
            a = ei[t]; b = ej[t]
            val = scores[a, b]
            svec[u] = val
            s += val
        end

        mui = s / m
        mu[i] = mui

        ss = 0.0
        for u in 1:m
            d = svec[u] - mui
            ss += d * d
        end
        # sample variance (Julia var default)
        v[i] = (m > 1) ? (ss / (m - 1)) : 0.0

        # Try gamma fit; if it fails, we’ll CLR-fallback per edge
        try
            gamma_fit[i] = fit(Gamma, svec)
            gamma_ok[i]  = true
        catch
            gamma_fit[i] = nothing
            gamma_ok[i]  = false
        end
    end

    w = Vector{Float64}(undef, length(ei))
    @inbounds for t in eachindex(ei)
        i = ei[t]; j = ej[t]
        sc = scores[i, j]

        # legacy PIDC: try Gamma per node, but fallback is EDGE-LEVEL:
        # if either endpoint gamma is unavailable, use CLR formula.
        if gamma_ok[i] && gamma_ok[j]
            w[t] = cdf(gamma_fit[i]::Gamma, sc) + cdf(gamma_fit[j]::Gamma, sc)
        else
            di = sc - mu[i]
            dj = sc - mu[j]

            zi = (v[i] == 0.0 || di < 0.0) ? 0.0 : (di * di) / v[i]
            zj = (v[j] == 0.0 || dj < 0.0) ? 0.0 : (dj * dj) / v[j]

            w[t] = sqrt(zi + zj)
        end
    end

    return w, ei, ej
end



"""
    build_edges_from_pairs(nodes, ei, ej, w)

Construct Edge objects from (ei,ej,w), then sort descending by weight.
"""
function build_edges_from_pairs(nodes::Vector{Node},
                                ei::Vector{Int},
                                ej::Vector{Int},
                                w::Vector{Float64})
    @assert length(ei) == length(ej) == length(w)
    edges = Edge[]
    sizehint!(edges, length(ei))

    @inbounds for t in eachindex(ei)
        i = ei[t]; j = ej[t]
        push!(edges, Edge([nodes[i], nodes[j]], w[t]))
    end

    sort!(edges; by = get_weight, rev = true)
    return edges
end
