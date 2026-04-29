# src/puc_full.jl
#
# Optimized Full PUC computation using the Sorting Trick and fused Information Measures.
# Reduces complexity from O(N^3 B) to O(N^2 B log N).
using SharedArrays
using Distributed

function compute_puc_full(nodes::Vector{Node};
    estimator::String = "maximum_likelihood",
    base::Int = 2,
    config::PIDCConfig = PIDCConfig())

    n = length(nodes)
    max_bins = maximum(node -> node.number_of_bins, nodes)
    S = length(nodes[1].binned_values)
    inv_S = 1.0 / S
    inv_log_base = 1.0 / log(base)

    # --- Pre-calculate marginal logs -------------------------------------------
    marginal_logs = [log.(node.probabilities) for node in nodes]

    # --- Allocate caches -------------------------------------------------------
    mi_matrix = SharedArray{Float64}(n, n)
    si_tensor = SharedArray{Float64}(n, max_bins, n)
    fill!(mi_matrix, 0.0)
    fill!(si_tensor, 0.0)
    
    puc_scores = SharedArray{Float64}(n, n)
    fill!(puc_scores, 0.0)

    # --- Fused MI + SI cache fill ----------------------------------------------
    @sync @distributed for i in 1:n
        si2_acc = zeros(Float64, max_bins)
        freqs = zeros(Int, max_bins, max_bins)
        
        for j in i+1:n
            node_i = nodes[i]
            node_j = nodes[j]
            nb_i = node_i.number_of_bins
            nb_j = node_j.number_of_bins
            
            fill!(freqs, 0)
            bids_i = node_i.binned_values
            bids_j = node_j.binned_values
            @inbounds for k in 1:S
                freqs[bids_i[k], bids_j[k]] += 1
            end
            
            mi = 0.0
            fill!(si2_acc, 0.0)
            for bj in 1:nb_j
                pj = node_j.probabilities[bj]
                sum_si1 = 0.0
                for bi in 1:nb_i
                    f = freqs[bi, bj]
                    if f > 0
                        p_ij = f * inv_S
                        term = p_ij * (log(p_ij) - marginal_logs[i][bi] - marginal_logs[j][bj])
                        mi += term
                        sum_si1 += term
                        si2_acc[bi] += term
                    end
                end
                if pj > 0
                    si_tensor[i, bj, j] = (sum_si1 / pj) * inv_log_base
                end
            end
            
            for bi in 1:nb_i
                pi = node_i.probabilities[bi]
                if pi > 0
                    si_tensor[j, bi, i] = (si2_acc[bi] / pi) * inv_log_base
                end
            end
            
            mi_val = mi * inv_log_base
            mi_matrix[i, j] = mi_val
            mi_matrix[j, i] = mi_val
        end
    end

    # --- PUC computation using the Sorting Trick -------------------------------
    @sync @distributed for z in 1:n
        pz = nodes[z].probabilities
        nz = length(pz)
        S_agg = zeros(Float64, n)
        
        for b in 1:nz
            v = view(si_tensor, :, b, z)
            p = sortperm(v)
            u = v[p]
            
            prefix = copy(u)
            for i in 2:n
                prefix[i] += prefix[i-1]
            end
            
            vz = v[z]
            for k in 1:n
                x = p[k]
                vx = v[x]
                S_val = prefix[k] + (n - k) * vx
                # Subtract y=x and y=z
                S_eff = S_val - vx - min(vx, vz)
                S_agg[x] += pz[b] * S_eff
            end
        end
        
        for x in 1:n
            x == z && continue
            mi = mi_matrix[x, z]
            if mi > 0
                # Numerical clamping as in original code
                # Original sum was sum_y max(0, (mi - Ry)/mi)
                # Since R <= mi, this is sum_y (1 - Ry/mi)
                score = (n - 2) - (S_agg[x] / mi)
                puc_scores[x, z] = max(0.0, score)
                puc_scores[z, x] = max(0.0, score)
            end
        end
    end

    return Array(mi_matrix), Array(puc_scores)
end
