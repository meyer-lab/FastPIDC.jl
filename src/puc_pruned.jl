# src/puc_pruned.jl
#
# Optimized Pruned PUC using fused Information Measures and efficient SIMD loops.
using SharedArrays

function compute_puc_pruned(nodes::Vector{Node};
    estimator::String = "maximum_likelihood",
    base::Int = 2,
    config::PIDCConfig)

    n = length(nodes)
    k = min(config.triplet_block_k, max(n - 1, 0))

    # Fallback if pruning is disabled
    if k == 0
        return compute_puc_full(nodes; estimator = estimator, base = base, config = config)
    end

    max_bins = maximum(node -> node.number_of_bins, nodes)
    S = length(nodes[1].binned_values)
    inv_S = 1.0 / S
    inv_log_base = 1.0 / log(base)

    # --- Pre-calculate marginal logs -------------------------------------------
    marginal_logs = [log.(node.probabilities) for node in nodes]

    # --- build NodePair cache (MI + specific information) ------------
    mi_matrix = zeros(Float64, n, n)
    si_tensor = zeros(Float64, n, max_bins, n)

    Threads.@threads for i in 1:n
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
            @inbounds for s in 1:S
                freqs[bids_i[s], bids_j[s]] += 1
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

    # --- MI-based neighbor lists for each gene -----------------------
    neighbors = Vector{Vector{Int}}(undef, n)
    for t in 1:n
        mivals = [(mi_matrix[t, u], u) for u in 1:n if u != t]
        sort!(mivals; by = x -> x[1], rev = true)
        k_eff = min(k, length(mivals))
        neighbors[t] = [mivals[i][2] for i in 1:k_eff]
    end

    # --- allocate PUC scores ----------------------------------------
    puc_scores = zeros(Float64, n, n)

    # --- multithreaded triplet loop with SIMD redundancy ------------
    Threads.@threads for x in 1:n
        if config.verbose && x % 500 == 0 && Threads.threadid() == 1
            println("[FastPIDC] Threads PUC progress: x = $x / $n")
        end
        
        for z in x+1:n
            pz = nodes[z].probabilities
            nz = length(pz)
            mi_xz = mi_matrix[x, z]
            inv_mi_xz = mi_xz > 0 ? 1.0 / mi_xz : 0.0
            sum_puc = 0.0
            
            # target z
            for y in neighbors[z]
                y == x && continue
                Rz = 0.0
                @inbounds for b in 1:nz
                    Rz += pz[b] * min(si_tensor[x, b, z], si_tensor[y, b, z])
                end
                score = (mi_xz - Rz) * inv_mi_xz
                if score > 0
                    sum_puc += score
                end
            end
            
            # target x
            px = nodes[x].probabilities
            nx = length(px)
            mi_zx = mi_matrix[z, x]
            inv_mi_zx = mi_zx > 0 ? 1.0 / mi_zx : 0.0
            for y in neighbors[x]
                y == z && continue
                Rx = 0.0
                @inbounds for b in 1:nx
                    Rx += px[b] * min(si_tensor[z, b, x], si_tensor[y, b, x])
                end
                score = (mi_zx - Rx) * inv_mi_zx
                if score > 0
                    sum_puc += score
                end
            end
            
            puc_scores[x, z] = sum_puc
            puc_scores[z, x] = sum_puc
        end
    end

    if config.verbose
        println("[FastPIDC] Finished PUC computation.")
    end

    return mi_matrix, puc_scores
end
