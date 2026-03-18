# src/puc_pruned.jl
#
# Pruned PUC: use MI-based k-neighborhoods instead of all triplets.
# This is opt-in via PIDCConfig.triplet_block_k > 0.
#

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

    # --- build NodePair cache (MI + specific information) ------------

    node_pairs = Array{NodePair}(undef, n, n)

    function get_mi_and_si(node1::Node, node2::Node)
        probabilities, probabilities1, probabilities2 =
        get_joint_probabilities(node1, node2, estimator)

        mi = apply_mutual_information_formula(
        probabilities, probabilities1, probabilities2, base)

        si1 = apply_specific_information_formula(
        probabilities, probabilities1, probabilities2, 1, base)

        si2 = apply_specific_information_formula(
        probabilities, probabilities2, probabilities1, 2, base)

        return mi, si1, si2
    end

    # Turn NodePair cache into a dense MI matrix (symmetric).
    function nodepairs_to_mi(node_pairs::Array{NodePair,2})
        n = size(node_pairs, 1)
        mi = zeros(Float64, n, n)
        for i in 1:n
            mi[i,i] = 0.0
            for j in i+1:n
                m = node_pairs[i,j].mi
                mi[i,j] = m
                mi[j,i] = m
            end
        end
        return mi
    end

    for i in 1:n
        for j in i+1:n
            mi, si1, si2 = get_mi_and_si(nodes[i], nodes[j])
            node_pairs[i, j] = NodePair(mi, si1)  # source = i, target = j
            node_pairs[j, i] = NodePair(mi, si2)  # source = j, target = i
        end
    end

    # --- MI-based neighbor lists for each gene -----------------------

    neighbors = Vector{Vector{Int}}(undef, n)

    for t in 1:n
        mivals = Vector{Tuple{Float64,Int}}()
        sizehint!(mivals, n - 1)

        for u in 1:n
            u == t && continue
            mi_tu = node_pairs[t, u].mi
            push!(mivals, (mi_tu, u))
        end

        sort!(mivals; by = x -> x[1], rev = true)
        k_eff = min(k, length(mivals))
        neighbors[t] = [mivals[i][2] for i in 1:k_eff]
    end

    # --- Build full MI matrix from NodePair cache --------------------
    
    mi_scores = nodepairs_to_mi(node_pairs)

    # --- allocate PUC scores ----------------------------------------

    puc_scores = zeros(Float64, n, n)

    # Same clamping behavior as legacy increment_puc_scores.
    function increment_puc_scores!(x::Int, z::Int, mi::Float64, redundancy::Float64,
            scores::AbstractMatrix{Float64})
        puc_score = (mi - redundancy) / mi
        puc_score = isfinite(puc_score) && puc_score >= 0 ? puc_score : zero(puc_score)
        scores[x, z] += puc_score
        scores[z, x] += puc_score
    end

    # ---------------------- target-centric --------------------
    #
    # For an edge (x,z):
    #   - Redundancy around z uses ONLY neighbors[z] as candidate y.
    #   - Redundancy around x uses ONLY neighbors[x] as candidate y.
    #
    # When k >= n-1, neighbors[t] contains all other genes, so this
    # reproduces full PUC exactly (every triple {x,y,z} is seen once
    # for target=z and once for target=x, as in compute_puc_full).

        Threads.@threads for x in 1:n
            if config.verbose && x % 500 == 0 && Threads.threadid() == 1
                println("[FastPIDC] Threads PUC progress: x = $x / $n")
            end
    
            for z in x+1:n
                # --- target = z, sources = (x, y), y in neighbors[z] ----------
                for y in neighbors[z]
                    (y == x || y == z) && continue
    
                    np_xz = node_pairs[x, z]
                    np_yz = node_pairs[y, z]
    
                    Rz = apply_redundancy_formula(
                        nodes[z].probabilities,
                        np_xz.si,
                        np_yz.si,
                        base,
                    )
                    increment_puc_scores!(x, z, np_xz.mi, Rz, puc_scores)
                end
    
                # --- target = x, sources = (y, z), y in neighbors[x] ----------
                for y in neighbors[x]
                    (y == x || y == z) && continue
    
                    np_yx = node_pairs[y, x]
                    np_zx = node_pairs[z, x]
    
                    Rx = apply_redundancy_formula(
                        nodes[x].probabilities,
                        np_yx.si,
                        np_zx.si,
                        base,
                    )
                    increment_puc_scores!(x, z, np_zx.mi, Rx, puc_scores)
                end
            end
        end
    
        if config.verbose
            println("[FastPIDC] Finished PUC computation.")
        end
    
        return mi_scores, puc_scores
    end
