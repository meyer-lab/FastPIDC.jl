# src/puc_pruned_dist.jl
#
# Pruned PUC with MI-based k-neighborhoods, distributed over processes
# (outer loop over x). Same pruning semantics as compute_puc_pruned.

function compute_puc_pruned_dist(nodes::Vector{Node};
    estimator::String = "maximum_likelihood",
    base::Int = 2,
    config::PIDCConfig)

    n = length(nodes)
    k = min(config.triplet_block_k, max(n - 1, 0))

    # Fallback: disabled pruning → full PUC
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

    # Fill NodePair cache
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

    # --- allocate shared PUC scores ----------------------------------

    # Shared across processes; safe because each unordered (x,z) is
    # handled by exactly one process (z > x, outer loop partition).
    puc_scores = SharedArray{Float64}(n, n)

    # Same clamping behavior
    @inline function increment_puc_scores!(x::Int, z::Int,
                                           mi::Float64, redundancy::Float64,
                                           scores)
        puc_score = (mi - redundancy) / mi
        puc_score = isfinite(puc_score) && puc_score >= 0 ? puc_score : zero(puc_score)
        scores[x, z] += puc_score
        scores[z, x] += puc_score
    end

    mode = getfield(config, :neighbor_mode)

    if mode == :union
        # ---------------------- Option A: edge-centric ----------------
        @sync @distributed for x in 1:n
            if config.verbose && x % 500 == 0
                println("[FastPIDC] Distributed PUC progress: x = $x / $n")

            end
            for z in x+1:n
                # K(x,z) = neighbors[x] u neighbors[z] (deduplicated)
                cands = Vector{Int}()
                sizehint!(cands, length(neighbors[x]) + length(neighbors[z]))
                append!(cands, neighbors[x])
                append!(cands, neighbors[z])
                sort!(cands)

                last = 0
                for y in cands
                    # Skip duplicates and self/endpoint genes
                    if y == last || y == x || y == z
                        last = y
                        continue
                    end
                    last = y

                    # target = z, sources = (x, y)
                    np_xz = node_pairs[x, z]
                    np_yz = node_pairs[y, z]

                    Rz = apply_redundancy_formula(
                        nodes[z].probabilities,
                        np_xz.si,
                        np_yz.si,
                        base
                    )
                    increment_puc_scores!(x, z, np_xz.mi, Rz, puc_scores)

                    # target = x, sources = (y, z)
                    np_yx = node_pairs[y, x]
                    np_zx = node_pairs[z, x]

                    Rx = apply_redundancy_formula(
                        nodes[x].probabilities,
                        np_yx.si,
                        np_zx.si,
                        base
                    )
                    increment_puc_scores!(x, z, np_zx.mi, Rx, puc_scores)
                end
            end
        end

    elseif mode == :target
        # ---------------------- Option B: target-centric --------------
        @sync @distributed for x in 1:n
            if config.verbose && x % 500 == 0
                println("[FastPIDC] Distributed PUC progress: x = $x / $n")
            end
            for z in x+1:n
                # target = z, sources = (x, y), y in neighbors[z]
                for y in neighbors[z]
                    (y == x || y == z) && continue

                    np_xz = node_pairs[x, z]
                    np_yz = node_pairs[y, z]

                    Rz = apply_redundancy_formula(
                        nodes[z].probabilities,
                        np_xz.si,
                        np_yz.si,
                        base
                    )
                    increment_puc_scores!(x, z, np_xz.mi, Rz, puc_scores)
                end

                # target = x, sources = (y, z), y in neighbors[x]
                for y in neighbors[x]
                    (y == x || y == z) && continue

                    np_yx = node_pairs[y, x]
                    np_zx = node_pairs[z, x]

                    Rx = apply_redundancy_formula(
                        nodes[x].probabilities,
                        np_yx.si,
                        np_zx.si,
                        base
                    )
                    increment_puc_scores!(x, z, np_zx.mi, Rx, puc_scores)
                end
            end
        end
    else
        error("Unknown neighbor_mode = $(mode); expected :union or :target")
    end

    # Convert SharedArray → regular Matrix for downstream code
    return mi_scores, Array(puc_scores)
end
