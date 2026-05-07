module FastPIDCCUDAExt

using FastPIDC
using CUDA
using Atomix

# --- Kernels ---

"""
    joint_counts_kernel_batched(data, counts, n, m, k_bins)
    Grid: (n, n)
"""
function joint_counts_kernel_batched(data, counts, n, m, k_bins)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    z = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if x > n || z > n
        ;
        return nothing;
    end
    if x == z
        ;
        return nothing;
    end

    # Each thread computes joint counts for pair (x, z)
    for s = 1:m
        u = data[s, x]
        v = data[s, z]
        if u >= 1 && u <= k_bins && v >= 1 && v <= k_bins
            # idx for counts[u, v, x, z]
            idx = UInt64((z-1)*n*k_bins*k_bins + (x-1)*k_bins*k_bins + (v-1)*k_bins + u)
            counts[idx] += Int32(1)
        end
    end

    return nothing
end

"""
    mi_si_kernel_batched(counts, marginals, mi_matrix, si_matrix, n, m, k_bins)
    Grid: (n, n)
"""
function mi_si_kernel_batched(counts, marginals, mi_matrix, si_matrix, n, m, k_bins)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    z = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if x > n || z > n || x == z
        ;
        return nothing;
    end

    inv_m = 1.0 / Float64(m)
    mi_val = 0.0

    for v = 1:k_bins
        p_z_v = marginals[v, z]
        if p_z_v <= 0.0
            ;
            continue;
        end

        si_v = 0.0
        for u = 1:k_bins
            p_x_u = marginals[u, x]
            if p_x_u <= 0.0
                ;
                continue;
            end

            c_uv = counts[(z-1)*n*k_bins*k_bins+(x-1)*k_bins*k_bins+(v-1)*k_bins+u]
            p_uv = Float64(c_uv) * inv_m

            if p_uv > 0.0
                mi_val += p_uv * log2(p_uv / (p_x_u * p_z_v))
                p_u_cond_v = p_uv / p_z_v
                si_v += p_u_cond_v * log2(p_u_cond_v / p_x_u)
            end
        end
        si_matrix[v, x, z] = si_v
    end

    mi_matrix[x, z] = mi_val
    return nothing
end

"""
    puc_accumulation_kernel_batched(si_matrix, mi_matrix, puc_scores, marginals, n, k_bins)
    Grid: (n, n)
"""
function puc_accumulation_kernel_batched(
    si_matrix,
    mi_matrix,
    puc_scores,
    marginals,
    n,
    k_bins,
)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    z = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if x > n || z > n || x == z
        ;
        return nothing;
    end

    mi_xz = mi_matrix[x, z]
    if mi_xz <= 1e-12
        ;
        return nothing;
    end

    local_puc = 0.0
    for y = 1:n
        if y == x || y == z
            ;
            continue;
        end

        redundancy = 0.0
        for k = 1:k_bins
            p_z_k = marginals[k, z]
            if p_z_k <= 0.0
                ;
                continue;
            end
            # Williams & Beer: min(SI(x;z_k), SI(y;z_k))
            redundancy += p_z_k * min(si_matrix[k, x, z], si_matrix[k, y, z])
        end

        score = (mi_xz - redundancy) / mi_xz
        if isfinite(score) && score > 0.0
            local_puc += score
        end
    end

    puc_scores[x, z] = local_puc
    return nothing
end

# --- Host Implementation ---

function FastPIDC.compute_puc_full_cuda(nodes, config, base)
    num_nodes = length(nodes)
    num_samples = length(nodes[1].binned_values)
    k_bins = maximum(n -> n.number_of_bins, nodes)

    # 1. Prepare data on GPU
    data_cpu = zeros(Int64, num_samples, num_nodes)
    marginals_cpu = zeros(Float64, k_bins, num_nodes)
    for i = 1:num_nodes
        data_cpu[:, i] .= Int32.(nodes[i].binned_values)
        p = nodes[i].probabilities
        marginals_cpu[1:length(p), i] .= Float64.(p)
    end

    data_gpu = CuArray(data_cpu)
    marginals_gpu = CuArray(marginals_cpu)

    # Global result matrices
    puc_scores_gpu = CuArray(zeros(Float64, num_nodes, num_nodes))
    mi_matrix_gpu = CuArray(zeros(Float64, num_nodes, num_nodes))

    # Large intermediate matrices
    counts_matrix_gpu = CuArray(zeros(Int64, k_bins * k_bins * num_nodes * num_nodes))
    si_matrix_gpu = CuArray(zeros(Float64, k_bins, num_nodes, num_nodes))

    if config.verbose
        println(
            "[FastPIDC] GPU Fully Batched PUC: Processing $num_nodes x $num_nodes pairs...",
        )
    end

    # Step 1: Joint Counts
    gs = (16, 16)
    gr = (div(num_nodes + 15, 16), div(num_nodes + 15, 16))

    @cuda threads=gs blocks=gr joint_counts_kernel_batched(
        data_gpu,
        counts_matrix_gpu,
        Int32(num_nodes),
        Int32(num_samples),
        Int32(k_bins),
    )

    # Step 2: MI and SI
    @cuda threads=gs blocks=gr mi_si_kernel_batched(
        counts_matrix_gpu,
        marginals_gpu,
        mi_matrix_gpu,
        si_matrix_gpu,
        Int32(num_nodes),
        Int32(num_samples),
        Int32(k_bins),
    )

    # Step 3: PUC Accumulation
    @cuda threads=gs blocks=gr puc_accumulation_kernel_batched(
        si_matrix_gpu,
        mi_matrix_gpu,
        puc_scores_gpu,
        marginals_gpu,
        Int32(num_nodes),
        Int32(k_bins),
    )

    # 3. Copy results back
    puc_scores_cpu = Array(puc_scores_gpu)
    mi_matrix_cpu = Array(mi_matrix_gpu)

    # Symmetrize PUC scores
    for i = 1:num_nodes
        for j = (i+1):num_nodes
            val = puc_scores_cpu[i, j] + puc_scores_cpu[j, i]
            puc_scores_cpu[i, j] = val
            puc_scores_cpu[j, i] = val
        end
    end

    return mi_matrix_cpu, puc_scores_cpu
end

end # module
