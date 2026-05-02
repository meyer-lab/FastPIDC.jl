module FastPIDCMetalExt

using FastPIDC
using Metal
using Atomix

# --- Kernels ---

"""
    joint_counts_kernel(data, target_id, counts, n, m, k_bins)
"""
function joint_counts_kernel(data, target_id, counts, n, m, k_bins)
    # Grid: (num_samples, num_nodes)
    s = thread_position_in_grid().x
    x = thread_position_in_grid().y
    
    if s > m || x > n; return nothing; end
    
    u = data[s, x]
    v = data[s, target_id]
    
    if u >= 1 && u <= k_bins && v >= 1 && v <= k_bins
        # Flat index for counts[u, v, x]
        idx = UInt32((x-1)*k_bins*k_bins + (v-1)*k_bins + u)
        # Atomic increment in global memory
        Atomix.@atomic counts[idx] += Int32(1)
    end
    
    return nothing
end

"""
    mi_si_from_counts_kernel(counts, marginals, target_id, mi_out, si_out, n, m, k_bins)
"""
function mi_si_from_counts_kernel(counts, marginals, target_id, mi_out, si_out, n, m, k_bins)
    x = thread_position_in_grid().x
    if x > n; return nothing; end
    
    inv_m = 1.0f0 / Float32(m)
    mi_val = 0.0f0
    
    for v in 1:k_bins
        p_z_v = marginals[v, target_id]
        if p_z_v <= 0.0f0; continue; end
        
        si_v = 0.0f0
        for u in 1:k_bins
            p_x_u = marginals[u, x]
            if p_x_u <= 0.0f0; continue; end
            
            c_uv = counts[(x-1)*k_bins*k_bins + (v-1)*k_bins + u]
            p_uv = Float32(c_uv) * inv_m
            
            if p_uv > 0.0f0
                mi_val += p_uv * log2(p_uv / (p_x_u * p_z_v))
                p_u_cond_v = p_uv / p_z_v
                si_v += p_u_cond_v * log2(p_u_cond_v / p_x_u)
            end
        end
        si_out[v, x] = si_v
    end
    
    mi_out[x] = mi_val
    return nothing
end

"""
    puc_accumulation_kernel(si_batch, mi_batch, puc_scores, z, n, k_bins, p_z)
"""
function puc_accumulation_kernel(si_batch, mi_batch, puc_scores, z, n, k_bins, p_z)
    x = thread_position_in_grid().x
    if x > n || x == z; return nothing; end
    
    mi_xz = mi_batch[x]
    if mi_xz <= 0.0f0; return nothing; end
    
    local_puc = 0.0f0
    for y in 1:n
        if y == x || y == z; continue; end
        
        redundancy = 0.0f0
        for k in 1:k_bins
            # Williams & Beer: min(SI(x;z_k), SI(y;z_k))
            redundancy += p_z[k] * min(si_batch[k, x], si_batch[k, y])
        end
        
        score = (mi_xz - redundancy) / mi_xz
        if isfinite(score) && score > 0.0f0
            local_puc += score
        end
    end
    
    puc_scores[x, z] += local_puc
    return nothing
end

# --- Host Implementation ---

function FastPIDC.compute_puc_full_metal(nodes, config, base)
    num_nodes = length(nodes)
    num_samples = length(nodes[1].binned_values)
    k_bins = maximum(n -> n.number_of_bins, nodes)
    
    # 1. Prepare data on GPU
    data_cpu = zeros(Int32, num_samples, num_nodes)
    marginals_cpu = zeros(Float32, k_bins, num_nodes)
    for i in 1:num_nodes
        data_cpu[:, i] .= Int32.(nodes[i].binned_values)
        p = nodes[i].probabilities
        marginals_cpu[1:length(p), i] .= Float32.(p)
    end
    
    data_gpu = MtlArray(data_cpu)
    marginals_gpu = MtlArray(marginals_cpu)
    
    # Result matrices
    puc_scores_gpu = MtlArray(zeros(Float32, num_nodes, num_nodes))
    mi_matrix_cpu = zeros(Float64, num_nodes, num_nodes)
    
    # Target-specific buffers
    # counts_batch: (k_bins, k_bins, num_nodes)
    counts_batch_gpu = MtlArray(zeros(Int32, k_bins * k_bins * num_nodes))
    si_batch_gpu = MtlArray(zeros(Float32, k_bins, num_nodes))
    mi_batch_gpu = MtlArray(zeros(Float32, num_nodes))
    
    println("[FastPIDC] GPU Batched PUC: Processing $num_nodes targets...")
    
    for z in 1:num_nodes
        if config.verbose && z % 100 == 0
            println("  Target $z / $num_nodes")
        end
        
        # Step 0: Clear counts
        fill!(counts_batch_gpu, Int32(0))
        
        # Step 1: Joint Counts
        gs_1 = (16, 16, 1)
        gr_1 = (div(num_samples + 15, 16), div(num_nodes + 15, 16), 1)
        @metal threads=gs_1 groups=gr_1 joint_counts_kernel(
            data_gpu, Int32(z), counts_batch_gpu,
            Int32(num_nodes), Int32(num_samples), Int32(k_bins)
        )
        
        # Step 2: MI and SI
        gs_2 = (256, 1, 1)
        gr_2 = (div(num_nodes + 255, 256), 1, 1)
        @metal threads=gs_2 groups=gr_2 mi_si_from_counts_kernel(
            counts_batch_gpu, marginals_gpu, Int32(z), mi_batch_gpu, si_batch_gpu,
            Int32(num_nodes), Int32(num_samples), Int32(k_bins)
        )
        
        # Capture MI row for final output
        # (Could be optimized by moving the whole matrix at once, but this is fine)
        mi_matrix_cpu[:, z] .= Float64.(Array(mi_batch_gpu))
        
        # Step 3: PUC Accumulation
        p_z_gpu = view(marginals_gpu, :, z)
        @metal threads=gs_2 groups=gr_2 puc_accumulation_kernel(
            si_batch_gpu, mi_batch_gpu, puc_scores_gpu, Int32(z),
            Int32(num_nodes), Int32(k_bins), p_z_gpu
        )
    end
    
    # 3. Copy results back
    puc_scores_cpu = Array(puc_scores_gpu)
    
    # Symmetrize PUC scores
    # Each cell (x, z) currently only contains contributions where z was the target.
    # We need to add the cases where x was the target.
    for i in 1:num_nodes
        for j in i+1:num_nodes
            val = puc_scores_cpu[i, j] + puc_scores_cpu[j, i]
            puc_scores_cpu[i, j] = val
            puc_scores_cpu[j, i] = val
        end
    end
    
    return mi_matrix_cpu, puc_scores_cpu
end

end # module
