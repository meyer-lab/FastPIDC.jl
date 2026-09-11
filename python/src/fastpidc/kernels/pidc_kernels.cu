// Shared CUDA kernels for PIDC network inference.
//
// This is the canonical GPU implementation of the two compute-heavy stages of
// FastPIDC:
//
//   * The chunked PUC algorithm: for each chunk of "target" genes z, it (1)
//     accumulates joint bin-count histograms between every source gene x and
//     each target z in the chunk, (2) turns those histograms into pairwise
//     mutual information and specific-information values, and (3) accumulates
//     the resulting Proportional Unique Contribution (PUC) scores.
//   * The Bayesian-blocks dynamic program used to discretize each gene, with
//     one CUDA block per gene (see the second section below).
//
// It is written in plain CUDA C (no Julia- or Python-specific glue) so it
// is compiled and launched from either language's GPU bindings, both at
// runtime, from this one file:
//   * Python: loaded via `cupy.RawModule` (see `fastpidc.cuda`), which
//     invokes nvrtc to compile this source.
//   * Julia: the FastPIDC.jl CUDA extension (`ext/FastPIDCCUDAExt`) shells
//     out to `nvcc --ptx` to compile this source for the active device's
//     compute capability, then loads the result with `CUDA.CuModule` and
//     drives it with `CUDA.cudacall`.
//
// Both hosts pass 0-indexed bin ids, marginals and offsets in the shapes
// documented per kernel below; FastPIDC.jl is 1-indexed internally, so its host
// code shifts bin ids and block offsets down by one before upload, and shifts
// the Bayesian-block back-pointers back up after download. Only float scores
// otherwise cross that boundary.
//
// All arrays are indexed 0-based, row-major (C order), and passed as flat
// buffers with the shapes documented per kernel.
//
// Indexing contract: the scalar parameters (n, m, k_bins, z_start, z_chunk_size)
// are int32 on both hosts, but the BUFFERS ARE NOT BOUNDED BY INT32. `counts`
// alone holds k_bins^2 * n * z_chunk_size elements, which exceeds 2^31 on
// ordinary single-cell datasets - 12k genes x 33 bins x a 256-gene chunk is
// 3.4e9 elements, and computing that index in `int` wrapped it negative and
// faulted with CUDA_ERROR_ILLEGAL_ADDRESS. Every composed flat offset below is
// therefore `long long`, hoisted out of the hot loops as a base offset plus a
// loop-invariant stride so the inner bodies cost 64-bit adds rather than 32-bit
// multiply-add chains. The bin-pair term `u * k_bins + v` is widened before
// multiplication as well, so no composed flat-buffer offset relies on 32-bit
// arithmetic.

extern "C" {

// data:        (m, n) int32         -- data[s * n + x] = bin id of gene x, sample s
// counts:      (k_bins, k_bins, n, chunk) int32, zero-initialized by the caller
//              counts[((u * k_bins + v) * n + x) * chunk + z_local]
// Both of these routinely exceed 2^31 elements; see the indexing contract above.
// One thread handles one (x, z_local) pair, looping over all m samples.
__global__ void joint_counts_kernel(
    const int* __restrict__ data,
    int* counts,
    int n, int m, int k_bins,
    int z_start, int z_chunk_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z_local = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || z_local >= z_chunk_size) return;

    int z_global = z_start + z_local;
    if (z_global >= n || x == z_global) return;

    // 64-bit offsets: counts holds k_bins^2 * n * z_chunk_size elements, which
    // exceeds 2^31 on ordinary datasets. The strides are loop-invariant, so they
    // are hoisted here and the sample loop costs one widening multiply-add.
    const long long plane_stride = (long long)n * z_chunk_size;   // one (u, v) plane
    const long long cell = (long long)x * z_chunk_size + z_local; // this thread's (x, z_local)

    // Walking the sample row also keeps data's m * n index 64-bit. One
    // accumulator serves both reads, since they differ only by a fixed column.
    long long data_row = 0;

    for (int s = 0; s < m; ++s, data_row += n) {
        int u = data[data_row + x];
        int v = data[data_row + z_global];
        if (u >= 0 && u < k_bins && v >= 0 && v < k_bins) {
            // Widen before forming the bin-pair index: k_bins is an int32
            // launch scalar, but k_bins^2 need not fit in int32.
            const long long bin_pair = (long long)u * k_bins + v;
            long long idx = bin_pair * plane_stride + cell;
            atomicAdd(&counts[idx], 1);
        }
    }
}

// counts:      (k_bins, k_bins, n, chunk) int32, as produced above
// marginals:   (k_bins, n) float64        -- marginals[v * n + x] = P(gene x == bin v)
// mi_matrix:   (n, n) float64             -- mi_matrix[x * n + z]
// si_matrix:   (k_bins, n, chunk) float64 -- si_matrix[(v * n + x) * chunk + z_local]
//              specific information of source x with respect to target z_global,
//              at target bin v.
// counts may exceed 2^31 elements; see the indexing contract above.
// One thread handles one (x, z_local) pair.
__global__ void mi_si_kernel(
    const int* __restrict__ counts,
    const double* __restrict__ marginals,
    double* mi_matrix,
    double* si_matrix,
    int n, int m, int k_bins,
    int z_start, int z_chunk_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z_local = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || z_local >= z_chunk_size) return;

    int z_global = z_start + z_local;
    if (z_global >= n || x == z_global) return;

    // 64-bit offsets, hoisted once per thread. counts (k_bins, k_bins, n, chunk)
    // advances by plane_stride per v and u_stride per u; si_matrix (k_bins, n,
    // chunk) advances by plane_stride per v. Both loops then cost 64-bit adds
    // only. The advances live in the for-increment clause so the `continue`s
    // below cannot skip them.
    const long long plane_stride = (long long)n * z_chunk_size;
    const long long u_stride = plane_stride * k_bins;
    const long long cell = (long long)x * z_chunk_size + z_local;

    double inv_m = 1.0 / (double)m;
    double mi_val = 0.0;

    long long counts_v = cell;    // counts[((0 * k_bins + v) * n + x) * chunk + z_local]
    long long si_idx = cell;      // si_matrix[(v * n + x) * chunk + z_local]
    long long marg_z = z_global;  // marginals[v * n + z_global]

    for (int v = 0; v < k_bins; ++v,
         counts_v += plane_stride, si_idx += plane_stride, marg_z += n) {
        double p_z_v = marginals[marg_z];
        if (p_z_v <= 0.0) continue;

        double si_v = 0.0;
        long long counts_uv = counts_v;  // u = 0
        long long marg_x = x;            // marginals[u * n + x]
        for (int u = 0; u < k_bins; ++u, counts_uv += u_stride, marg_x += n) {
            double p_x_u = marginals[marg_x];
            if (p_x_u <= 0.0) continue;

            int c_uv = counts[counts_uv];
            double p_uv = (double)c_uv * inv_m;

            if (p_uv > 0.0) {
                mi_val += p_uv * log2(p_uv / (p_x_u * p_z_v));
                double p_u_cond_v = p_uv / p_z_v;
                si_v += p_u_cond_v * log2(p_u_cond_v / p_x_u);
            }
        }
        si_matrix[si_idx] = si_v;
    }

    mi_matrix[(long long)x * n + z_global] = mi_val;
}

// si_matrix:   (k_bins, n, chunk) float64, as produced above
// mi_matrix:   (n, n) float64
// puc_scores:  (n, n) float64 -- puc_scores[x * n + z_global] (one direction only;
//              the caller must symmetrize puc_scores[i,j] + puc_scores[j,i])
// marginals:   (k_bins, n) float64
// One thread handles one (x, z_local) pair, looping internally over source y.
__global__ void puc_accumulation_kernel(
    const double* __restrict__ si_matrix,
    const double* __restrict__ mi_matrix,
    double* puc_scores,
    const double* __restrict__ marginals,
    int n, int k_bins,
    int z_start, int z_chunk_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z_local = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || z_local >= z_chunk_size) return;

    int z_global = z_start + z_local;
    if (z_global >= n || x == z_global) return;

    double mi_xz = mi_matrix[(long long)x * n + z_global];
    if (mi_xz <= 1e-12) return;

    // si_matrix is (k_bins, n, chunk): one bin plane is plane_stride apart, one
    // source gene is z_chunk_size apart. Both walks are 64-bit, and the advances
    // live in the for-increment clauses so the `continue`s cannot skip them.
    const long long plane_stride = (long long)n * z_chunk_size;
    const long long si_x_cell = (long long)x * z_chunk_size + z_local;

    double local_puc = 0.0;
    long long si_y_cell = z_local;  // y = 0: (0 * n + y) * chunk + z_local

    for (int y = 0; y < n; ++y, si_y_cell += z_chunk_size) {
        if (y == x || y == z_global) continue;

        double redundancy = 0.0;
        long long si_x_idx = si_x_cell;
        long long si_y_idx = si_y_cell;
        long long marg_idx = z_global;  // marginals[k * n + z_global]

        for (int k = 0; k < k_bins; ++k,
             si_x_idx += plane_stride, si_y_idx += plane_stride, marg_idx += n) {
            double p_z_k = marginals[marg_idx];
            if (p_z_k <= 0.0) continue;

            double si_x = si_matrix[si_x_idx];
            double si_y = si_matrix[si_y_idx];
            redundancy += p_z_k * fmin(si_x, si_y);
        }

        double score = (mi_xz - redundancy) / mi_xz;
        if (isfinite(score) && score > 0.0) {
            local_puc += score;
        }
    }

    puc_scores[(long long)x * n + z_global] = local_puc;
}

} // extern "C"

// ---------------------------------------------------------------------------
// Bayesian blocks dynamic program
// ---------------------------------------------------------------------------
//
// Exact Bayesian-blocks segmentation (Scargle 2012) of one gene's sorted,
// unique-value-collapsed observations. One CUDA block handles one gene.
//
// Endpoints K stay sequential because best[K] depends on earlier endpoints,
// while the threads within a block evaluate the candidate block starts i <= K
// in parallel and reduce them with a deterministic first-maximum rule: a
// higher score wins, and an exact tie takes the smaller i. That reproduces the
// CPU reference's strict-`>` left-to-right scan for any launch geometry, so
// CPU and GPU agree on the selected partition bit for bit.
//
// Several genes' problems are packed back to back into flat buffers; each
// gene's slices start at state_offsets[gene] (arrays with one entry per unique
// value) and block_offsets[gene] (block lengths, one entry per unique value
// plus one). Both offsets are 0-based, like every other index in this file.
//
// The prefix-count and back-pointer element types are chosen per batch by the
// host, from the observation count and the largest U_g respectively, so wide
// integers are not paid for on datasets that do not need them. C has no
// generics, so the entry points below are macro-generated per (count,
// back-pointer) type pair; the host picks the matching name.
//
// prefix_counts: (total_states,) CountT   -- cumulative multiplicities per gene
// block_lengths: (total_states + n_genes,) float64 -- edges[end] - edges[j]
// state_offsets: (n_genes,) int64         -- 0-based start in prefix/best/last
// block_offsets: (n_genes,) int64         -- 0-based start in block_lengths
// unique_counts: (n_genes,) int32         -- U_g per gene
// best:          (total_states,) float64  -- output, per-endpoint objective
// last:          (total_states,) IndexT   -- output, 0-based predecessor index
// final_scores:  (n_genes,) float64       -- output, objective at endpoint U_g
// priors:        (>= max U_g,) float64    -- prior per endpoint, host-computed
//
// Launch with blocks = n_genes and threads in {32, 64, 128, 256}.

#define FASTPIDC_BB_MAX_THREADS 256
#define FASTPIDC_BB_MAX_WARPS (FASTPIDC_BB_MAX_THREADS / 32)
#define FASTPIDC_BB_NO_CANDIDATE 0x7FFFFFFF

// Built as a bit pattern rather than via <math.h>'s INFINITY / <limits.h>:
// nvrtc (which cupy uses) compiles without the host headers nvcc pulls in, and
// this file must build identically under both.
__device__ __forceinline__ double fastpidc_negative_infinity()
{
    return __longlong_as_double((long long)0xFFF0000000000000ULL);
}

// Higher score wins; an exact tie keeps the smaller candidate index.
__device__ __forceinline__ bool fastpidc_bb_take_other(
    double other_score, int other_i, double current_score, int current_i)
{
    return other_score > current_score ||
           (other_score == current_score && other_i < current_i);
}

template <typename CountT, typename IndexT>
__device__ void fastpidc_bayesian_blocks_dp(
    const CountT* __restrict__ prefix_counts,
    const double* __restrict__ block_lengths,
    const long long* __restrict__ state_offsets,
    const long long* __restrict__ block_offsets,
    const int* __restrict__ unique_counts,
    double* best,
    IndexT* last,
    double* final_scores,
    const double* __restrict__ priors)
{
    // One candidate per thread, then one reduced candidate per warp. Shared
    // memory rather than shuffles keeps the reduction order fixed and explicit.
    __shared__ double thread_scores[FASTPIDC_BB_MAX_THREADS];
    __shared__ int thread_indices[FASTPIDC_BB_MAX_THREADS];
    __shared__ double warp_scores[FASTPIDC_BB_MAX_WARPS];
    __shared__ int warp_indices[FASTPIDC_BB_MAX_WARPS];

    const int gene = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarps = nthreads >> 5;

    const long long state_start = state_offsets[gene];
    const long long block_start = block_offsets[gene];
    const int n_unique = unique_counts[gene];

    // Match the CPU reference's singleton behavior explicitly. The generic
    // event-fitness expression has zero block width when U_g == 1, whereas a
    // constant gene should deterministically return its two outer edges and an
    // objective score of zero. Every thread returns, so no barrier diverges.
    if (n_unique == 1) {
        if (tid == 0) {
            best[state_start] = 0.0;
            last[state_start] = (IndexT)0;
            final_scores[gene] = 0.0;
        }
        return;
    }

    for (int k = 0; k < n_unique; ++k) {
        const double block_length_end = block_lengths[block_start + k + 1];
        const double prefix_k = (double)prefix_counts[state_start + k];
        const double prior = priors[k];

        double local_best = fastpidc_negative_infinity();
        int local_i = FASTPIDC_BB_NO_CANDIDATE;

        // Use a 64-bit loop cursor so the final `i += nthreads` cannot wrap if
        // n_unique approaches the signed-int32 ABI ceiling. Candidate values
        // themselves are still <= INT32_MAX and are narrowed only after checking.
        for (long long i64 = tid; i64 <= (long long)k; i64 += nthreads) {
            const int i = (int)i64;
            const double prefix_before =
                (i == 0) ? 0.0 : (double)prefix_counts[state_start + i64 - 1];
            const double count = prefix_k - prefix_before;
            const double width = block_lengths[block_start + i64] - block_length_end;

            // Fitness function (eq. 19) and prior (eq. 21) from Scargle 2012.
            double fit = count * log(count / width) - prior;
            if (i > 0) {
                fit += best[state_start + i64 - 1];
            }

            if (fastpidc_bb_take_other(fit, i, local_best, local_i)) {
                local_best = fit;
                local_i = i;
            }
        }

        thread_scores[tid] = local_best;
        thread_indices[tid] = local_i;
        __syncthreads();

        // Each warp leader scans its 32 thread-local candidates in a fixed
        // order, so ties still resolve to the smaller i regardless of how the
        // candidates were distributed across threads.
        if (lane == 0) {
            const int warp_start = warp * 32;
            double warp_best = thread_scores[warp_start];
            int warp_i = thread_indices[warp_start];
            for (int slot = warp_start + 1; slot < warp_start + 32; ++slot) {
                if (fastpidc_bb_take_other(
                        thread_scores[slot], thread_indices[slot], warp_best, warp_i)) {
                    warp_best = thread_scores[slot];
                    warp_i = thread_indices[slot];
                }
            }
            warp_scores[warp] = warp_best;
            warp_indices[warp] = warp_i;
        }
        __syncthreads();

        if (tid == 0) {
            double block_best = warp_scores[0];
            int block_i = warp_indices[0];
            for (int slot = 1; slot < nwarps; ++slot) {
                if (fastpidc_bb_take_other(
                        warp_scores[slot], warp_indices[slot], block_best, block_i)) {
                    block_best = warp_scores[slot];
                    block_i = warp_indices[slot];
                }
            }
            best[state_start + k] = block_best;
            // IndexT was selected by the host from max(U_g), so this narrowing
            // conversion is exact.
            last[state_start + k] = (IndexT)block_i;
            if (k == n_unique - 1) {
                final_scores[gene] = block_best;
            }
        }

        // best[k] lives in global memory and every thread reads it at the next
        // endpoint, so the write must be visible before advancing k.
        __syncthreads();
    }
}

#define FASTPIDC_BB_KERNEL(NAME, COUNT_T, INDEX_T)                             \
    extern "C" __global__ void NAME(                                           \
        const COUNT_T* __restrict__ prefix_counts,                             \
        const double* __restrict__ block_lengths,                              \
        const long long* __restrict__ state_offsets,                           \
        const long long* __restrict__ block_offsets,                           \
        const int* __restrict__ unique_counts,                                 \
        double* best,                                                          \
        INDEX_T* last,                                                         \
        double* final_scores,                                                  \
        const double* __restrict__ priors)                                     \
    {                                                                          \
        fastpidc_bayesian_blocks_dp<COUNT_T, INDEX_T>(                         \
            prefix_counts, block_lengths, state_offsets, block_offsets,        \
            unique_counts, best, last, final_scores, priors);                  \
    }

// U_g never exceeds the observation count, so the back-pointer type is never
// wider than the prefix-count type; only those pairs are instantiated.
FASTPIDC_BB_KERNEL(bayesian_blocks_dp_u8_u8, unsigned char, unsigned char)
FASTPIDC_BB_KERNEL(bayesian_blocks_dp_u16_u8, unsigned short, unsigned char)
FASTPIDC_BB_KERNEL(bayesian_blocks_dp_u16_u16, unsigned short, unsigned short)
FASTPIDC_BB_KERNEL(bayesian_blocks_dp_u32_u8, unsigned int, unsigned char)
FASTPIDC_BB_KERNEL(bayesian_blocks_dp_u32_u16, unsigned int, unsigned short)
FASTPIDC_BB_KERNEL(bayesian_blocks_dp_u32_u32, unsigned int, unsigned int)
FASTPIDC_BB_KERNEL(bayesian_blocks_dp_u64_u8, unsigned long long, unsigned char)
FASTPIDC_BB_KERNEL(bayesian_blocks_dp_u64_u16, unsigned long long, unsigned short)
FASTPIDC_BB_KERNEL(bayesian_blocks_dp_u64_u32, unsigned long long, unsigned int)
FASTPIDC_BB_KERNEL(bayesian_blocks_dp_u64_u64, unsigned long long, unsigned long long)
