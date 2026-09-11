"""GPU-accelerated PUC scoring and Bayesian-blocks discretization.

Host-side driver for the shared CUDA kernels in
``fastpidc/kernels/pidc_kernels.cu`` (see that file for the algorithm and
the sharing strategy with FastPIDC.jl's CUDA extension). Requires the
optional ``cupy`` dependency (install with ``pip install fastpidc[cuda]``)
and a functional GPU.

Genes are processed along the target ("z") axis in chunks sized to fit the
currently free device memory, mirroring
``FastPIDCCUDAExt.compute_puc_full_cuda`` in FastPIDC.jl.

Those chunked buffers legitimately exceed 2^31 elements on large gene sets
(``counts`` alone is ``k_bins**2 * n * chunk_size``), so the shared kernels
index them in 64-bit; see the indexing contract in the kernel source.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from .discretizers import BayesianBlocksProblem, BayesianBlocksSolution
from .types import Node

__all__ = ["compute_puc_full_cuda", "cuda_available", "solve_bayesian_blocks_cuda"]

_KERNEL_SOURCE_PATH = Path(__file__).with_name("kernels") / "pidc_kernels.cu"
_MAX_CHUNK_SIZE = 256
_KERNEL_INT_MAX = np.iinfo(np.int32).max
_GPU_MEMORY_BUDGET_NUMERATOR = 65
_GPU_MEMORY_BUDGET_DENOMINATOR = 100
_FALLBACK_CUDA_HEADER_DIRS = ("/usr", "/usr/local/cuda")


def _ensure_cuda_path() -> None:
    """Best-effort fallback for systems (e.g. some Debian/Ubuntu CUDA
    installs) where the CUDA toolkit headers live under a system prefix
    that cupy's nvrtc backend does not probe automatically.

    Must run before cupy is first imported: cupy resolves its include-path
    search once (effectively at import time), so setting ``CUDA_PATH`` any
    later has no effect. This is why it is invoked at import time below,
    rather than lazily inside ``_load_module``.
    """
    if "CUDA_PATH" in os.environ:
        return
    for candidate in _FALLBACK_CUDA_HEADER_DIRS:
        if (Path(candidate) / "include" / "cuda_runtime.h").exists():
            os.environ["CUDA_PATH"] = candidate
            return


_ensure_cuda_path()


def cuda_available() -> bool:
    """Whether ``cupy`` is importable and reports a functional GPU."""
    try:
        import cupy as cp
    except ImportError:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def _load_module():
    import cupy as cp

    _ensure_cuda_path()
    source = _KERNEL_SOURCE_PATH.read_text()
    return cp.RawModule(code=source, options=("--std=c++11",))


def _gpu_memory_budget_bytes(free_bytes: int) -> int:
    if free_bytes <= 0:
        raise ValueError("free_bytes must be positive")
    return free_bytes * _GPU_MEMORY_BUDGET_NUMERATOR // _GPU_MEMORY_BUDGET_DENOMINATOR


def _reusable_gpu_memory_bytes(
    driver_free_bytes: int,
    pool_free_bytes: int = 0,
    *,
    pool_used_bytes: int = 0,
    pool_limit_bytes: int = 0,
) -> int:
    """Memory this process can reuse without exceeding the device/pool limits."""
    for name, value in (
        ("driver_free_bytes", driver_free_bytes),
        ("pool_free_bytes", pool_free_bytes),
        ("pool_used_bytes", pool_used_bytes),
        ("pool_limit_bytes", pool_limit_bytes),
    ):
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")

    available = driver_free_bytes + pool_free_bytes
    if pool_limit_bytes:
        available = min(available, max(pool_limit_bytes - pool_used_bytes, 0))
    return available


def _available_gpu_memory_bytes(cp) -> int:
    """Return VRAM reusable by CuPy, including cached free pool blocks.

    ``cudaMemGetInfo`` does not include blocks CuPy has already reserved but is
    no longer using. Counting those blocks prevents repeated FastPIDC calls from
    becoming artificially more conservative. A configured CuPy pool limit is
    also honored so the planner cannot approve a batch the allocator will reject.
    """
    driver_free_bytes = int(cp.cuda.runtime.memGetInfo()[0])
    pool = cp.get_default_memory_pool()
    return _reusable_gpu_memory_bytes(
        driver_free_bytes,
        int(pool.free_bytes()),
        pool_used_bytes=int(pool.used_bytes()),
        pool_limit_bytes=int(pool.get_limit()),
    )


def _check_kernel_scalar_limits(n: int, m: int, k_bins: int) -> None:
    """Validate the signed-int32 launch scalars used by the shared kernels.

    Every composed flat-buffer offset is 64-bit. Only the ABI-level dimensions
    remain int32, so reject impossible values before numpy/cupy can narrow them.
    """
    for name, value in (("n", n), ("m", m), ("k_bins", k_bins)):
        if value < 1:
            raise ValueError(f"{name} must be positive; got {value}")
        if value > _KERNEL_INT_MAX:
            raise ValueError(
                f"compute_puc_full_cuda: {name}={value} exceeds the CUDA kernel's "
                f"signed 32-bit scalar limit of {_KERNEL_INT_MAX}."
            )


def _puc_memory_plan(
    n: int,
    m: int,
    k_bins: int,
    free_bytes: int,
    *,
    requested_chunk_size: int | None = None,
) -> tuple[int, int, int, int]:
    """Plan the entire PUC device footprint against 65% of currently reusable VRAM.

    Returns ``(chunk_size, fixed_bytes, bytes_per_chunk_column, budget_bytes)``.
    Python integers are unbounded, so the planning arithmetic itself cannot wrap.
    """
    budget_bytes = _gpu_memory_budget_bytes(free_bytes)

    fixed_bytes = (
        n * m * np.dtype(np.int32).itemsize
        + n * k_bins * np.dtype(np.float64).itemsize
        + 2 * n * n * np.dtype(np.float64).itemsize
    )
    bytes_per_chunk_column = (
        k_bins * k_bins * n * np.dtype(np.int32).itemsize
        + k_bins * n * np.dtype(np.float64).itemsize
    )

    if fixed_bytes + bytes_per_chunk_column > budget_bytes:
        raise RuntimeError(
            "compute_puc_full_cuda: the fixed GPU buffers plus a one-gene chunk "
            f"would require {(fixed_bytes + bytes_per_chunk_column) / 2**30:.2f} GiB, "
            "which exceeds the configured 65% memory budget "
            f"({budget_bytes / 2**30:.2f} GiB of {free_bytes / 2**30:.2f} GiB currently reusable). "
            "Reduce the number of genes/samples/bins, use a fixed small-bin discretizer, "
            "or use config.backend = 'cpu'."
        )

    safe_max_chunk = min((budget_bytes - fixed_bytes) // bytes_per_chunk_column, _MAX_CHUNK_SIZE, n)
    if requested_chunk_size is None:
        chunk_size = int(safe_max_chunk)
    else:
        if requested_chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if requested_chunk_size > n:
            raise ValueError(f"chunk_size={requested_chunk_size} exceeds n={n}")
        if requested_chunk_size > safe_max_chunk:
            raise RuntimeError(
                f"compute_puc_full_cuda: requested chunk_size={requested_chunk_size} would exceed "
                f"the configured 65% GPU-memory budget; the largest safe chunk is {safe_max_chunk} "
                f"with {free_bytes / 2**30:.2f} GiB currently reusable."
            )
        chunk_size = requested_chunk_size

    return int(chunk_size), int(fixed_bytes), int(bytes_per_chunk_column), int(budget_bytes)


def compute_puc_full_cuda(
    nodes: list[Node], *, base: float = 2, verbose: bool = False, chunk_size: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """GPU implementation of :func:`fastpidc.puc.compute_puc_full`.

    The complete device footprint is planned before large allocations against
    65% of VRAM currently reusable by CuPy (driver-free memory plus unused pool
    blocks, bounded by any configured pool limit). ``chunk_size=None`` chooses
    the largest safe chunk up to 256; an explicit chunk must also fit the same budget.
    """
    del base
    if not cuda_available():
        raise RuntimeError(
            "CUDA backend requested but cupy is not installed or no functional "
            "GPU was detected. Install the 'cuda' extra and ensure a GPU is "
            "available, or use config.backend = 'cpu'."
        )
    if not nodes:
        raise ValueError("compute_puc_full_cuda requires at least one node")

    import cupy as cp

    module = _load_module()
    joint_counts_kernel = module.get_function("joint_counts_kernel")
    mi_si_kernel = module.get_function("mi_si_kernel")
    puc_accumulation_kernel = module.get_function("puc_accumulation_kernel")

    n = len(nodes)
    m = int(nodes[0].binned_values.size)
    if any(node.binned_values.size != m for node in nodes):
        raise ValueError("all nodes must contain the same number of discretized samples")
    k_bins = max(node.number_of_bins for node in nodes)
    _check_kernel_scalar_limits(n, m, k_bins)

    free_bytes = _available_gpu_memory_bytes(cp)
    chunk_size, fixed_bytes, _, budget_bytes = _puc_memory_plan(
        n,
        m,
        k_bins,
        free_bytes,
        requested_chunk_size=chunk_size,
    )

    data_host = np.zeros((m, n), dtype=np.int32)
    marginals_host = np.zeros((k_bins, n), dtype=np.float64)
    for i, node in enumerate(nodes):
        if node.number_of_bins < 1:
            raise ValueError(f"node {node.label!r} has non-positive number_of_bins")
        if node.binned_values.size and (
            node.binned_values.min() < 0 or node.binned_values.max() >= node.number_of_bins
        ):
            raise ValueError(f"node {node.label!r} contains a bin id outside [0, number_of_bins)")
        if node.probabilities.size > k_bins:
            raise ValueError(f"node {node.label!r} has more probabilities than k_bins")

        data_host[:, i] = node.binned_values
        marginals_host[: node.number_of_bins, i] = node.probabilities

    data_gpu = marginals_gpu = None
    puc_scores_gpu = mi_matrix_gpu = None
    counts_chunk_gpu = si_chunk_gpu = None

    try:
        data_gpu = cp.asarray(data_host)
        marginals_gpu = cp.asarray(marginals_host)
        puc_scores_gpu = cp.zeros((n, n), dtype=cp.float64)
        mi_matrix_gpu = cp.zeros((n, n), dtype=cp.float64)
        counts_chunk_gpu = cp.zeros((k_bins, k_bins, n, chunk_size), dtype=cp.int32)
        si_chunk_gpu = cp.zeros((k_bins, n, chunk_size), dtype=cp.float64)

        threads = (16, 16)

        if verbose:
            n_chunks = -(-n // chunk_size)
            print(f"[fastpidc] GPU chunked PUC: processing {n} x {n} pairs (k_bins={k_bins})...")
            print(
                f"[fastpidc] GPU memory: {free_bytes / 2**30:.2f} GiB reusable; "
                f"65% budget={budget_bytes / 2**30:.2f} GiB; "
                f"fixed={fixed_bytes / 2**30:.2f} GiB"
            )
            print(f"[fastpidc] Using chunk size {chunk_size} ({n_chunks} iterations)")

        for z_start in range(0, n, chunk_size):
            z_chunk_size = min(chunk_size, n - z_start)

            counts_chunk_gpu.fill(0)
            si_chunk_gpu.fill(0.0)

            blocks = (-(-n // threads[0]), -(-z_chunk_size // threads[1]))

            joint_counts_kernel(
                blocks,
                threads,
                (
                    data_gpu,
                    counts_chunk_gpu,
                    np.int32(n),
                    np.int32(m),
                    np.int32(k_bins),
                    np.int32(z_start),
                    np.int32(z_chunk_size),
                ),
            )
            mi_si_kernel(
                blocks,
                threads,
                (
                    counts_chunk_gpu,
                    marginals_gpu,
                    mi_matrix_gpu,
                    si_chunk_gpu,
                    np.int32(n),
                    np.int32(m),
                    np.int32(k_bins),
                    np.int32(z_start),
                    np.int32(z_chunk_size),
                ),
            )
            puc_accumulation_kernel(
                blocks,
                threads,
                (
                    si_chunk_gpu,
                    mi_matrix_gpu,
                    puc_scores_gpu,
                    marginals_gpu,
                    np.int32(n),
                    np.int32(k_bins),
                    np.int32(z_start),
                    np.int32(z_chunk_size),
                ),
            )

        puc_scores = cp.asnumpy(puc_scores_gpu)
        mi_matrix = cp.asnumpy(mi_matrix_gpu)
        puc_scores = puc_scores + puc_scores.T
        return mi_matrix, puc_scores
    finally:
        # Drop references immediately even on a failed launch/allocation. CuPy's
        # pool may cache the released blocks, but they are no longer live and can
        # be reused or returned under memory pressure.
        del counts_chunk_gpu, si_chunk_gpu
        del puc_scores_gpu, mi_matrix_gpu, marginals_gpu, data_gpu


# --- Bayesian blocks --------------------------------------------------------
#
# Mirrors ``FastPIDCCUDAExt``'s host code: genes are grouped into U_g quantile
# buckets (so each launch picks a thread count suited to its block sizes), each
# bucket is split into batches that fit a memory budget, and each batch is
# flattened into the packed buffers the shared kernel indexes with per-gene
# offsets. Back-pointers come back 0-based and are backtracked on the host.

_BB_VALID_THREAD_COUNTS = (32, 64, 128, 256)
_BB_TYPE_SUFFIX = {
    np.dtype(np.uint8): "u8",
    np.dtype(np.uint16): "u16",
    np.dtype(np.uint32): "u32",
    np.dtype(np.uint64): "u64",
}


def _smallest_unsigned_dtype(max_value: int) -> np.dtype:
    """Narrowest unsigned dtype that represents ``max_value`` exactly."""
    if max_value < 0:
        raise ValueError("max_value must be nonnegative")
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise ValueError(f"max_value {max_value} exceeds uint64")


def _bb_kernel_name(count_dtype: np.dtype, index_dtype: np.dtype) -> str:
    """Entry point in the shared kernel source for these element types.

    CUDA C has no generics, so ``pidc_kernels.cu`` macro-generates one kernel
    per valid (prefix-count, back-pointer) type pair and the host selects by
    name. FastPIDC.jl's ``_bb_kernel_name`` resolves the same names.
    """
    for dtype in (count_dtype, index_dtype):
        if dtype not in _BB_TYPE_SUFFIX:
            raise ValueError(f"Bayesian-block buffers must be an unsigned type; got {dtype}")
    if index_dtype.itemsize > count_dtype.itemsize:
        raise ValueError(
            f"back-pointer type {index_dtype} is wider than the prefix-count type "
            f"{count_dtype}, which the shared kernel does not instantiate "
            "(U_g never exceeds the observation count)"
        )
    return f"bayesian_blocks_dp_{_BB_TYPE_SUFFIX[count_dtype]}_{_BB_TYPE_SUFFIX[index_dtype]}"


def _bb_threads_for_max_u(max_u: int) -> int:
    """Threads per block for a bucket whose largest problem has ``max_u``
    unique values."""
    if max_u <= 32:
        return 32
    if max_u <= 512:
        return 64
    if max_u <= 4096:
        return 128
    return 256


def _bb_prior_values(max_u: int) -> np.ndarray:
    """Prior (eq. 21, Scargle 2012) per endpoint.

    It depends only on the endpoint, not on the gene, so it is computed once on
    the host with the CPU solver's expression - which also removes it as a
    source of CPU/GPU numerical variation.
    """
    endpoints = np.arange(1, max_u + 1, dtype=np.float64)
    return 4 - np.log(73.53 * 0.05 * (endpoints**-0.478))


def _bb_quantile_buckets(problems: list[BayesianBlocksProblem]) -> list[list[int]]:
    """Group gene indices into up to four buckets of similar U_g.

    U_g is already known from preparation, so this is an O(G log G) sort of
    indices rather than another pass over the expression data.
    """
    n = len(problems)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: problems[i].prefix_counts.size)
    n_buckets = min(4, n)
    buckets = []
    for bucket in range(1, n_buckets + 1):
        lo = (bucket - 1) * n // n_buckets
        hi = bucket * n // n_buckets
        if lo < hi:
            buckets.append(order[lo:hi])
    return buckets


def _bb_problem_bytes(problem: BayesianBlocksProblem, count_dtype: np.dtype, index_dtype: np.dtype) -> int:
    """Device bytes one prepared problem contributes to a batch."""
    u = problem.prefix_counts.size
    return (
        8 * (u + 1)  # block lengths
        + count_dtype.itemsize * u  # prefix counts
        + 8 * u  # best scores
        + index_dtype.itemsize * u  # back-pointers
        + 8  # state offset
        + 8  # block offset
        + 4  # unique count
        + 8  # final score
    )


def _bb_memory_batches(
    bucket: list[int],
    problems: list[BayesianBlocksProblem],
    budget_bytes: int,
    count_dtype: np.dtype,
    index_dtype: np.dtype,
) -> list[list[int]]:
    """Split a bucket into batches whose packed buffers fit ``budget_bytes``."""
    batches: list[list[int]] = []
    current: list[int] = []
    current_bytes = 0

    for problem_index in bucket:
        problem_bytes = _bb_problem_bytes(problems[problem_index], count_dtype, index_dtype)
        if problem_bytes > budget_bytes:
            raise RuntimeError(
                f"One Bayesian-block problem requires {problem_bytes} bytes, which exceeds "
                f"the CUDA batch budget of {budget_bytes} bytes. Reduce the number of unique "
                "input values for that gene, or use bb_backend='cpu'."
            )
        if current and current_bytes + problem_bytes > budget_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(problem_index)
        current_bytes += problem_bytes

    if current:
        batches.append(current)
    return batches


def _flatten_bb_batch(
    problems: list[BayesianBlocksProblem], problem_indices: list[int], count_dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack a batch of problems into the flat buffers the shared kernel reads,
    with 0-based per-gene offsets."""
    unique_count_values = [int(problems[i].prefix_counts.size) for i in problem_indices]
    if any(u > _KERNEL_INT_MAX for u in unique_count_values):
        raise ValueError(
            f"Bayesian-blocks CUDA backend supports at most {_KERNEL_INT_MAX} unique values per gene"
        )
    unique_counts = np.asarray(unique_count_values, dtype=np.int32)

    state_offsets = np.zeros(len(problem_indices), dtype=np.int64)
    block_offsets = np.zeros(len(problem_indices), dtype=np.int64)
    state_offsets[1:] = np.cumsum(unique_counts, dtype=np.int64)[:-1]
    block_offsets[1:] = np.cumsum(unique_counts.astype(np.int64) + 1)[:-1]

    total_states = int(unique_counts.sum())
    prefix_counts = np.empty(total_states, dtype=count_dtype)
    block_lengths = np.empty(total_states + len(problem_indices), dtype=np.float64)

    for local_gene, problem_index in enumerate(problem_indices):
        problem = problems[problem_index]
        u = problem.prefix_counts.size
        state_start = int(state_offsets[local_gene])
        block_start = int(block_offsets[local_gene])
        prefix_counts[state_start : state_start + u] = problem.prefix_counts
        block_lengths[block_start : block_start + u + 1] = problem.edges[-1] - problem.edges

    return prefix_counts, block_lengths, state_offsets, block_offsets, unique_counts


def _change_points_from_last(last_values: np.ndarray, offset: int, n: int) -> np.ndarray:
    """Backtrack one gene's 0-based device back-pointers into an edge path.

    A valid partition may place every unique value in its own block, so the
    path holds up to ``n + 1`` entries.
    """
    if n < 1:
        raise ValueError("Bayesian-block backtracking requires n >= 1")

    change_points = np.empty(n + 1, dtype=np.int64)
    cursor = n + 1
    ind = n
    while True:
        cursor -= 1
        change_points[cursor] = ind
        if ind == 0:
            break
        next_ind = int(last_values[offset + ind - 1])
        if not 0 <= next_ind < ind:
            raise RuntimeError(f"invalid Bayesian-block back-pointer {next_ind} for state {ind}")
        ind = next_ind

    return change_points[cursor:]


def _solve_bb_batch(
    problems: list[BayesianBlocksProblem],
    problem_indices: list[int],
    threads: int,
    count_dtype: np.dtype,
    index_dtype: np.dtype,
    priors_gpu,
) -> list[BayesianBlocksSolution]:
    import cupy as cp

    if threads not in _BB_VALID_THREAD_COUNTS:
        raise ValueError(f"CUDA Bayesian blocks requires a thread count from {_BB_VALID_THREAD_COUNTS}; got {threads}")
    if len(problem_indices) > _KERNEL_INT_MAX:
        raise ValueError(
            f"CUDA Bayesian blocks batch has {len(problem_indices)} genes, "
            "which exceeds the signed 32-bit block-index limit."
        )

    prefix_counts, block_lengths, state_offsets, block_offsets, unique_counts = _flatten_bb_batch(
        problems, problem_indices, count_dtype
    )

    prefix_gpu = cp.asarray(prefix_counts)
    block_gpu = cp.asarray(block_lengths)
    state_offsets_gpu = cp.asarray(state_offsets)
    block_offsets_gpu = cp.asarray(block_offsets)
    unique_counts_gpu = cp.asarray(unique_counts)
    best_gpu = cp.zeros(prefix_counts.size, dtype=cp.float64)
    last_gpu = cp.zeros(prefix_counts.size, dtype=index_dtype)
    final_scores_gpu = cp.zeros(len(problem_indices), dtype=cp.float64)

    kernel = _load_module().get_function(_bb_kernel_name(count_dtype, index_dtype))
    kernel(
        (len(problem_indices),),
        (threads,),
        (
            prefix_gpu,
            block_gpu,
            state_offsets_gpu,
            block_offsets_gpu,
            unique_counts_gpu,
            best_gpu,
            last_gpu,
            final_scores_gpu,
            priors_gpu,
        ),
    )

    last_values = cp.asnumpy(last_gpu)
    final_scores = cp.asnumpy(final_scores_gpu)
    # Return this batch's device buffers to cupy's pool now: a run may process
    # many U_g buckets, and holding them until the next collection would eat
    # into the budget the next batch was sized against.
    del prefix_gpu, block_gpu, state_offsets_gpu, block_offsets_gpu
    del unique_counts_gpu, best_gpu, last_gpu, final_scores_gpu

    return [
        BayesianBlocksSolution(
            _change_points_from_last(last_values, int(state_offsets[local_gene]), int(unique_counts[local_gene])),
            float(final_scores[local_gene]),
        )
        for local_gene in range(len(problem_indices))
    ]


def solve_bayesian_blocks_cuda(
    problems: list[BayesianBlocksProblem], *, verbose: bool = False
) -> list[BayesianBlocksSolution]:
    """Solve prepared Bayesian-block problems on the GPU.

    Drives the shared ``bayesian_blocks_dp_*`` kernels, which use the same
    deterministic first-maximum reduction as
    :func:`fastpidc.discretizers.solve_bayesian_blocks_cpu`, so the selected
    change points agree exactly with the CPU reference.

    Each workload bucket is sized from 65% of the device memory reusable by
    CuPy at that point in the run. The per-problem byte estimate includes every
    packed device buffer and metadata array.
    """
    if not cuda_available():
        raise RuntimeError(
            "CUDA Bayesian blocks requested but cupy is not installed or no functional "
            "GPU was detected. Install the 'cuda' extra, or use bb_backend='cpu'."
        )
    if not problems:
        return []

    import cupy as cp

    # Compile/load before measuring reusable memory so module residency is
    # already reflected in the driver/pool accounting.
    _load_module()

    sample_count = max(int(round(float(p.prefix_counts[-1]))) for p in problems)
    max_u = max(int(p.prefix_counts.size) for p in problems)
    if max_u > _KERNEL_INT_MAX:
        raise ValueError(
            f"Bayesian-blocks CUDA backend supports at most {_KERNEL_INT_MAX} unique values per gene"
        )

    # A cumulative prefix count reaches the observation count, and back-pointers
    # only need to index candidates up to U_g, so pick the narrowest exact type.
    count_dtype = _smallest_unsigned_dtype(sample_count)
    index_dtype = _smallest_unsigned_dtype(max_u)

    if verbose:
        unique_counts = sorted(p.prefix_counts.size for p in problems)
        median_u = unique_counts[(len(unique_counts) - 1) // 2]
        print(
            f"[fastpidc] CUDA Bayesian blocks: {len(problems)} genes, "
            f"U_g median={median_u}, max={unique_counts[-1]}, "
            f"prefix counts={count_dtype}, back-pointers={index_dtype}"
        )

    priors_gpu = cp.asarray(_bb_prior_values(max_u))
    solutions: list[BayesianBlocksSolution | None] = [None] * len(problems)

    try:
        for bucket_number, bucket in enumerate(_bb_quantile_buckets(problems), start=1):
            bucket_max_u = max(problems[i].prefix_counts.size for i in bucket)
            threads = _bb_threads_for_max_u(bucket_max_u)

            # priors_gpu is already live. Re-evaluate memory available to CuPy
            # for every bucket, including reusable free blocks in its memory
            # pool and respecting any configured pool limit.
            free_bytes = _available_gpu_memory_bytes(cp)
            budget_bytes = _gpu_memory_budget_bytes(free_bytes)
            batches = _bb_memory_batches(bucket, problems, budget_bytes, count_dtype, index_dtype)

            if verbose:
                bucket_min_u = min(problems[i].prefix_counts.size for i in bucket)
                print(
                    f"[fastpidc] CUDA BB bucket {bucket_number}: {len(bucket)} genes, "
                    f"U_g={bucket_min_u}:{bucket_max_u}, threads={threads}, batches={len(batches)}, "
                    f"reusable={free_bytes / 2**30:.2f} GiB, 65% budget={budget_bytes / 2**30:.2f} GiB"
                )

            for batch in batches:
                for problem_index, solution in zip(
                    batch, _solve_bb_batch(problems, batch, threads, count_dtype, index_dtype, priors_gpu)
                ):
                    solutions[problem_index] = solution

        return solutions  # type: ignore[return-value]
    finally:
        del priors_gpu
