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
# Largest bins-per-gene the shared kernels can index: joint_counts_kernel forms
# the bin-pair index u * k_bins + v in int32, so k_bins**2 must fit there
# (isqrt(2**31 - 1) == 46340). Every other flat offset is 64-bit.
_MAX_K_BINS = 46340
# Headroom for the fixed buffers, allocator overhead and fragmentation, matching
# the safety factor used by FastPIDC.jl's CUDA extension.
_CHUNK_MEMORY_SAFETY_FACTOR = 0.8
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


def _check_kernel_index_limits(k_bins: int) -> None:
    """Validate the one kernel limit that is not lifted by 64-bit indexing.

    The shared kernels compute every flat buffer offset in 64-bit, with a single
    exception: ``joint_counts_kernel`` forms the bin-pair index
    ``u * k_bins + v`` in ``int``, which is exact only while ``k_bins**2`` fits
    in int32. FastPIDC.jl's ``_check_kernel_index_limits`` enforces the same
    bound.
    """
    if k_bins > _MAX_K_BINS:
        raise RuntimeError(
            f"compute_puc_full_cuda: the discretizer selected k_bins={k_bins} bins per "
            f"gene, but the CUDA kernels index bin pairs in 32-bit and support at most "
            f'{_MAX_K_BINS}. Use discretizer="uniform_width" with a fixed, small '
            f"number_of_bins, or config.backend = 'cpu'."
        )


def _chunk_size_for_free_memory(n: int, k_bins: int, free_bytes: int) -> int:
    """Largest target-gene chunk whose intermediate buffers fit in
    ``free_bytes`` of device memory, capped at :data:`_MAX_CHUNK_SIZE`.

    The chunked intermediates scale with ``k_bins**2 * n`` (joint counts) and
    ``k_bins * n`` (specific information) per target gene, so an adaptive
    discretizer that picks many bins can make even one gene per chunk too
    large; that case raises instead of failing inside the allocator.

    This bounds *memory availability* only - it is deliberately not a bound on
    the flat element index. The kernels index these buffers in 64-bit precisely
    so the chunk can be sized from free memory without an int32 element cap.
    """
    bytes_per_chunk_column = k_bins**2 * n * np.dtype(np.int32).itemsize + k_bins * n * np.dtype(np.float64).itemsize
    usable_bytes = free_bytes * _CHUNK_MEMORY_SAFETY_FACTOR
    max_chunk_size = int(usable_bytes // bytes_per_chunk_column)
    if max_chunk_size < 1:
        raise RuntimeError(
            f"compute_puc_full_cuda: even a single-gene chunk would require "
            f"{bytes_per_chunk_column / 2**30:.2f} GiB of GPU memory (only "
            f"{usable_bytes / 2**30:.2f} GiB usable), because the discretizer selected "
            f"k_bins={k_bins} bins per gene. This is usually caused by an adaptive "
            f'discretizer (e.g. "bayesian_blocks", the default) picking an unbounded '
            f'number of bins on a dataset with many samples. Try discretizer="uniform_width" '
            f"with a fixed, small number_of_bins (e.g. 10-20), or config.backend = 'cpu'."
        )
    return min(max_chunk_size, _MAX_CHUNK_SIZE, n)


def compute_puc_full_cuda(
    nodes: list[Node], *, base: float = 2, verbose: bool = False, chunk_size: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """GPU implementation of :func:`fastpidc.puc.compute_puc_full`.

    Parameters
    ----------
    base : accepted for signature parity with the CPU path but unused: mutual
        information is always computed in base 2 on the GPU, matching
        FastPIDC.jl's CUDA extension.
    chunk_size : number of target genes processed per pass. ``None`` (the
        default) sizes it from the currently free device memory, as
        FastPIDC.jl does.
    """
    del base
    if not cuda_available():
        raise RuntimeError(
            "CUDA backend requested but cupy is not installed or no functional "
            "GPU was detected. Install the 'cuda' extra and ensure a GPU is "
            "available, or use config.backend = 'cpu'."
        )

    import cupy as cp

    module = _load_module()
    joint_counts_kernel = module.get_function("joint_counts_kernel")
    mi_si_kernel = module.get_function("mi_si_kernel")
    puc_accumulation_kernel = module.get_function("puc_accumulation_kernel")

    n = len(nodes)
    m = nodes[0].binned_values.size
    k_bins = max(node.number_of_bins for node in nodes)
    # Checked here rather than in _chunk_size_for_free_memory, which is skipped
    # entirely when the caller passes an explicit chunk_size.
    _check_kernel_index_limits(k_bins)

    if chunk_size is None:
        chunk_size = _chunk_size_for_free_memory(n, k_bins, int(cp.cuda.runtime.memGetInfo()[0]))

    data_host = np.zeros((m, n), dtype=np.int32)
    marginals_host = np.zeros((k_bins, n), dtype=np.float64)
    for i, node in enumerate(nodes):
        data_host[:, i] = node.binned_values
        marginals_host[: node.number_of_bins, i] = node.probabilities

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

    # Symmetrize: each ordered pair (x, z) only holds one of the two
    # directional contributions (see the module docstring in pidc_kernels.cu).
    puc_scores = puc_scores + puc_scores.T

    return mi_matrix, puc_scores


# --- Bayesian blocks --------------------------------------------------------
#
# Mirrors ``FastPIDCCUDAExt``'s host code: genes are grouped into U_g quantile
# buckets (so each launch picks a thread count suited to its block sizes), each
# bucket is split into batches that fit a memory budget, and each batch is
# flattened into the packed buffers the shared kernel indexes with per-gene
# offsets. Back-pointers come back 0-based and are backtracked on the host.

# Keep headroom for the CUDA context, allocator bookkeeping and other live
# allocations while still using most of the free device memory.
_BB_MEMORY_BUDGET_FRACTION = 0.65
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
    unique_counts = np.array([problems[i].prefix_counts.size for i in problem_indices], dtype=np.int32)
    if unique_counts.size and unique_counts.max() > np.iinfo(np.int32).max:
        raise ValueError("Bayesian-blocks CUDA backend supports at most 2**31-1 unique values per gene")

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

    Raises
    ------
    RuntimeError
        If no GPU backend is available, or if one gene's problem alone exceeds
        the device memory budget.
    """
    if not cuda_available():
        raise RuntimeError(
            "CUDA Bayesian blocks requested but cupy is not installed or no functional "
            "GPU was detected. Install the 'cuda' extra, or use bb_backend='cpu'."
        )
    if not problems:
        return []

    import cupy as cp

    # A cumulative prefix count reaches the observation count, and back-pointers
    # only need to index candidates up to U_g, so pick the narrowest exact type
    # for each rather than paying for 64-bit buffers on every dataset.
    sample_count = max(int(round(float(p.prefix_counts[-1]))) for p in problems)
    max_u = max(p.prefix_counts.size for p in problems)
    count_dtype = _smallest_unsigned_dtype(sample_count)
    index_dtype = _smallest_unsigned_dtype(max_u)

    free_bytes = int(cp.cuda.runtime.memGetInfo()[0])
    budget_bytes = max(1, int(_BB_MEMORY_BUDGET_FRACTION * free_bytes))

    if verbose:
        unique_counts = sorted(p.prefix_counts.size for p in problems)
        median_u = unique_counts[(len(unique_counts) - 1) // 2]
        print(
            f"[fastpidc] CUDA Bayesian blocks: {len(problems)} genes, "
            f"U_g median={median_u}, max={unique_counts[-1]}, "
            f"prefix counts={count_dtype}, back-pointers={index_dtype}"
        )
        print(f"[fastpidc] CUDA Bayesian blocks memory budget: {budget_bytes / 2**30:.2f} GiB")

    priors_gpu = cp.asarray(_bb_prior_values(max_u))
    solutions: list[BayesianBlocksSolution | None] = [None] * len(problems)

    for bucket_number, bucket in enumerate(_bb_quantile_buckets(problems), start=1):
        bucket_max_u = max(problems[i].prefix_counts.size for i in bucket)
        threads = _bb_threads_for_max_u(bucket_max_u)
        batches = _bb_memory_batches(bucket, problems, budget_bytes, count_dtype, index_dtype)

        if verbose:
            bucket_min_u = min(problems[i].prefix_counts.size for i in bucket)
            print(
                f"[fastpidc] CUDA BB bucket {bucket_number}: {len(bucket)} genes, "
                f"U_g={bucket_min_u}:{bucket_max_u}, threads={threads}, batches={len(batches)}"
            )

        for batch in batches:
            for problem_index, solution in zip(
                batch, _solve_bb_batch(problems, batch, threads, count_dtype, index_dtype, priors_gpu)
            ):
                solutions[problem_index] = solution

    return solutions  # type: ignore[return-value]
