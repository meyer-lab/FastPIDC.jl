"""GPU backend: optionality, chunk sizing, and CPU/GPU agreement.

The GPU tests skip themselves when ``cupy`` or a functional device is
missing; the rest run everywhere, since keeping CUDA strictly optional is
itself part of the package's contract.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from fastpidc.cuda import (
    _KERNEL_INT_MAX,
    _MAX_CHUNK_SIZE,
    _bb_kernel_name,
    _bb_memory_batches,
    _bb_problem_bytes,
    _bb_quantile_buckets,
    _bb_threads_for_max_u,
    _check_kernel_scalar_limits,
    _gpu_memory_budget_bytes,
    _puc_memory_plan,
    _reusable_gpu_memory_bytes,
    _smallest_unsigned_dtype,
    cuda_available,
)
from fastpidc.discretizers import prepare_bayesian_blocks, solve_bayesian_blocks_cpu
from fastpidc.io import get_nodes
from fastpidc.puc import compute_puc_full
from fastpidc.types import Node, PIDCConfig

GIB = 2**30


def _make_nodes(n_nodes: int = 8, n_samples: int = 200, n_bins: int = 4, seed: int = 0) -> list[Node]:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, n_bins, size=n_samples)
    nodes = []
    for i in range(n_nodes):
        noise = rng.integers(0, n_bins, size=n_samples)
        mix = np.where(rng.random(n_samples) < 0.8 - 0.08 * i, base, noise) % n_bins
        nodes.append(
            Node.from_raw_values(f"N{i}", mix.astype(np.float64), "uniform_width", "maximum_likelihood", n_bins)
        )
    return nodes


def test_import_does_not_require_cupy():
    # A CPU-only install must be able to import the package (and everything it
    # re-exports) without cupy being present or even imported.
    script = "import sys; import fastpidc; assert 'cupy' not in sys.modules, sorted(sys.modules)"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_cuda_available_returns_a_bool():
    assert isinstance(cuda_available(), bool)


def test_gpu_memory_budget_is_65_percent_of_currently_free_memory():
    assert _gpu_memory_budget_bytes(1000) == 650


def test_reusable_gpu_memory_includes_cached_pool_blocks():
    assert _reusable_gpu_memory_bytes(1000, 400, pool_used_bytes=250) == 1400


def test_reusable_gpu_memory_honors_a_cupy_pool_limit():
    # The pool can reuse its free blocks and grow only until the configured
    # limit. Here the physical device would allow 1400 bytes, but the pool has
    # only 650 bytes of headroom from its current live usage.
    assert (
        _reusable_gpu_memory_bytes(1000, 400, pool_used_bytes=250, pool_limit_bytes=900)
        == 650
    )


def test_puc_memory_plan_counts_fixed_and_chunk_buffers_exactly():
    n, m, k_bins = 3, 5, 7
    _, fixed_bytes, bytes_per_chunk_column, _ = _puc_memory_plan(
        n=n, m=m, k_bins=k_bins, free_bytes=64 * GIB
    )
    assert fixed_bytes == n * m * 4 + n * k_bins * 8 + 2 * n * n * 8
    assert bytes_per_chunk_column == k_bins * k_bins * n * 4 + k_bins * n * 8


def test_chunk_size_is_capped_by_the_maximum():
    chunk, _, _, _ = _puc_memory_plan(n=1000, m=1000, k_bins=4, free_bytes=64 * GIB)
    assert chunk == _MAX_CHUNK_SIZE


def test_chunk_size_never_exceeds_the_number_of_genes():
    chunk, _, _, _ = _puc_memory_plan(n=10, m=1000, k_bins=4, free_bytes=64 * GIB)
    assert chunk == 10


def test_chunk_size_shrinks_when_memory_is_tight():
    tight, _, _, _ = _puc_memory_plan(n=5000, m=1000, k_bins=64, free_bytes=8 * GIB)
    roomy, _, _, _ = _puc_memory_plan(n=5000, m=1000, k_bins=64, free_bytes=64 * GIB)
    assert 1 <= tight < roomy <= _MAX_CHUNK_SIZE


def test_memory_plan_raises_before_allocating_when_one_gene_chunk_does_not_fit():
    with pytest.raises(RuntimeError, match="one-gene chunk"):
        _puc_memory_plan(n=20000, m=1000, k_bins=4000, free_bytes=8 * GIB)


def test_explicit_chunk_size_must_fit_the_same_memory_budget():
    safe, _, _, _ = _puc_memory_plan(n=5000, m=1000, k_bins=64, free_bytes=8 * GIB)
    with pytest.raises(RuntimeError, match="requested chunk_size"):
        _puc_memory_plan(
            n=5000,
            m=1000,
            k_bins=64,
            free_bytes=8 * GIB,
            requested_chunk_size=safe + 1,
        )


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_matches_cpu_backend():
    nodes = _make_nodes()
    cpu_mi, cpu_puc = compute_puc_full(nodes, config=PIDCConfig(backend="cpu"))
    gpu_mi, gpu_puc = compute_puc_full(nodes, config=PIDCConfig(backend="cuda"))

    np.testing.assert_allclose(gpu_mi, cpu_mi, atol=1e-12)
    np.testing.assert_allclose(gpu_puc, cpu_puc, atol=1e-9)


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_result_is_independent_of_chunk_size():
    from fastpidc.cuda import compute_puc_full_cuda

    nodes = _make_nodes(n_nodes=12)
    auto_mi, auto_puc = compute_puc_full_cuda(nodes)
    chunked_mi, chunked_puc = compute_puc_full_cuda(nodes, chunk_size=3)

    np.testing.assert_array_equal(chunked_mi, auto_mi)
    np.testing.assert_array_equal(chunked_puc, auto_puc)


# --- Bayesian blocks: kernel selection ------------------------------------


@pytest.mark.parametrize(
    "max_value,expected",
    [(0, np.uint8), (255, np.uint8), (256, np.uint16), (65535, np.uint16), (65536, np.uint32)],
)
def test_smallest_unsigned_dtype(max_value, expected):
    assert _smallest_unsigned_dtype(max_value) == np.dtype(expected)


def test_smallest_unsigned_dtype_rejects_negative():
    with pytest.raises(ValueError):
        _smallest_unsigned_dtype(-1)


@pytest.mark.parametrize(
    "count,index,expected",
    [
        (np.uint8, np.uint8, "bayesian_blocks_dp_u8_u8"),
        (np.uint32, np.uint16, "bayesian_blocks_dp_u32_u16"),
        (np.uint64, np.uint64, "bayesian_blocks_dp_u64_u64"),
    ],
)
def test_bb_kernel_name_matches_the_shared_source(count, index, expected):
    # FastPIDC.jl's `_bb_kernel_name` must resolve these same names.
    assert _bb_kernel_name(np.dtype(count), np.dtype(index)) == expected


def test_bb_kernel_name_rejects_a_wider_back_pointer_type():
    # U_g never exceeds the observation count, so this pair is not instantiated.
    with pytest.raises(ValueError, match="wider"):
        _bb_kernel_name(np.dtype(np.uint8), np.dtype(np.uint16))


def test_bb_kernel_name_rejects_signed_types():
    with pytest.raises(ValueError, match="unsigned"):
        _bb_kernel_name(np.dtype(np.int32), np.dtype(np.uint8))


@pytest.mark.parametrize("max_u,expected", [(32, 32), (33, 64), (512, 64), (513, 128), (4096, 128), (4097, 256)])
def test_bb_threads_for_max_u(max_u, expected):
    assert _bb_threads_for_max_u(max_u) == expected


# --- Bayesian blocks: workload bucketing and the memory budget --------------


def _bb_problems(sizes=(3, 40, 7, 900, 120, 5)):
    rng = np.random.default_rng(0)
    return [prepare_bayesian_blocks(rng.normal(size=size)) for size in sizes]


def test_bb_quantile_buckets_cover_every_problem_exactly_once():
    problems = _bb_problems()
    buckets = _bb_quantile_buckets(problems)
    assert sorted(i for bucket in buckets for i in bucket) == list(range(len(problems)))
    assert 1 <= len(buckets) <= 4


def test_bb_quantile_buckets_of_empty_input():
    assert _bb_quantile_buckets([]) == []


def test_bb_problem_bytes_counts_every_device_buffer():
    problem = _bb_problems(sizes=(40,))[0]
    u = problem.prefix_counts.size
    expected = (
        8 * (u + 1) + 1 * u + 8 * u + 1 * u + 8 + 8 + 4 + 8
    )  # state arrays + per-gene metadata
    assert _bb_problem_bytes(problem, np.dtype(np.uint8), np.dtype(np.uint8)) == expected


def test_bb_memory_batches_keeps_one_batch_when_everything_fits():
    problems = _bb_problems()
    bucket = list(range(len(problems)))
    assert _bb_memory_batches(bucket, problems, 2**40, np.dtype(np.uint8), np.dtype(np.uint8)) == [bucket]


def test_bb_memory_batches_splits_to_stay_under_the_budget():
    # Equal-sized problems make the packing exact: a budget of two problems
    # must yield three batches of two, in order.
    problems = _bb_problems(sizes=(50,) * 6)
    bucket = list(range(len(problems)))
    count_dtype = index_dtype = np.dtype(np.uint16)
    per_problem = _bb_problem_bytes(problems[0], count_dtype, index_dtype)
    assert all(_bb_problem_bytes(p, count_dtype, index_dtype) == per_problem for p in problems)

    batches = _bb_memory_batches(bucket, problems, 2 * per_problem, count_dtype, index_dtype)

    assert batches == [[0, 1], [2, 3], [4, 5]]
    for batch in batches:
        assert sum(_bb_problem_bytes(problems[i], count_dtype, index_dtype) for i in batch) <= 2 * per_problem


def test_bb_memory_batches_raises_when_one_problem_cannot_fit():
    # Bounding device memory is the point of the budget, so a problem that
    # cannot fit at all must be an explicit error, not an allocator failure.
    problems = _bb_problems(sizes=(900,))
    with pytest.raises(RuntimeError, match="exceeds the CUDA batch budget"):
        _bb_memory_batches([0], problems, 1024, np.dtype(np.uint16), np.dtype(np.uint16))


# --- Bayesian blocks: GPU behavior -----------------------------------------


def _bb_gpu_cases() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    cases = [
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 8.0, 13.0]),
        np.concatenate((np.zeros(20), [5.0])),  # maximal partition
        np.concatenate((np.zeros(12), np.arange(0.5, 8.5, 0.5), np.full(12, 15.0))),
        np.array([i**2 / 101 for i in range(300)]),  # needs the uint16 kernel
        rng.normal(size=1000),
        np.round(rng.normal(size=800), 2),  # heavy repeats
    ]
    cases += [np.where(rng.random(400) < 0.3, 0.0, rng.normal(size=400)) for _ in range(8)]
    return cases


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_bayesian_blocks_matches_the_cpu_reference():
    from fastpidc.cuda import solve_bayesian_blocks_cuda

    problems = [prepare_bayesian_blocks(values) for values in _bb_gpu_cases()]
    cpu = [solve_bayesian_blocks_cpu(problem) for problem in problems]
    gpu = solve_bayesian_blocks_cuda(problems)

    for problem, cpu_solution, gpu_solution in zip(problems, cpu, gpu):
        # The selected partition - and so every bin edge - must match exactly.
        np.testing.assert_array_equal(gpu_solution.change_points, cpu_solution.change_points)
        np.testing.assert_array_equal(
            problem.edges[gpu_solution.change_points], problem.edges[cpu_solution.change_points]
        )
        # CUDA and host `log` may differ by a few ULP, so only the objective
        # value is compared with a tolerance (as FastPIDC.jl's tests do).
        assert gpu_solution.score == pytest.approx(cpu_solution.score, abs=1e-9, rel=1e-12)


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_bayesian_blocks_handles_a_constant_problem():
    from fastpidc.cuda import solve_bayesian_blocks_cuda

    problem = prepare_bayesian_blocks(np.full(32, 2.5))
    solution = solve_bayesian_blocks_cuda([problem])[0]
    np.testing.assert_array_equal(solution.change_points, [0, 1])
    assert solution.score == 0.0


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_bayesian_blocks_is_deterministic():
    from fastpidc.cuda import solve_bayesian_blocks_cuda

    problems = [prepare_bayesian_blocks(values) for values in _bb_gpu_cases()]
    first = solve_bayesian_blocks_cuda(problems)
    second = solve_bayesian_blocks_cuda(problems)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.change_points, b.change_points)
        assert a.score == b.score


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
def test_gpu_bayesian_blocks_of_no_problems():
    from fastpidc.cuda import solve_bayesian_blocks_cuda

    assert solve_bayesian_blocks_cuda([]) == []


@pytest.mark.skipif(not cuda_available(), reason="no functional GPU / cupy backend available")
@pytest.mark.parametrize("data_file_name", ["toy_small_200.txt", "toy_small_200.h5"])
def test_gpu_and_cpu_node_building_agree(julia_test_data, data_file_name):
    # End-to-end: the batched GPU path must produce nodes indistinguishable
    # from the per-node CPU path, including the probability vectors.
    path = str(julia_test_data / data_file_name)
    gpu_nodes = get_nodes(path, bb_backend="cuda")
    cpu_nodes = get_nodes(path, bb_backend="cpu")

    assert [n.label for n in gpu_nodes] == [n.label for n in cpu_nodes]
    for from_gpu, from_cpu in zip(gpu_nodes, cpu_nodes):
        assert from_gpu.number_of_bins == from_cpu.number_of_bins, from_gpu.label
        np.testing.assert_array_equal(from_gpu.binned_values, from_cpu.binned_values)
        np.testing.assert_array_equal(from_gpu.probabilities, from_cpu.probabilities)


def test_bayesian_blocks_falls_back_to_cpu_without_a_gpu(monkeypatch, julia_test_data):
    # Unlike the PUC backend, a missing GPU here warns and falls back, since
    # both solvers select the same bin edges.
    import fastpidc.cuda

    monkeypatch.setattr(fastpidc.cuda, "cuda_available", lambda: False)
    path = str(julia_test_data / "yeast1_10_data.txt")

    with pytest.warns(RuntimeWarning, match="Falling back to the CPU reference solver"):
        fallback_nodes = get_nodes(path, bb_backend="cuda")

    cpu_nodes = get_nodes(path, bb_backend="cpu")
    for from_fallback, from_cpu in zip(fallback_nodes, cpu_nodes):
        np.testing.assert_array_equal(from_fallback.binned_values, from_cpu.binned_values)


# --- Flat index range: the >2^31 regression ---------------------------------
#
# A 12,071-gene x 38,176-cell production run crashed with
# CUDA_ERROR_ILLEGAL_ADDRESS because the shared kernels computed the `counts`
# index in 32-bit: k_bins^2 * n * chunk_size = 33^2 * 12071 * 256 is
# 3,365,201,664 elements, whose top index wraps past INT32_MAX. All flat offsets
# are 64-bit now; these tests pin that.

INT32_MAX = 2**31 - 1

# n=64 genes at 725 bins with a 64-gene chunk is the cheapest configuration that
# clears 2^31 counts elements: 8.02 GiB, versus ~22 MiB of everything else.
_OVERFLOW_N_NODES = 64
_OVERFLOW_N_SAMPLES = 2000
_OVERFLOW_N_BINS = 725
_OVERFLOW_CHUNK = 64
# A chunk small enough that the same problem stays provably inside int32,
# giving a control the overflow cannot have touched.
_CONTROL_CHUNK = 32
# The overflowing chunk needs ~8 GiB plus room for cupy's pool and the other
# buffers; require real headroom so the test never competes for a full device.
_OVERFLOW_FREE_MEMORY_FLOOR = 14 * GIB


def _counts_elements(k_bins: int, n: int, chunk: int) -> int:
    """Element count of the per-chunk joint-count buffer the kernels index."""
    return k_bins**2 * n * chunk


def test_production_configuration_exceeds_int32_indexing():
    # Documents why the kernels must index in 64-bit, and pins the arithmetic
    # that the large-memory test below relies on. No GPU needed.
    assert _counts_elements(k_bins=33, n=12071, chunk=256) - 1 > INT32_MAX
    assert _counts_elements(_OVERFLOW_N_BINS, _OVERFLOW_N_NODES, _OVERFLOW_CHUNK) - 1 > INT32_MAX
    assert _counts_elements(_OVERFLOW_N_BINS, _OVERFLOW_N_NODES, _CONTROL_CHUNK) - 1 <= INT32_MAX


def test_kernel_scalar_limits_allow_bin_pair_products_past_int32():
    # k_bins itself fits int32; k_bins**2 deliberately does not. The shared
    # kernel widens u before multiplying by k_bins, so this is now valid
    # indexing arithmetic (memory availability is a separate concern).
    _check_kernel_scalar_limits(12_071, 38_176, 50_000)
    assert 50_000**2 > INT32_MAX


@pytest.mark.parametrize(
    ("n", "m", "k_bins"),
    [
        (_KERNEL_INT_MAX + 1, 1, 1),
        (1, _KERNEL_INT_MAX + 1, 1),
        (1, 1, _KERNEL_INT_MAX + 1),
    ],
)
def test_kernel_scalar_limits_reject_values_that_would_narrow(n, m, k_bins):
    with pytest.raises(ValueError, match="signed 32-bit scalar limit"):
        _check_kernel_scalar_limits(n, m, k_bins)


def test_chunk_sizing_is_not_capped_by_int32_element_count():
    # Memory, not int32 indexing, controls the chunk. With enough free VRAM the
    # production-sized shape still reaches the 256-gene cap even though the
    # joint-count buffer contains >2^31 elements.
    chunk, _, _, _ = _puc_memory_plan(
        n=12_071,
        m=38_176,
        k_bins=33,
        free_bytes=32 * GIB,
    )
    assert chunk == 256
    assert _counts_elements(33, 12_071, chunk) - 1 > INT32_MAX


@pytest.mark.largemem
def test_puc_indexing_past_int32_matches_a_smaller_chunk():
    """Drive the PUC kernels past 2^31 counts elements and require the result to
    match a chunking that stays inside int32.

    Same kernels, same nodes, two buffer layouts: only the flat offsets differ,
    so any 32-bit truncation shows up as a mismatch (before the fix it faulted
    outright with CUDA_ERROR_ILLEGAL_ADDRESS).
    """
    if not cuda_available():
        pytest.skip("no functional GPU / cupy backend available")

    import cupy as cp

    from fastpidc.cuda import _available_gpu_memory_bytes, compute_puc_full_cuda

    free_bytes = _available_gpu_memory_bytes(cp)
    if free_bytes < _OVERFLOW_FREE_MEMORY_FLOOR:
        pytest.skip(
            f"needs {_OVERFLOW_FREE_MEMORY_FLOOR / GIB:.0f} GiB free device memory, have {free_bytes / GIB:.1f} GiB"
        )

    rng = np.random.default_rng(0)
    values = rng.random((_OVERFLOW_N_SAMPLES, _OVERFLOW_N_NODES))
    nodes = [
        Node.from_raw_values(f"N{i}", values[:, i], "uniform_width", "maximum_likelihood", _OVERFLOW_N_BINS)
        for i in range(_OVERFLOW_N_NODES)
    ]
    k_bins = max(node.number_of_bins for node in nodes)

    # Fail loudly rather than pass vacuously if the configuration drifts below
    # the threshold this test exists to cross.
    assert _counts_elements(k_bins, len(nodes), _OVERFLOW_CHUNK) - 1 > INT32_MAX

    overflow_mi, overflow_puc = compute_puc_full_cuda(nodes, chunk_size=_OVERFLOW_CHUNK)
    control_mi, control_puc = compute_puc_full_cuda(nodes, chunk_size=_CONTROL_CHUNK)

    np.testing.assert_array_equal(overflow_mi, control_mi)
    np.testing.assert_array_equal(overflow_puc, control_puc)
    assert np.all(np.isfinite(overflow_mi))
    assert np.all(overflow_puc >= 0)
