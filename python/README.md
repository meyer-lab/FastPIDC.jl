# fastpidc

Python port of [FastPIDC.jl](https://github.com/meyer-lab/FastPIDC.jl):
information-theoretic network inference (MI, CLR, PUC, PIDC) from per-node
(e.g. per-gene) measurements, following Chan, Stumpf & Babtie (2017),
*Gene Regulatory Network Inference from Single-Cell Data Using Multivariate
Information Measures*, Cell Systems.

This package is a standalone, native Python/NumPy implementation - it does
not call out to Julia. See the repository root README for how this package
and the Julia package are organized and released together.

## Installation

This package is managed with [uv](https://docs.astral.sh/uv/). From this
directory:

```sh
uv sync                 # install (CPU-only)
uv sync --extra cuda    # also install the optional GPU (cupy) backend
```

## Basic usage

```python
from fastpidc import get_nodes, infer_network, PIDCNetworkInference, PIDCConfig

# One step: infer directly from a data file.
network = infer_network(
    "data.txt",
    PIDCNetworkInference(),
    config=PIDCConfig(backend="cpu"),  # or "cuda", if cupy + a GPU are available
)

# Multiple steps: build nodes, then infer.
from fastpidc.network import infer_network_from_nodes

nodes = get_nodes("data.txt")
network = infer_network_from_nodes(PIDCNetworkInference(), nodes)
```

The input data file's first line is treated as a header (discarded); each
subsequent line is `label value1 value2 ...`.

`network.edges` is sorted by descending weight (confidence). Use
`fastpidc.write_network_file`/`write_network_npy` to save it, and
`fastpidc.get_adjacency_matrix` to threshold it into an adjacency matrix.

## Perturbation-aware inference

When the data carries a per-cell perturbation label - which gene was targeted,
and in which direction - `fastpidc.perturbation` offers a stronger quantity
than observational dependence: the mutual information between the
*intervention* and a gene's expression.

```python
from fastpidc.io import get_nodes, read_h5ad_obs
from fastpidc.perturbation import PerturbationDesign, intervention_information

nodes = get_nodes("experiment.h5ad")
obs = read_h5ad_obs("experiment.h5ad", ["target_gene", "perturbation_type"])
design = PerturbationDesign.from_obs(obs, [n.label for n in nodes])
T = intervention_information(nodes, design)["T"]      # T[a, b] = I(P_a ; B)
```

Because cells are assigned to conditions by design, `T` is interventional
rather than associational, and it is asymmetric - so one matrix both ranks
edges and orients them, with no fitted parameters. It has rows only for genes
that were actually perturbed, so it supplements PIDC rather than replacing it.

This mode is additive: on control cells alone, or on data with no labels,
FastPIDC behaves exactly as before.

See [`docs/interventional.md`](docs/interventional.md) for the definition, the
estimator caveats, the positive control to run first, the benchmark results
(including where it loses), and the approaches that were tried and dropped.

## Command line

```sh
uv run fastpidc --infile data.txt --outfile edges.tsv --backend cpu
```

Run `uv run fastpidc --help` for the full option list (discretizer,
estimator, output format, diagnostic score dumps, etc.) - it mirrors
`CLI_fastpidc.jl` in the Julia package.

## Backends

`PIDCConfig` selects a backend for each of the two compute-heavy stages:

* `backend` - PUC/PIDC scoring. `"cuda"` (default) or `"cpu"`.
* `bb_backend` - the Bayesian-blocks dynamic program used to discretize each
  node. `"cuda"` (default) or `"cpu"`.

```python
from fastpidc import PIDCConfig

PIDCConfig(backend="cpu", bb_backend="cpu")  # no GPU needed
```

The two fail differently, mirroring FastPIDC.jl: `backend="cuda"` without a
usable GPU raises, because there is no automatic CPU equivalent to fall back
to, whereas `bb_backend="cuda"` warns and uses the CPU solver, since both
Bayesian-block solvers select the same bin edges and only throughput differs.

The CPU paths are direct NumPy ports of the reference algorithm and are the
numerical reference; PUC/PIDC is inherently `O(N^3)`, so the CPU backend suits
small-to-moderate networks (tens to low hundreds of nodes).

Both GPU paths bound their device memory from what is currently free rather
than assuming a fixed allocation: PUC sizes its target-gene chunk, and
Bayesian blocks groups nodes into workload buckets and splits each into
batches that fit a memory budget. If a single node or gene cannot fit even
alone, that is an explicit error naming the cause, not an allocator failure.

### One shared kernel source

Neither package reimplements the other's kernels. `pidc_kernels.cu` is the
single canonical CUDA C implementation of both the chunked PUC algorithm and
the Bayesian-blocks dynamic program, compiled and launched at runtime from
either language:

| | Python | Julia |
|---|---|---|
| compiler | `cupy.RawModule` (nvrtc) | `nvcc --ptx` |
| loader | cupy | `CUDA.CuModule` |
| launcher | `RawKernel.__call__` | `CUDA.cudacall` |

It lives at `src/fastpidc/kernels/pidc_kernels.cu` in this package, and
FastPIDC.jl's `ext/FastPIDCCUDAExt` resolves it relative to the repository
root. Edits to it affect both packages: build it with **both** compilers (nvrtc
is stricter - it has none of the host headers `nvcc` pulls in) and run both
test suites. `tests/test_bayesian_blocks.py` and Julia's
`test/cuda_bayesian_blocks_tests.jl` each pin the GPU results against their own
CPU reference, and against each other through it.

Because CUDA C has no generics, the Bayesian-block entry points are
macro-generated per (prefix-count, back-pointer) integer type pair, so narrow
datasets do not pay for 64-bit buffers. Both hosts resolve the same names
(`_bb_kernel_name` on either side).

## Testing

```sh
uv run pytest
```

When a working `julia` executable is on `PATH`, the tests marked `julia`
cross-check results against a freshly run `FastPIDC.jl` (bin edges, MI, PUC
and inferred edge weights); GPU tests likewise require `cupy` and a device.
Both groups skip themselves automatically when unavailable.

Across languages the selected Bayesian-block partition is compared exactly,
while objective scores use a tolerance: the per-endpoint prior involves
`K ** -0.478`, and Julia's and NumPy's `pow` differ by one ULP at a few
endpoints. That shifts a score by ~1e-15 without changing which partition
wins.

```sh
uv run pytest -m julia        # only the Julia cross-validation tests
uv run pytest -m "not julia"  # everything else
uv run ruff check .
```
