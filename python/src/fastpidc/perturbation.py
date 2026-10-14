"""Perturbation-aware network inference.

FastPIDC scores edges from observational dependence between gene expression
levels. When the data carries a per-cell perturbation label -- which gene was
targeted, and in which direction -- a stronger quantity is available: the
mutual information between the *intervention* and a gene's expression.
Because cells are assigned to conditions by experimental design, that
quantity is interventional rather than associational, and it is asymmetric,
so it both ranks and orients edges.

This module adds that mode. It does not change any existing code path: on
control cells alone, or on data with no labels, FastPIDC behaves exactly as
before.

Recommended entry point
-----------------------

:func:`intervention_information` -- ``T[a, b] = I(P_a; B)``, the total
interventional effect of perturbing gene ``a`` on gene ``b``, as an
asymmetric matrix. Rank a skeleton edge by ``max(T[a,b], T[b,a])`` and orient
it by which is larger.

.. code-block:: python

    from fastpidc.io import get_nodes, read_h5ad_obs
    from fastpidc.perturbation import PerturbationDesign, intervention_information

    nodes = get_nodes("experiment.h5ad")
    obs = read_h5ad_obs("experiment.h5ad", ["target_gene", "perturbation_type"])
    design = PerturbationDesign.from_obs(obs, [n.label for n in nodes])
    T = intervention_information(nodes, design)["T"]

On the 11-gene HSC benchmark (10 replicates) this scores 0.858 skeleton AP
against 0.691 for pooled PIDC and 0.620 for control-only PIDC, and orients
89% of unidirectional edges correctly. See ``scratch/docs/`` for the full
evaluation, including where it does *not* win.

The condition decomposition
---------------------------

:func:`mi_decomposition` and :func:`correlation_decomposition` split pairwise
dependence into the part surviving *within* a condition and the part carried
*between* conditions:

.. code-block:: text

    Cov(A,B)  =  E_P[ Cov(A,B | P) ]  +  Cov_P( E[A|P], E[B|P] )
       total   =        within        +            between

with the analogue ``I(A;B) = I(A;B|P) + II(A;B;P)``, where ``II`` is the
interaction information -- positive when the condition label accounts for
part of the A-B dependence. Pooling all cells scores ``total``; control cells
alone score a single stratum. Both are what FastPIDC does today.

Measured result: the ``within`` arm carries **no** edge information beyond a
shuffled-label null. Conditioning on perturbation identity removes the
interventional signal, which is why it does not help. The decomposition is
kept because it is what makes that statement checkable.

Two estimator facts that matter when reading any stratified number:

- ``I(A;B|P) = sum_p p(p) I(A;B|P=p)`` is **cell-count weighted** by
  definition, and only that weighting satisfies the identity above. Equal
  weighting per condition (D-SPIN's own choice) is computed alongside as a
  labelled alternative, never as the decomposition.
- Stratified plug-in MI is biased upward by roughly
  ``(B_A - 1)(B_B - 1) / (2 N ln2)`` bits per stratum, so k strata inflate
  the bias about k-fold. It scales with both genes' bin counts, which
  Bayesian blocks sets per gene, so it does **not** cancel in a ranking
  metric. Compare any stratified arm against a shuffled-label null.

What was tried and removed
--------------------------

Mediator correction -- stripping *indirect* paths from ``T``, since
``a -> c -> b`` scores as well as a direct edge -- was implemented two ways
and removed after both lost to raw ``T`` at detection on a 36% dense network
*and* on one 5.6x sparser:

- Williams-Beer ``I_min`` redundancy degenerates when the source is an
  intervention label. A neighbouring gene is far more informative about the
  target, so redundancy equals the intervention's whole mutual information
  and unique information is zero *by construction*, path or no path.
- Conditional-independence screening avoids that and still lost.

See ``scratch/docs/stage2_results.md`` and ``modular_results.md``. The
scripts that produced those numbers import the removed functions and are
kept only as a record of the runs, not as something re-runnable.

"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .information import get_frequencies_from_bin_ids, get_probabilities
from .network import get_mi_and_si
from .types import Node

__all__ = [
    "ACTIVATION",
    "INHIBITION",
    "PerturbationDesign",
    "check_covariance_identity",
    "compose_directed",
    "correlation_decomposition",
    "dose_intervention_node",
    "intervention_information",
    "intervention_node",
    "mi_decomposition",
    "subset_node",
]

ACTIVATION = "+"
INHIBITION = "-"

_DEFAULT_ACTIVATION_VALUES = ("activation", "activated", "overexpression", "crispra", "a")
_DEFAULT_INHIBITION_VALUES = ("knockdown", "knockout", "inhibition", "inhibited", "crispri", "i", "kd", "ko")
_DEFAULT_CONTROL_VALUES = ("control", "ctrl", "none", "nontargeting", "non-targeting", "na", "")


def subset_node(node: Node, mask: np.ndarray, estimator: str = "maximum_likelihood") -> Node:
    """Restrict ``node`` to the cells selected by ``mask``, keeping its bins.

    Bin *ids* are carried over unchanged and only the marginal is
    re-estimated on the subset. Re-discretizing per subset instead would
    give each subset its own bin edges and its own bin count, which makes
    values from different subsets incommensurable - so never do that; mask
    a globally discretized node instead.
    """
    binned = node.binned_values[mask]
    frequencies = get_frequencies_from_bin_ids(binned, node.number_of_bins)
    return Node(node.label, binned, node.number_of_bins, get_probabilities(estimator, frequencies))


@dataclass(frozen=True)
class PerturbationDesign:
    """Which gene was perturbed, in which direction, in every cell.

    Built from an ``.h5ad``'s ``obs`` (see
    :func:`fastpidc.io.read_h5ad_obs`), whose arrays are in the same cell
    order as the nodes :func:`fastpidc.io.get_nodes` returns - so every mask
    here indexes a node's ``binned_values`` directly.

    Attributes
    ----------
    target : per-cell perturbed gene label, or ``""`` in a control cell.
    direction : per-cell :data:`ACTIVATION`, :data:`INHIBITION`, or ``""``
        in a control cell.
    gene_labels : the node labels, in node order.
    """

    target: np.ndarray
    direction: np.ndarray
    gene_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.target.shape != self.direction.shape:
            raise ValueError("target and direction must describe the same cells")

    @property
    def n_cells(self) -> int:
        return int(self.target.size)

    @classmethod
    def from_obs(
        cls,
        obs: Mapping[str, np.ndarray],
        gene_labels: Sequence[str],
        *,
        target_column: str = "target_gene",
        direction_column: str = "perturbation_type",
        activation_values: Sequence[str] = _DEFAULT_ACTIVATION_VALUES,
        inhibition_values: Sequence[str] = _DEFAULT_INHIBITION_VALUES,
        control_values: Sequence[str] = _DEFAULT_CONTROL_VALUES,
    ) -> PerturbationDesign:
        """Build a design from ``obs`` columns.

        Column names and the vocabularies that mark activation, inhibition
        and control are all configurable, since no two perturbation datasets
        agree on them. Matching is case-insensitive. A cell whose target is
        not one of ``gene_labels`` is treated as a control cell for the
        purposes of this design - its gene is not in the matrix, so there is
        no intervention variable it could define - and that is reported.
        """
        for column in (target_column, direction_column):
            if column not in obs:
                raise ValueError(f"obs has no column {column!r}. Available: {sorted(obs)}")

        known = {str(label) for label in gene_labels}
        activation = {v.lower() for v in activation_values}
        inhibition = {v.lower() for v in inhibition_values}
        control = {v.lower() for v in control_values}

        raw_target = np.asarray(obs[target_column], dtype=object)
        raw_direction = np.asarray(obs[direction_column], dtype=object)

        target = np.empty(raw_target.shape, dtype=object)
        direction = np.empty(raw_target.shape, dtype=object)
        unknown_directions: set[str] = set()
        off_matrix: set[str] = set()

        for i, (t, d) in enumerate(zip(raw_target, raw_direction)):
            t_str = "" if t is None else str(t)
            d_str = ("" if d is None else str(d)).lower()
            if d_str in control or t_str.lower() in control:
                target[i] = ""
                direction[i] = ""
                continue
            if t_str not in known:
                off_matrix.add(t_str)
                target[i] = ""
                direction[i] = ""
                continue
            if d_str in activation:
                target[i], direction[i] = t_str, ACTIVATION
            elif d_str in inhibition:
                target[i], direction[i] = t_str, INHIBITION
            else:
                unknown_directions.add(d_str)
                target[i] = ""
                direction[i] = ""

        if unknown_directions:
            warnings.warn(
                f"Unrecognized {direction_column!r} values treated as control: {sorted(unknown_directions)}. "
                "Pass activation_values/inhibition_values/control_values to classify them.",
                RuntimeWarning,
                stacklevel=2,
            )
        if off_matrix:
            warnings.warn(
                f"{len(off_matrix)} perturbation target(s) are not among the genes in the matrix and were "
                f"treated as control: {sorted(off_matrix)[:5]}{'...' if len(off_matrix) > 5 else ''}",
                RuntimeWarning,
                stacklevel=2,
            )
        return cls(target=target, direction=direction, gene_labels=tuple(str(g) for g in gene_labels))

    # --- masks and coverage ---------------------------------------------

    def control_mask(self) -> np.ndarray:
        return self.target == ""

    def mask(self, gene: str, direction: str) -> np.ndarray:
        return (self.target == gene) & (self.direction == direction)

    def directions(self, gene: str) -> tuple[str, ...]:
        """Perturbation directions actually present for ``gene``, in a stable
        order. Empty when the gene was never perturbed."""
        return tuple(d for d in (ACTIVATION, INHIBITION) if np.any(self.mask(gene, d)))

    def coverage(self) -> dict[str, dict[str, int]]:
        """Cell count per ``(gene, direction)``, for every perturbed gene.

        This is the table every direction-dependent branch should read, so
        that a gene with one direction and a gene with two take the same code
        path and differ only in how many strata they contribute. The control
        count is included under the key ``""``.
        """
        table: dict[str, dict[str, int]] = {"": {"": int(self.control_mask().sum())}}
        for gene in self.gene_labels:
            present = {d: int(self.mask(gene, d).sum()) for d in self.directions(gene)}
            if present:
                table[gene] = present
        return table

    def perturbed_genes(self) -> tuple[str, ...]:
        return tuple(g for g in self.gene_labels if self.directions(g))

    def strata(self, min_cells: int = 1) -> list[tuple[str, np.ndarray]]:
        """Every condition as a ``(label, cell mask)`` pair: control first,
        then one per ``(gene, direction)`` that exists.

        Strata with fewer than ``min_cells`` cells are dropped with a
        warning rather than contributing a noisy estimate.
        """
        candidates: list[tuple[str, np.ndarray]] = [("control", self.control_mask())]
        for gene in self.gene_labels:
            for d in self.directions(gene):
                candidates.append((f"{gene}{d}", self.mask(gene, d)))

        kept, dropped = [], []
        for label, mask in candidates:
            (kept if mask.sum() >= min_cells else dropped).append((label, mask))
        if dropped:
            warnings.warn(
                f"Dropped {len(dropped)} stratum/strata with fewer than {min_cells} cells: "
                f"{[label for label, _ in dropped]}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not kept:
            raise ValueError("no strata left after the min_cells filter")
        return kept

    def summary(self) -> str:
        table = self.coverage()
        n_control = table.pop("", {"": 0})[""]
        both = [g for g, d in table.items() if len(d) == 2]
        one = [g for g, d in table.items() if len(d) == 1]
        never = [g for g in self.gene_labels if g not in table]

        def show(items: list, limit: int = 8) -> str:
            """Real datasets have thousands of unperturbed genes; printing
            them all buries the counts that matter."""
            head = ", ".join(str(i) for i in items[:limit])
            return f"[{head}{', ...' if len(items) > limit else ''}]" if items else "[]"

        lines = [
            f"{self.n_cells} cells, {len(self.gene_labels)} genes, "
            f"{len(table)} perturbed gene(s), {n_control} control cells",
            f"  both directions: {len(both)}  {show(both)}",
            f"  one direction:   {len(one)}  {show([(g, list(table[g])[0]) for g in one])}",
            f"  never perturbed: {len(never)}  {show(never)}",
        ]
        return "\n".join(lines)


# --- decompositions -------------------------------------------------------


def _weights(strata: Sequence[tuple[str, np.ndarray]], equal: bool) -> np.ndarray:
    counts = np.array([float(mask.sum()) for _, mask in strata])
    return np.full(counts.size, 1.0 / counts.size) if equal else counts / counts.sum()


def correlation_decomposition(
    values: np.ndarray,
    design: PerturbationDesign,
    *,
    min_cells: int = 1,
) -> dict[str, np.ndarray]:
    """Split pairwise correlation into total, within- and between-condition.

    ``values`` is a ``(cells, genes)`` matrix in the design's cell order.
    Returns correlation matrices ``total``, ``within``, ``between`` and
    ``within_equal``, plus the three covariance matrices they come from.

    The covariance identity ``total = within + between`` holds exactly for
    the cell-count-weighted arms (it is the law of total covariance), and
    :func:`check_covariance_identity` asserts it. ``within_equal`` weights
    each condition the same regardless of size and therefore does *not*
    satisfy the identity - it is reported as a labelled alternative.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != design.n_cells:
        raise ValueError(f"values has {values.shape[0]} rows but the design describes {design.n_cells} cells")

    strata = design.strata(min_cells=min_cells)
    grand_mean = values.mean(axis=0)

    within_cov = np.zeros((values.shape[1], values.shape[1]))
    within_cov_equal = np.zeros_like(within_cov)
    between_cov = np.zeros_like(within_cov)

    w_count = _weights(strata, equal=False)
    w_equal = _weights(strata, equal=True)

    for (_, mask), wc, we in zip(strata, w_count, w_equal):
        block = values[mask]
        centred = block - block.mean(axis=0)
        # Population (ddof=0) moments, so the identity is exact rather than
        # off by the Bessel correction.
        cov_p = centred.T @ centred / block.shape[0]
        within_cov += wc * cov_p
        within_cov_equal += we * cov_p
        shift = block.mean(axis=0) - grand_mean
        between_cov += wc * np.outer(shift, shift)

    centred_all = values - grand_mean
    total_cov = centred_all.T @ centred_all / values.shape[0]

    def to_corr(cov: np.ndarray) -> np.ndarray:
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.outer(sd, sd)
        return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "total": to_corr(total_cov),
        "within": to_corr(within_cov),
        "within_equal": to_corr(within_cov_equal),
        "between": to_corr(between_cov),
        "total_cov": total_cov,
        "within_cov": within_cov,
        "between_cov": between_cov,
    }


def check_covariance_identity(result: Mapping[str, np.ndarray], atol: float = 1e-8) -> float:
    """Assert the law of total covariance on a
    :func:`correlation_decomposition` result, returning the max abs residual."""
    residual = float(np.max(np.abs(result["total_cov"] - result["within_cov"] - result["between_cov"])))
    scale = float(np.max(np.abs(result["total_cov"]))) or 1.0
    if residual / scale > atol:
        raise AssertionError(f"total != within + between (max residual {residual:.3e}, scale {scale:.3e})")
    return residual


def mi_decomposition(
    nodes: Sequence[Node],
    design: PerturbationDesign,
    *,
    estimator: str = "maximum_likelihood",
    base: float = 2,
    min_cells: int = 1,
) -> dict[str, np.ndarray]:
    """Split pairwise mutual information into total, within and interaction.

    Returns symmetric matrices:

    ``total``
        ``I(A;B)`` over all cells - what pooled FastPIDC scores.
    ``within``
        ``I(A;B|P) = sum_p p(p) I(A;B|P=p)``, cell-count weighted.
    ``within_equal``
        the same strata averaged with equal weight per condition.
    ``coinformation``
        ``II(A;B;P) = total - within``. Positive means the condition label
        explains part of the pairwise dependence.

    Every stratum reuses the globally fitted bins (see :func:`subset_node`),
    so the per-stratum estimates are commensurable. With one stratum,
    ``within`` reduces exactly to ``total``.
    """
    strata = design.strata(min_cells=min_cells)
    n = len(nodes)

    def pairwise(subset: Sequence[Node]) -> np.ndarray:
        out = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                out[i, j] = out[j, i] = get_mi_and_si(subset[i], subset[j], estimator, base)[0]
        return out

    total = pairwise(list(nodes))
    within = np.zeros_like(total)
    within_equal = np.zeros_like(total)
    for (_, mask), wc, we in zip(strata, _weights(strata, False), _weights(strata, True)):
        mi_p = pairwise([subset_node(node, mask, estimator) for node in nodes])
        within += wc * mi_p
        within_equal += we * mi_p

    return {
        "total": total,
        "within": within,
        "within_equal": within_equal,
        "coinformation": total - within,
    }


# --- the directed arm ------------------------------------------------------


def intervention_node(
    design: PerturbationDesign,
    gene: str,
    *,
    estimator: str = "maximum_likelihood",
) -> tuple[Node, np.ndarray]:
    """The intervention variable ``P_gene``, and the cells it is defined on.

    ``P_gene`` is categorical over ``S_gene = control cells + cells whose
    condition targets gene``:

    .. code-block:: text

        0  control
        1  first available direction   (activation, if present)
        2  second available direction  (inhibition, if both are present)

    Its arity is ``1 + len(design.directions(gene))`` - 2 for a
    knockdown-only gene, 3 when both directions exist - so a dataset with
    mixed coverage needs no branch, only a different number of levels.

    Because cells are assigned to conditions by experimental design, this
    variable is independent of every common cause of gene expression. That
    is what makes ``I(P_gene; B)`` an interventional quantity rather than
    an observational association, and what makes it asymmetric:
    ``I(P_A; B)`` and ``I(P_B; A)`` are statements about different
    experiments.
    """
    directions = design.directions(gene)
    if not directions:
        raise ValueError(f"{gene!r} was never perturbed; it has no intervention variable")

    mask = design.control_mask().copy()
    for d in directions:
        mask |= design.mask(gene, d)

    target, direction = design.target[mask], design.direction[mask]
    codes = np.zeros(int(mask.sum()), dtype=np.int64)
    for level, d in enumerate(directions, start=1):
        codes[(target == gene) & (direction == d)] = level

    n_levels = 1 + len(directions)
    frequencies = get_frequencies_from_bin_ids(codes, n_levels)
    node = Node(f"do({gene})", codes, n_levels, get_probabilities(estimator, frequencies))
    return node, mask


def intervention_information(
    nodes: Sequence[Node],
    design: PerturbationDesign,
    *,
    estimator: str = "maximum_likelihood",
    base: float = 2,
    n_permutations: int = 0,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Total interventional effect ``T[a, b] = I(P_a ; B)``, as an
    asymmetric matrix.

    Row ``a`` is the effect of perturbing gene ``a``; a gene that was never
    perturbed has no intervention variable, so its row is structurally zero
    (in general this matrix is rectangular - only targeted genes get
    outgoing scores). The diagonal ``T[a, a] = I(P_a ; A)`` is the
    perturbation's effect on its own target: not an edge, but a useful
    positive control.

    With ``n_permutations > 0`` the condition labels of each ``P_a`` are
    shuffled within its own cell set, preserving every level's size, and
    the result also carries

    ``z``
        ``(T - null_mean) / null_sd``.

    Prefer ``z`` to ``T`` for anything that compares two entries. Plug-in MI
    bias grows with the *target's* bin count, roughly
    ``(|P_a| - 1)(B_b - 1) / (2 |S_a| ln2)`` bits, so ``T[a, b]`` and
    ``T[b, a]`` are biased by different amounts whenever genes ``a`` and
    ``b`` were discretized into different numbers of bins - which Bayesian
    blocks routinely does. Left uncorrected that is a systematic bias
    toward calling the finer-binned gene the effect.
    """
    n = len(nodes)
    rng = np.random.default_rng() if rng is None else rng
    labels = [node.label for node in nodes]

    t = np.zeros((n, n), dtype=np.float64)
    null_mean = np.zeros((n, n), dtype=np.float64)
    null_sd = np.zeros((n, n), dtype=np.float64)

    for a, gene in enumerate(labels):
        if not design.directions(gene):
            continue
        p_node, mask = intervention_node(design, gene, estimator=estimator)
        targets = [subset_node(node, mask, estimator) for node in nodes]

        for b, target in enumerate(targets):
            t[a, b] = get_mi_and_si(p_node, target, estimator, base)[0]

        if n_permutations <= 0:
            continue

        draws = np.empty((n_permutations, n), dtype=np.float64)
        for k in range(n_permutations):
            shuffled = Node(
                p_node.label,
                rng.permutation(p_node.binned_values),
                p_node.number_of_bins,
                p_node.probabilities,
            )
            for b, target in enumerate(targets):
                draws[k, b] = get_mi_and_si(shuffled, target, estimator, base)[0]
        null_mean[a] = draws.mean(axis=0)
        null_sd[a] = draws.std(axis=0, ddof=1)

    result = {"T": t}
    if n_permutations > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (t - null_mean) / null_sd
        result |= {"z": np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), "null_mean": null_mean, "null_sd": null_sd}
    return result


def compose_directed(detect: np.ndarray, orient: np.ndarray, *, soft: bool = True) -> np.ndarray:
    """Combine a detection matrix with an orientation matrix.

    Stage 2 found that mediator correction detects worse than raw ``T`` but
    orients better, so the two jobs are best done by different scores. This
    ranks each gene pair by ``detect`` and splits that score between the two
    directions according to ``orient``:

    .. code-block:: text

        s[i,j]  = max(|detect[i,j]|, |detect[j,i]|)      # skeleton strength
        w[i,j]  = |orient[i,j]| / (|orient[i,j]| + |orient[j,i]|)
        out     = s * w

    With ``soft=True`` a pair that both scores orient comparably keeps weight
    in both directions, which is the right answer for a genuinely reciprocal
    edge. With ``soft=False`` the winning direction takes the whole score,
    and the skeleton ranking is then *exactly* ``detect``'s - the composite
    can only change orientation, never detection.

    Where ``orient`` is zero in both directions the score is split evenly,
    since there is no orientation claim to make. An orienter that is zero
    *everywhere* makes no directional claim at all - which is exactly what
    happens on unperturbed data, where ``T`` is identically zero - so that
    case warns instead of quietly returning a half-and-half matrix.

    Prefer ``soft=False``. A bootstrap study over both benchmarks found the
    soft split rescales the skeleton and costs real accuracy where the
    detector is strong (modular skeleton AP 0.771 -> 0.567), while the hard
    form preserves it exactly. See ``scratch/docs/w1_w2_results.md``.
    """
    if detect.shape != orient.shape:
        raise ValueError(f"detect {detect.shape} and orient {orient.shape} must have the same shape")
    if not np.any(orient):
        warnings.warn(
            "orient is zero everywhere, so every edge will be split evenly between its two "
            "directions. On unperturbed data `intervention_information` returns an all-zero T; "
            "build the orienter from perturbed cells.",
            RuntimeWarning,
            stacklevel=2,
        )

    strength = np.maximum(np.abs(detect), np.abs(detect).T)
    forward, reverse = np.abs(orient), np.abs(orient).T
    total = forward + reverse

    if soft:
        weight = np.where(total > 0, forward / np.where(total > 0, total, 1.0), 0.5)
    else:
        weight = np.where(forward > reverse, 1.0, np.where(forward < reverse, 0.0, 0.5))

    out = strength * weight
    np.fill_diagonal(out, 0.0)
    return out


def dose_intervention_node(
    strengths: np.ndarray,
    control_mask: np.ndarray,
    label: str,
    *,
    include_control: bool = False,
    estimator: str = "maximum_likelihood",
) -> tuple[Node, np.ndarray]:
    """``P_a`` for a *dense* perturbation design, coded by dose sign.

    Returns ``(node, cell mask)``, like :func:`intervention_node`.

    :class:`PerturbationDesign` models one targeted gene per cell. Some
    benchmarks instead perturb **every** gene in **every** condition with a
    continuous random strength (D-SPIN's large-network protocol draws
    ``r* ~ Normal(0, 0.8)`` truncated to ``[-1, 1]`` for all genes). There is
    no targeted gene there, so a "was this gene targeted" indicator would be
    constant and carry no information; the faithful analogue is the sign of
    the dose.

    ``strengths`` is this gene's per-cell applied strength, zero in control
    cells.

    **Leaving out the control cells (the default) scores better.**
    Including them adds a level that is *identical for every gene* - the
    same control cells against the same perturbed cells - so part of
    ``I(P_a; B)`` is a shared control-vs-perturbed contrast that says
    nothing about which gene was dosed. On the modular benchmark this costs
    roughly 0.03 skeleton AP and 0.05 orientation accuracy. It is a
    dilution, not a disaster: the positive control (a dose moving its own
    gene most) passes either way.

    A single-gene design does not have the issue at all, because there each
    gene's cell set is control plus *its own* conditions, so the control
    group differs per gene.

    Because every perturbed cell carries a dose for every gene, all genes
    share one cell set here - which lets a caller build the
    specific-information cache once instead of once per gene.

    Magnitude is discarded. That loses real information in a graded design,
    but keeps the statistic identical in form to the single-gene case so the
    two benchmarks stay comparable.
    """
    strengths = np.asarray(strengths, dtype=np.float64)
    control_mask = np.asarray(control_mask, dtype=bool)
    if strengths.shape != control_mask.shape:
        raise ValueError(f"strengths {strengths.shape} and control_mask {control_mask.shape} must match")

    if include_control:
        mask = np.ones(strengths.size, dtype=bool)
        codes = np.zeros(strengths.size, dtype=np.int64)
        perturbed = ~control_mask
        codes[perturbed & (strengths > 0)] = 1
        codes[perturbed & (strengths < 0)] = 2
        n_levels = 3
    else:
        # A zero dose in a perturbed cell is neither sign, so it is excluded
        # rather than silently folded into one of them.
        mask = (~control_mask) & (strengths != 0)
        codes = (strengths[mask] > 0).astype(np.int64)
        n_levels = 2

    frequencies = get_frequencies_from_bin_ids(codes, n_levels)
    return Node(label, codes, n_levels, get_probabilities(estimator, frequencies)), mask
