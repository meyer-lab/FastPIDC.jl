"""Tests for .h5ad input and the perturbation-condition decomposition."""

import h5py
import numpy as np
import pytest

from fastpidc.io import get_nodes, read_h5ad_obs
from fastpidc.network import get_mi_and_si
from fastpidc.perturbation import (
    ACTIVATION,
    INHIBITION,
    PerturbationDesign,
    check_covariance_identity,
    compose_directed,
    correlation_decomposition,
    dose_intervention_node,
    intervention_information,
    intervention_node,
    mi_decomposition,
    subset_node,
)
from fastpidc.types import Node

STR = h5py.string_dtype()


def _write_categorical(group, name, values):
    categories = sorted(set(values))
    codes = np.array([categories.index(v) for v in values], dtype=np.int8)
    sub = group.create_group(name)
    sub.attrs["encoding-type"] = "categorical"
    sub.attrs["encoding-version"] = "0.2.0"
    sub.create_dataset("categories", data=np.array(categories, dtype=object), dtype=STR)
    sub.create_dataset("codes", data=codes)


def write_h5ad(path, x, genes, obs_columns):
    """Write a minimal AnnData-encoded .h5ad with h5py alone."""
    with h5py.File(path, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        f.create_dataset("X", data=np.asarray(x, dtype=np.float32))

        var = f.create_group("var")
        var.attrs["_index"] = "gene"
        var.attrs["encoding-type"] = "dataframe"
        var.create_dataset("gene", data=np.array(list(genes), dtype=object), dtype=STR)

        obs = f.create_group("obs")
        obs.attrs["_index"] = "cell"
        obs.attrs["encoding-type"] = "dataframe"
        obs.create_dataset("cell", data=np.array([f"c{i}" for i in range(len(x))], dtype=object), dtype=STR)
        for name, values in obs_columns.items():
            if np.asarray(values).dtype == bool:
                obs.create_dataset(name, data=np.asarray(values))
            else:
                _write_categorical(obs, name, list(values))
    return path


@pytest.fixture
def toy(tmp_path):
    """3 genes, 4 conditions: control, A both directions, B knockdown only.
    C is never perturbed. 20 cells each, unequal control size on purpose."""
    rng = np.random.default_rng(0)
    blocks = [("Control", "control", 40), ("A", "activation", 20), ("A", "knockdown", 20), ("B", "knockdown", 20)]
    x, target, direction = [], [], []
    for offset, (gene, kind, n) in enumerate(blocks):
        x.append(rng.normal(loc=offset, scale=1.0, size=(n, 3)))
        target += [gene] * n
        direction += [kind] * n
    path = write_h5ad(
        tmp_path / "toy.h5ad",
        np.vstack(x),
        ["A", "B", "C"],
        {"target_gene": target, "perturbation_type": direction},
    )
    return path


# --- .h5ad input ---------------------------------------------------------


def test_read_h5ad_obs_decodes_categoricals_in_cell_order(toy):
    obs = read_h5ad_obs(str(toy))
    assert set(obs) == {"cell", "target_gene", "perturbation_type"}
    assert obs["target_gene"][0] == "Control"
    assert obs["target_gene"][40] == "A"
    assert obs["perturbation_type"][40] == "activation"
    assert obs["cell"][0] == "c0"
    assert all(len(v) == 100 for v in obs.values())


def test_read_h5ad_obs_selects_columns_and_reports_missing(toy):
    assert list(read_h5ad_obs(str(toy), ["target_gene"])) == ["target_gene"]
    with pytest.raises(ValueError, match="not found"):
        read_h5ad_obs(str(toy), ["nope"])


def test_get_nodes_h5ad_reads_var_index_and_dense_x(toy):
    nodes = get_nodes(str(toy), discretizer="uniform_width", number_of_bins=4, bb_backend="cpu")
    assert [n.label for n in nodes] == ["A", "B", "C"]
    assert all(n.binned_values.size == 100 for n in nodes)


def test_get_nodes_h5ad_sparse_csr(tmp_path):
    from scipy import sparse

    rng = np.random.default_rng(1)
    dense = rng.random((30, 3))
    csr = sparse.csr_matrix(dense)
    path = tmp_path / "sparse.h5ad"
    with h5py.File(path, "w") as f:
        f.attrs["encoding-type"] = "anndata"
        x = f.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["shape"] = np.array(csr.shape)
        x.create_dataset("data", data=csr.data)
        x.create_dataset("indices", data=csr.indices)
        x.create_dataset("indptr", data=csr.indptr)
        var = f.create_group("var")
        var.attrs["_index"] = "gene"
        var.create_dataset("gene", data=np.array(["A", "B", "C"], dtype=object), dtype=STR)

    nodes = get_nodes(str(path), discretizer="uniform_width", number_of_bins=3, bb_backend="cpu")
    assert [n.label for n in nodes] == ["A", "B", "C"]
    assert nodes[0].binned_values.size == 30


def test_get_nodes_h5ad_rejects_mismatched_var(tmp_path):
    path = tmp_path / "bad.h5ad"
    write_h5ad(path, np.zeros((5, 3)), ["A", "B"], {"target_gene": ["Control"] * 5})
    with pytest.raises(ValueError, match="columns but var has"):
        get_nodes(str(path), discretizer="uniform_width", bb_backend="cpu")


# --- design and coverage -------------------------------------------------


def test_design_coverage_handles_mixed_direction_availability(toy):
    obs = read_h5ad_obs(str(toy))
    design = PerturbationDesign.from_obs(obs, ["A", "B", "C"])

    assert design.directions("A") == (ACTIVATION, INHIBITION)
    assert design.directions("B") == (INHIBITION,)
    assert design.directions("C") == ()
    assert design.perturbed_genes() == ("A", "B")
    assert design.coverage() == {
        "": {"": 40},
        "A": {ACTIVATION: 20, INHIBITION: 20},
        "B": {INHIBITION: 20},
    }
    # One gene with two directions and one with one gives 3 perturbed strata
    # plus control -- no dataset-level branch decides this.
    assert [label for label, _ in design.strata()] == ["control", f"A{ACTIVATION}", f"A{INHIBITION}", f"B{INHIBITION}"]


def test_design_warns_on_unknown_direction_and_off_matrix_target(toy):
    obs = read_h5ad_obs(str(toy))
    with pytest.warns(RuntimeWarning, match="Unrecognized"):
        PerturbationDesign.from_obs(obs, ["A", "B", "C"], inhibition_values=("nothing_matches",))
    with pytest.warns(RuntimeWarning, match="not among the genes"):
        PerturbationDesign.from_obs(obs, ["B", "C"])


def test_strata_drops_small_strata_with_a_warning(toy):
    obs = read_h5ad_obs(str(toy))
    design = PerturbationDesign.from_obs(obs, ["A", "B", "C"])
    with pytest.warns(RuntimeWarning, match="fewer than 30 cells"):
        kept = design.strata(min_cells=30)
    assert [label for label, _ in kept] == ["control"]


# --- subset_node ---------------------------------------------------------


def test_subset_node_keeps_global_bins_and_reestimates_marginal():
    node = Node.from_raw_values("A", np.arange(100.0), "uniform_width", "maximum_likelihood", 4)
    mask = np.arange(100) < 25
    sub = subset_node(node, mask)

    assert sub.number_of_bins == node.number_of_bins
    np.testing.assert_array_equal(sub.binned_values, node.binned_values[:25])
    # The first quarter of a uniform ramp falls entirely in the first bin.
    np.testing.assert_allclose(sub.probabilities, [1.0, 0.0, 0.0, 0.0])
    assert pytest.approx(sub.probabilities.sum()) == 1.0


# --- the decompositions --------------------------------------------------


def test_covariance_identity_is_exact(toy):
    obs = read_h5ad_obs(str(toy))
    design = PerturbationDesign.from_obs(obs, ["A", "B", "C"])
    with h5py.File(toy, "r") as f:
        values = np.asarray(f["X"][()], dtype=np.float64)

    result = correlation_decomposition(values, design)
    # Law of total covariance, to machine precision.
    assert check_covariance_identity(result) < 1e-10


def test_single_stratum_reduces_to_unconditioned(toy):
    """The regression test the plan requires: with one stratum the
    decomposition must reproduce the plain pooled statistic exactly."""
    nodes = get_nodes(str(toy), discretizer="uniform_width", number_of_bins=4, bb_backend="cpu")
    with h5py.File(toy, "r") as f:
        values = np.asarray(f["X"][()], dtype=np.float64)

    empty = np.array([""] * values.shape[0], dtype=object)
    one_stratum = PerturbationDesign(target=empty, direction=empty.copy(), gene_labels=("A", "B", "C"))

    mi = mi_decomposition(nodes, one_stratum)
    np.testing.assert_allclose(mi["within"], mi["total"], rtol=0, atol=0)
    np.testing.assert_allclose(mi["within_equal"], mi["total"], rtol=0, atol=0)
    np.testing.assert_array_equal(mi["coinformation"], np.zeros_like(mi["total"]))

    # ...and it must equal what the existing estimator gives on those nodes.
    expected = get_mi_and_si(nodes[0], nodes[1], "maximum_likelihood", 2)[0]
    assert mi["total"][0, 1] == pytest.approx(expected)

    corr = correlation_decomposition(values, one_stratum)
    np.testing.assert_allclose(corr["within"], corr["total"], atol=1e-12)
    np.testing.assert_allclose(corr["between"], 0.0, atol=1e-12)


def test_equal_and_cell_count_weighting_differ_on_unbalanced_strata(toy):
    """Control is 40 of 100 cells but one of four conditions, so the two
    weightings are materially different -- they must not be conflated."""
    nodes = get_nodes(str(toy), discretizer="uniform_width", number_of_bins=4, bb_backend="cpu")
    obs = read_h5ad_obs(str(toy))
    design = PerturbationDesign.from_obs(obs, ["A", "B", "C"])

    mi = mi_decomposition(nodes, design)
    assert not np.allclose(mi["within"], mi["within_equal"])
    # Only the cell-count arm satisfies I(A;B) = I(A;B|P) + II(A;B;P).
    np.testing.assert_allclose(mi["coinformation"], mi["total"] - mi["within"], atol=1e-12)


def test_shuffling_labels_collapses_within_onto_total(toy):
    """Shuffled condition labels make every stratum a random subsample, so
    the within arm must converge on the pooled statistic. This is why the
    within arm needs a shuffled-label null to mean anything."""
    obs = read_h5ad_obs(str(toy))
    design = PerturbationDesign.from_obs(obs, ["A", "B", "C"])
    with h5py.File(toy, "r") as f:
        values = np.asarray(f["X"][()], dtype=np.float64)

    rng = np.random.default_rng(0)
    order = rng.permutation(design.n_cells)
    null = PerturbationDesign(
        target=design.target[order], direction=design.direction[order], gene_labels=("A", "B", "C")
    )

    real = correlation_decomposition(values, design)
    shuffled = correlation_decomposition(values, null)
    offset = np.abs(real["within"] - real["total"]).max()
    assert np.abs(shuffled["within"] - shuffled["total"]).max() < offset


# --- the directed arm ----------------------------------------------------


def test_intervention_node_arity_follows_available_directions(toy):
    obs = read_h5ad_obs(str(toy))
    design = PerturbationDesign.from_obs(obs, ["A", "B", "C"])

    node_a, mask_a = intervention_node(design, "A")
    node_b, mask_b = intervention_node(design, "B")

    # A has both directions (3 levels), B only knockdown (2) -- same code path.
    assert node_a.number_of_bins == 3
    assert node_b.number_of_bins == 2
    assert set(np.unique(node_a.binned_values).tolist()) == {0, 1, 2}
    assert set(np.unique(node_b.binned_values).tolist()) == {0, 1}

    # S_A is control plus A's own cells, and nothing else.
    assert mask_a.sum() == 40 + 20 + 20
    assert mask_b.sum() == 40 + 20
    assert not np.any(mask_b & design.mask("A", ACTIVATION))
    assert node_a.binned_values.size == mask_a.sum()


def test_intervention_node_rejects_an_unperturbed_gene(toy):
    obs = read_h5ad_obs(str(toy))
    design = PerturbationDesign.from_obs(obs, ["A", "B", "C"])
    with pytest.raises(ValueError, match="never perturbed"):
        intervention_node(design, "C")


@pytest.fixture
def causal(tmp_path):
    """A -> B, with no path back. Perturbing A moves B; perturbing B leaves
    A alone. C is independent of both and never perturbed."""
    rng = np.random.default_rng(7)
    blocks = []
    for gene, kind, n, a_loc, b_override in [
        ("Control", "control", 300, 0.0, None),
        ("A", "knockdown", 150, -3.0, None),
        ("B", "knockdown", 150, 0.0, -3.0),
    ]:
        a = rng.normal(a_loc, 1.0, n)
        b = a + rng.normal(0, 0.3, n) if b_override is None else rng.normal(b_override, 0.3, n)
        c = rng.normal(0, 1.0, n)
        blocks.append((np.column_stack([a, b, c]), [gene] * n, [kind] * n))

    x = np.vstack([b[0] for b in blocks])
    target = [t for b in blocks for t in b[1]]
    direction = [d for b in blocks for d in b[2]]
    path = write_h5ad(
        tmp_path / "causal.h5ad", x, ["A", "B", "C"], {"target_gene": target, "perturbation_type": direction}
    )
    return path


def test_intervention_information_recovers_direction(causal):
    """The core claim: I(P_A; B) >> I(P_B; A) when A causes B."""
    nodes = get_nodes(str(causal), discretizer="uniform_width", number_of_bins=8, bb_backend="cpu")
    obs = read_h5ad_obs(str(causal))
    design = PerturbationDesign.from_obs(obs, ["A", "B", "C"])
    a, b, c = 0, 1, 2

    t = intervention_information(nodes, design)["T"]

    assert t[a, b] > 5 * t[b, a]
    # The perturbation moves its own target most of all (positive control).
    assert t[a, a] > t[a, b] > t[a, c]
    # C was never perturbed, so it has no outgoing interventional score.
    np.testing.assert_array_equal(t[c], np.zeros(3))
    # ...and nothing propagates to it either.
    assert t[a, c] < 0.05


def test_intervention_information_z_scores_against_a_shuffled_null(causal):
    nodes = get_nodes(str(causal), discretizer="uniform_width", number_of_bins=8, bb_backend="cpu")
    obs = read_h5ad_obs(str(causal))
    design = PerturbationDesign.from_obs(obs, ["A", "B", "C"])

    plain = intervention_information(nodes, design)
    assert "z" not in plain

    result = intervention_information(nodes, design, n_permutations=30, rng=np.random.default_rng(0))
    assert set(result) == {"T", "z", "null_mean", "null_sd"}

    # A real effect stands far above its null; an absent one does not.
    assert result["z"][0, 1] > 10
    assert abs(result["z"][0, 2]) < 5
    # The null is strictly positive: plug-in MI is biased upward, which is
    # exactly why the raw T of an absent edge is not zero.
    assert result["null_mean"][0, 2] > 0
    assert result["T"][0, 2] > 0


def test_plugin_bias_grows_with_the_targets_bin_count(causal):
    """The orientation artifact, demonstrated: with no real effect, the
    shuffled-null MI is larger for a target with more bins. This is why
    T[a,b] and T[b,a] are not directly comparable across genes binned
    differently."""
    obs = read_h5ad_obs(str(causal))
    rng = np.random.default_rng(3)
    nulls = {}
    for n_bins in (4, 16):
        nodes = get_nodes(str(causal), discretizer="uniform_width", number_of_bins=n_bins, bb_backend="cpu")
        design = PerturbationDesign.from_obs(obs, ["A", "B", "C"])
        result = intervention_information(nodes, design, n_permutations=30, rng=rng)
        # A -> C is absent, so this is bias alone.
        nulls[n_bins] = result["null_mean"][0, 2]
    assert nulls[16] > 2 * nulls[4]


# --- composing detection with orientation --------------------------------


def test_compose_hard_preserves_the_detection_skeleton_exactly():
    """The hard composite may only change orientation, never detection: the
    max over each pair's two directions must equal the detector's own
    symmetric skeleton score."""
    detect = np.array([[0.0, 0.9, 0.2], [0.1, 0.0, 0.5], [0.3, 0.4, 0.0]])
    orient = np.array([[0.0, 0.1, 0.8], [0.7, 0.0, 0.2], [0.1, 0.9, 0.0]])

    out = compose_directed(detect, orient, soft=False)
    expected = np.maximum(np.abs(detect), np.abs(detect).T)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(np.maximum(out, out.T), expected)


def test_compose_takes_direction_from_the_orientation_matrix():
    detect = np.array([[0.0, 0.9], [0.1, 0.0]])
    # detect prefers 0 -> 1, orient prefers 1 -> 0; orient must win.
    orient = np.array([[0.0, 0.2], [0.8, 0.0]])

    out = compose_directed(detect, orient, soft=False)
    assert out[1, 0] == pytest.approx(0.9)
    assert out[0, 1] == 0.0


def test_compose_soft_keeps_both_directions_for_a_balanced_pair():
    detect = np.array([[0.0, 0.8], [0.0, 0.0]])
    orient = np.array([[0.0, 0.5], [0.5, 0.0]])

    soft = compose_directed(detect, orient, soft=True)
    # A reciprocal-looking pair keeps weight both ways rather than being
    # forced to pick one.
    assert soft[0, 1] == pytest.approx(0.4)
    assert soft[1, 0] == pytest.approx(0.4)


def test_compose_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        compose_directed(np.zeros((3, 3)), np.zeros((2, 2)))


# --- dense (combinatorial) perturbation designs ---------------------------


def test_dose_intervention_node_sign_codes_over_perturbed_cells():
    #            control        +dose   -dose   zero dose
    strengths = np.array([0.0, 0.0, 0.7, -0.4, 0.0])
    control = np.array([True, True, False, False, False])

    node, mask = dose_intervention_node(strengths, control, "do(A)")

    assert node.number_of_bins == 2
    # Control cells are excluded, and so is a perturbed cell with zero dose --
    # it is neither sign, so folding it into one would be a silent error.
    np.testing.assert_array_equal(mask, [False, False, True, True, False])
    np.testing.assert_array_equal(node.binned_values, [1, 0])
    assert node.binned_values.size == mask.sum()


def test_dose_intervention_node_can_include_control_as_a_third_level():
    strengths = np.array([0.0, 0.7, -0.4])
    control = np.array([True, False, False])

    node, mask = dose_intervention_node(strengths, control, "do(A)", include_control=True)

    assert node.number_of_bins == 3
    assert mask.all()
    np.testing.assert_array_equal(node.binned_values, [0, 1, 2])


def test_dose_intervention_node_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        dose_intervention_node(np.zeros(4), np.zeros(3, dtype=bool), "do(A)")


def test_dose_intervention_recovers_its_own_target_in_a_dense_design():
    """The positive control for a dense design: every gene is dosed in every
    condition, and a gene's own dose must still move its own expression more
    than anything else's does.

    This is the check that caught a real off-by-one in dose/condition
    alignment, which otherwise looked exactly like a negative result.
    """
    rng = np.random.default_rng(5)
    n_conditions, n_cells, n_genes = 60, 20, 4
    doses = rng.uniform(-1, 1, size=(n_conditions, n_genes))

    rows, strengths = [], []
    for c in range(n_conditions):
        # Each gene responds to its OWN dose plus independent noise.
        base = rng.normal(0, 0.3, size=(n_cells, n_genes))
        rows.append(base + 3.0 * doses[c])
        strengths.append(np.repeat(doses[c][None, :], n_cells, axis=0))
    x = np.vstack(rows)
    strength = np.vstack(strengths)
    control = np.zeros(x.shape[0], dtype=bool)

    nodes = [Node.from_raw_values(f"g{i}", x[:, i], "uniform_width", "maximum_likelihood", 6) for i in range(n_genes)]
    t = np.zeros((n_genes, n_genes))
    for a in range(n_genes):
        p_node, _mask = dose_intervention_node(strength[:, a], control, f"do(g{a})")
        for b, target in enumerate(nodes):
            t[a, b] = get_mi_and_si(p_node, target, "maximum_likelihood", 2)[0]

    np.testing.assert_array_equal(t.argmax(axis=1), np.arange(n_genes))


def test_compose_warns_when_the_orienter_makes_no_claim():
    """An all-zero orienter is what unperturbed data produces; splitting
    every edge evenly in silence would hide that."""
    with pytest.warns(RuntimeWarning, match="zero everywhere"):
        out = compose_directed(np.array([[0.0, 0.6], [0.0, 0.0]]), np.zeros((2, 2)))
    # It still splits evenly -- it just no longer does so silently.
    assert out[0, 1] == pytest.approx(0.3)
    assert out[1, 0] == pytest.approx(0.3)
