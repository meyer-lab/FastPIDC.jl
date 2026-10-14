# Perturbation-aware inference: the interventional matrix `T`

Implemented in `fastpidc.perturbation`. Additive — nothing in FastPIDC's
existing path changes, and on data with no perturbation labels the package
behaves exactly as it always has.

## 1. The problem

PIDC, MI, CLR and PUC all score **observational dependence** between gene
expression levels. A Perturb-seq experiment carries something those methods
never read: a per-cell column saying *which gene this cell's guide targeted*.

That column is not expression data. It is the experimental design, and it is
assigned by the experimenter rather than by the cell's state. Using it changes
what kind of claim the score can support.

## 2. `P_a` and `T`

For each gene `a` that some guide targeted, define an **intervention
variable** `P_a` on the cells where the question "was `a` perturbed here?"
is meaningful — control cells plus `a`'s own condition cells:

```
S_a  =  control cells  ∪  cells whose guide targeted a

P_a  =  0   control cell
        1   first available direction   (activation, if present)
        2   second available direction  (inhibition, if both are present)

T[a, b]  =  I(P_a ; B)      estimated over S_a only
```

The arity is `1 + len(design.directions(a))` — 2 for a knockdown-only screen
like CRISPRi, 3 when both directions exist. A dataset with mixed coverage
needs no special case, only a different number of levels per gene.

Two properties follow from the construction, and they are the whole point:

**It is interventional, not associational.** Condition assignment is
randomized by design, so `P_a` is independent of every common cause of gene
expression. There is no backdoor path from `P_a` to `B`. `I(P_a; B) > 0`
means perturbing `a` *moved* `b`, not that `a` and `b` covary.

**It is asymmetric.** `I(P_A; B)` and `I(P_B; A)` are estimated on different
cell sets and describe different experiments. They are not two readings of one
correlation, so a single matrix both ranks edges and orients them.

It has **no fitted parameters**.

## 3. Reading the matrix

`T` is rectangular in general — `perturbed genes × all genes` — because only
targeted genes have an intervention variable. It is square only when every
gene was perturbed, which happens in benchmarks and not in real screens.

```
skeleton strength of the pair (a, b)    max(T[a,b], T[b,a])
orientation                             whichever of the two is larger
```

`fastpidc.perturbation.compose_directed(detect, orient, soft=False)` does this
when you want to detect with one score and orient with another. Prefer
`soft=False`: the hard split preserves the detector's skeleton ranking
exactly, while the soft split rescales it and costs real accuracy where the
detector is strong (modular skeleton AP 0.771 → 0.567).

`T[a, a] = I(P_a; A)` is the perturbation's effect on its own target. It is
not an edge — exclude it from scoring — but it is the **positive control**,
and you should look at it before you look at anything else. See §6.

## 4. Scope — what this does not do

`T` has rows only for genes that were perturbed. If gene `c` was never
targeted, `P_c` is the constant `0` across every cell, `H(P_c) = 0`, and
`I(P_c; B) = 0` for every `b` — identically, not approximately.
`intervention_node` raises for such a gene and `intervention_information`
skips the row. This is not a threshold or a sparsity artifact; there is no
experiment to read.

So: **`T` is not a replacement for PIDC.** It is an additional mode available
when perturbation labels exist, covering the targeted genes. Observational
inference over the full gene set is still PIDC's job, and FastPIDC's
control-only path is untouched by any of this.

Whether interventional evidence can be *propagated* to unperturbed regulators
is an open question with its own plan (`scratch/docs/propagation_plan.md`,
untracked -- `scratch/` is gitignored). One bivariate attempt at it
(perturbation-response-profile correlation) was tested and lost to doing
nothing with the perturbations at all — §7.

## 5. Estimator caveats

**Discretize globally once, then mask.** Bin edges must be shared across
strata or the per-stratum values are not commensurable. `subset_node` carries
bin ids over unchanged and only re-estimates the marginal, which is the
correct operation; re-running Bayesian blocks per subset is not.

**Plug-in MI bias is not symmetric.** It grows with the *target's* bin count,
roughly

```
(|P_a| − 1)(B_b − 1) / (2 |S_a| ln 2)   bits
```

so `T[a,b]` and `T[b,a]` are biased by different amounts whenever Bayesian
blocks gave genes `a` and `b` different bin counts — which it routinely does.
Left uncorrected, that is a systematic bias toward calling the finer-binned
gene the effect. Pass `n_permutations > 0` and prefer the returned `z` for
anything that compares two entries directly. (At 3k–20k cells the bias turned
out to be immaterial to the rankings measured here, but that is a result, not
something to assume on new data.)

**Stratified MI inflates k-fold.** `I(A;B|P) = Σ_p p(p) I(A;B|P=p)` carries
roughly the single-stratum bias once per stratum. Always compare a stratified
arm against a shuffled-label null rather than against zero.

## 6. The positive control — run it first

A knockdown of gene `a` must move `a` more than it moves any other gene:
`T[a, a]` should be the maximum of row `a`.

On the hESC CRISPRi pilot (2,000 genes, 31,074 cells) this held for
**138 of 150** regulators, worst rank 57 of 2,000. If it drops far below ~90%,
the label alignment is wrong.

This is not a formality. An off-by-one in a perturbation matrix — indexing a
per-condition dose table that already includes the control at row 0 — produced
a completely convincing but entirely fake negative result during development,
shifting every dose onto the neighbouring condition. The positive control is
what caught it; nothing else did.

## 7. Evidence

Simulated benchmarks are 10 replicates (5 for modular) from
`dspin-benchmark-sim`, scored by average precision against known truth.
Absolute values are not comparable to published numbers — these replicates are
harder than the authors' released realization (D-SPIN scores 0.802 skeleton
here against 0.9005 there). Only within-table comparisons are sound.

**HSC — 11 genes, 36% dense, single-gene conditions** (skeleton AP):

| arm | KD-only | full-22 |
|---|---:|---:|
| **T** | **0.858 ± 0.023** | **0.830 ± 0.018** |
| PIDC pooled | 0.691 ± 0.013 | 0.723 ± 0.007 |
| PIDC control-only | 0.620 ± 0.019 | 0.615 ± 0.014 |

Orientation: 0.893 ± 0.066 of 14 unidirectional pairs correct (t = 17.9 vs
chance). PIDC cannot orient at all.

**Modular — 125 genes, 6.4% dense, dense combinatorial doses:**

| arm | skeleton AP | directed AP | orientation (486 pairs) |
|---|---:|---:|---:|
| T | 0.345 ± 0.025 | 0.308 ± 0.028 | **0.814 ± 0.022** |
| PIDC pooled | **0.784 ± 0.009** | — | — |
| PIDC-detect + T-orient | **0.784** | **0.635 ± 0.018** | 0.814 |

**hESC — real CRISPRi, 150 targets, DE truth over 141 regulators**
(`log2` AUPR lift, i.e. fold enrichment over chance):

| network | median | mean | frac > 0 | paired vs T |
|---|---:|---:|---:|---|
| PIDC control-only | 1.332 | 1.675 | 0.993 | t = +9.94 |
| PIDC pooled | 1.373 | 1.687 | 0.972 | t = +9.20 |
| **T** | **2.676** | **3.135** | **1.000** | — |

≈2.5× enrichment over chance becomes ≈6.4×.

### The rule this adds up to

> **Median targeted genes per condition == 1 → `T` detects.
> Many → PIDC detects. `T` orients either way.**

Detection is design-dependent and the sign flips: `T` beats pooled PIDC by
+0.167 on single-gene designs and loses by −0.439 on dense combinatorial ones,
where 124 other simultaneous doses are overwhelming noise in every pairwise
interventional contrast. Orientation generalized across both.

Against D-SPIN 1.5.2 on identical data, with its published perturbation prior:
skeleton is a tie (0.830 vs 0.815, t = 1.92, not significant), directed AP
favours `T` (0.762 vs 0.643, t = 9.60), orientation favours D-SPIN (0.843 vs
0.950). *"Competitive with D-SPIN, with zero fitted parameters"* is
supportable; *"beats D-SPIN"* is not.

## 8. What was tried and dropped

Recorded so it is not rediscovered. The full evaluation write-ups and the
scripts that produced every number live in the untracked `scratch/` tree.

| attempt | why it was dropped |
|---|---|
| `I_min` mediator correction (`PUC_do` / `DUC`) | Williams–Beer redundancy degenerates when a source is an intervention label: a neighbouring gene is always more informative, so `Red = I(P_a;B)` exactly and unique information is zero *by construction*, path or no path. Measured on a direct edge: `T = 0.6651`, `Red = 0.6651`. Failed on both a 36%-dense and a 5.6× sparser network. |
| Conditional-independence screening | Avoids that degeneracy; still lost at detection on both benchmarks. |
| Bootstrap orientation confidence | Calibration was monotone but badly overconfident, and feeding it into `compose_directed` moved directed AP by ±0.001. A bootstrap over cells sees only sampling noise in `T`, which is tiny at these cell counts, and is structurally blind to the bin-count estimator bias. |
| A single universal detect/orient recipe | Fixing either detector costs 0.17–0.19 AP on HSC or 0.30 on modular. Use the design rule in §7 instead. |
| Perturbation-response-profile correlation | The one attempt at scoring *unperturbed* regulators. Worse than ignoring the perturbations: median log2 lift 0.453 vs 1.332 for control-only PIDC, paired mean −0.845, t = −5.71. It is a bivariate statistic, so two genes downstream of the same knockdowns look related whether or not either regulates the other. |
| Permutation `z` on HSC/modular | Neutral — the bias it corrects is immaterial at those cell counts. Kept, because that is a finding about this data and not a general one. |

The PID machinery contributes nothing here. The entire result is raw
`I(P_a; B)`, a plain mutual information. The honest framing is a
**perturbation mode that reuses FastPIDC's discretization**, not a PID
extension.

## 9. Worked example

Condensed from the hESC run script (`scratch/scripts/hesc_build_T.py`,
untracked).

```python
import numpy as np
from fastpidc.io import get_nodes, read_h5ad_obs
from fastpidc.network import get_mi_and_si
from fastpidc.perturbation import PerturbationDesign, intervention_node, subset_node

CONTROL_LABEL = "non-targeting"

nodes = get_nodes("hesc.h5ad", bb_backend="cuda")   # discretize ONCE, globally
labels = [n.label for n in nodes]

# hESC is CRISPRi: every perturbed cell is a knockdown, and the file has no
# direction column, so synthesize one rather than let the library guess.
target = np.asarray(read_h5ad_obs("hesc.h5ad", ["target_gene"])["target_gene"], dtype=object)
direction = np.array(
    ["control" if str(t) == CONTROL_LABEL else "knockdown" for t in target], dtype=object
)
design = PerturbationDesign.from_obs(
    {"target_gene": target, "perturbation_type": direction}, labels
)
print(design.summary())

perturbed = list(design.perturbed_genes())
index = {g: i for i, g in enumerate(labels)}
T = np.zeros((len(perturbed), len(labels)))

for row, gene in enumerate(perturbed):
    p_node, mask = intervention_node(design, gene)
    # Subset lazily: materializing every target node at once costs ~3.8 GB per
    # regulator at full scale, allocated and freed once per gene.
    T[row] = [get_mi_and_si(p_node, subset_node(n, mask), "maximum_likelihood", 2)[0]
              for n in nodes]

# Section 6. Check the rate over all regulators, not each gene individually --
# a handful of weak guides are expected.
hit = sum(T[r].argmax() == index[g] for r, g in enumerate(perturbed))
print(f"positive control: self-effect is row max for {hit}/{len(perturbed)}")
assert hit >= 0.9 * len(perturbed)
```

For the one-shot version, `intervention_information(nodes, design)["T"]` does
the same thing on a square gene set, optionally with a permutation null.

## 10. API

| name | what |
|---|---|
| `PerturbationDesign.from_obs` | build the design from `obs` columns; configurable target/direction column names and vocabularies, case-insensitive |
| `.summary()`, `.coverage()`, `.perturbed_genes()`, `.strata()` | inspect what the design actually contains before trusting any score |
| `intervention_node(design, gene)` | `(P_a, cell mask)` |
| `intervention_information(nodes, design)` | the full `T`, optionally with a permutation `z` |
| `subset_node(node, mask)` | restrict a node to cells, **keeping its global bins** |
| `dose_intervention_node(...)` | `P_a` for dense-dose designs, coded by dose sign |
| `compose_directed(detect, orient, soft=False)` | detect with one matrix, orient with another |
| `mi_decomposition`, `correlation_decomposition`, `check_covariance_identity` | split dependence into within- and between-condition parts |

A note on the decompositions: the measured result is that the `within` arm
carries **no** edge information beyond a shuffled-label null — conditioning on
perturbation identity removes the interventional signal. They are kept because
they are what makes that statement checkable, not because they are
recommended.
