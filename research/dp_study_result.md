# Differentially Private Baseline — the guarantee costs 20x fidelity and buys no measurable attack resistance (2026-10-22)

Post-course extension work (improvement plan item 4). The reported benchmark has no arm with a formal guarantee, so every privacy statement in it is attack-relative: it says an attack failed, not that leakage is bounded. This adds two DP mechanisms at two budgets each, on the same codec and metrics as the benchmark.

```bash
python -m research.dp_study --seeds 20 --workers 6     # 120 runs, 9 s
```

---

## The mechanism

`generators/dp_copula.py`, under bounded differential privacy (replacing one row moves a histogram by at most 2 in L1):

- **`dp_marginals`**: every column sampled from its own Laplace-noised histogram, no dependence between columns.
- **`dp_copula`**: the same marginals joined by a correlation matrix measured under the remaining budget. The data is mapped to latent normals, clipped to ±3, the covariance gets Laplace noise with sensitivity 2·3²/n per entry, and the result is projected back to the nearest positive-definite correlation matrix. Budget split 70/30 between marginals and correlation.

Composition is sequential across columns, so the per-column budget is `ε / 110` (or `0.7ε / 110` for the copula): **0.0064 to 0.073** for the budgets tested. Column ranges and levels are treated as public, taken from the schema rather than measured under the budget, which is the usual practical assumption for tabular DP and is an assumption rather than a theorem.

## Results, 20 seeds per arm

| Arm | Mean JSD ↓ | Corr. diff ↓ | TSTR RF ↑ | Gower MIA ↓ | Runs with an unusable label |
|---|---|---|---|---|---|
| `dp_marginals` ε=1 | 0.390 | 0.161 | 0.559 | 0.535 | **45%** |
| `dp_marginals` ε=8 | 0.286 | 0.163 | 0.536 | 0.533 | **20%** |
| `dp_copula` ε=1 | 0.395 | 0.162 | 0.518 | 0.546 | **40%** |
| `dp_copula` ε=8 | 0.310 | 0.145 | 0.522 | 0.558 | **25%** |
| `gaussian_copula_shrink010` (no DP) | **0.017** | **0.102** | **0.696** | 0.628 | 0% |
| `independent_marginals` (no DP) | **0.016** | 0.163 | 0.481 | 0.553 | 0% |

Utility is averaged over the runs where it is defined. Real train death rate is 0.362, and the DP arms reproduce it on average (0.319 to 0.357), so the damage is in the spread of each draw, not in the central tendency.

---

## Three findings

**1. A formal guarantee costs about 20× in fidelity here.** Mean JSD goes from 0.016 without DP to 0.29 to 0.40 with it, at every budget tested. The cause is structural rather than a tuning failure: 110 modeled columns under sequential composition leave each column 0.006 to 0.073 of budget, against 94 rows of signal.

**2. Between a fifth and nearly half of DP runs produce data that cannot be used for the downstream task at all.** The noisy mortality histogram collapses to a single class, so no classifier can be trained. That is the most concrete way to state the cost: at ε = 1, **45%** of `dp_marginals` draws are unusable for the project's own utility task. Reporting only the mean AUROC over surviving runs would hide this entirely.

**3. The attack cannot tell the DP arms from a non-private baseline that simply ignores dependence.** The Gower membership attack scores DP at 0.533 to 0.558 and non-private `independent_marginals` at 0.553, with the non-private arm having **24× better fidelity** (0.016 against 0.39). So on this dataset, at these budgets, the empirical privacy gain over an independent-marginals baseline is not measurable, while the fidelity loss is large and obvious.

Finding 3 is the one that matters for the paper, and it cuts both ways. It does not say DP is worthless: the guarantee is a worst-case bound over all attacks, including ones nobody has written, and an attack-relative result cannot deliver that. What it says is that **our attack suite cannot see the difference**, which is a statement about the attacks as much as about the mechanism, and it is precisely why a benchmark should carry a DP arm.

## Limits of this baseline

- This is a Laplace-histogram mechanism, not PrivBayes or DP-CTGAN. A mechanism that spends its budget on a selected subset of marginals rather than all 110 columns would do better, and the honest claim is "a straightforward DP mechanism at this width and sample size costs everything", not "DP is useless here".
- Only the Gower membership attack was run against the DP arms, not the full three-attack suite with calibration.
- ε = 1 and ε = 8 only; no accounting across the two mechanisms, and no δ, since every step is pure ε.
- Ranges and levels are treated as public, as stated above.
