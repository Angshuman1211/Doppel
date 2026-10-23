# Scaling the Split Count — the utility nulls were a power problem, and four of six comparisons survive at 30 splits (2026-10-24)

Post-course extension work (improvement plan item 7). The plan asked for a Kubernetes scaled sweep. My objection to that stands and is unchanged: the full grid is roughly 30,000 fits, about 80 compute-hours, so the binding constraint is **machines, not orchestration**, and running Minikube on a laptop changes nothing about it. So what got scaled is the one axis where more compute decides a question rather than adding digits: **the number of redrawn patient-level splits, 10 → 30.**

The reason is specific. [`stats_report_result.md`](stats_report_result.md) reported that no utility comparison survived Holm correction across 10 splits, and flagged that this was power-limited rather than a demonstrated absence: a signed-rank test on 10 pairs cannot return a p below 2/2¹⁰ = 0.002, which puts the Holm floor for a six-test family at 0.012, and the observed p-values sat at 0.0098 to 0.625. That flag turns out to have been the right one.

```bash
python -m research.split_study --splits 30 --seeds 3 --workers 6   # 810 fits, 2,087 s
python -m research.stats_report --from-cache                        # re-runs the tests, 14 s
```

Split seeds are `1000 + k`, so the original 10 splits are a subset of the 30 and the earlier numbers stay reproducible from `research/output/split_study_10splits.csv`. 2,087 s on one laptop, about 3.5 CPU-hours, or 2.6% of the full grid — for the question the full grid was meant to answer.

---

## 1. Four of six utility comparisons now survive

Paired Wilcoxon over splits, mean of the two classifiers (the report's TSTR metric), Holm-corrected within the six-test family:

| Comparison | 10 splits | | 30 splits | | |
|---|---|---|---|---|---|
| | Diff | Holm | Diff | Wins | Holm |
| `tabular_diffusion` − `gaussian_copula_shrink010` | +0.074 | 0.195 | +0.061 | 24/30 | **0.0009** |
| `tabular_diffusion_reg` − `gaussian_copula_shrink010` | +0.101 | 0.059 | +0.067 | 23/30 | **0.0018** |
| `tvae` − `gaussian_copula_shrink010` | +0.079 | 0.098 | +0.041 | 22/30 | **0.0104** |
| `gaussian_copula_shrink010` − `independent_marginals` | +0.036 | 0.320 | +0.070 | 24/30 | **0.0009** |
| `gaussian_copula_shrink010` − `gaussian_copula` | −0.024 | 0.316 | +0.012 | 17/30 | 0.329 |
| `ctgan_small` − `ctgan` | +0.021 | 0.625 | +0.023 | 19/30 | 0.210 |

**This supersedes §3 of [`stats_report_result.md`](stats_report_result.md).** "No utility comparison is established" was true at 10 splits and is false at 30. The correct statement is that the neural arms genuinely beat the tuned copula on utility, and the tuned copula genuinely beats the no-dependence floor.

Note how the point estimates moved as splits were added: the three neural margins all **shrank** (+0.101 → +0.067, +0.079 → +0.041, +0.074 → +0.061) while the copula-over-floor margin nearly **doubled** (+0.036 → +0.070). Ten splits was noisy in both directions, which is exactly why neither the original single-split numbers nor the 10-split corrections should be quoted as effect sizes.

## 2. The two claims that stay dead

`gaussian_copula_shrink010` − `gaussian_copula` reaches 17 wins of 30 and p = 0.329. **The killed shrinkage claim stays killed**, now on three times the evidence, and it is the cleanest result in this table: a difference that looked like +0.11 on the reported split is a coin flip across 30 draws of patients. The CTGAN tuning gain also fails on the report's metric (p = 0.210).

## 3. Two comparisons depend on which classifier is reported

Running the same family on random forest alone, which is the column [`split_study_result.md`](split_study_result.md) used:

| Comparison | Holm, mean of two | Holm, random forest only |
|---|---|---|
| `tabular_diffusion` − `gaussian_copula_shrink010` | **0.0009** | **0.0200** |
| `tvae` − `gaussian_copula_shrink010` | **0.0104** | **0.0318** |
| `gaussian_copula_shrink010` − `independent_marginals` | **0.0009** | **0.0004** |
| `tabular_diffusion_reg` − `gaussian_copula_shrink010` | **0.0018** | 0.188 |
| `ctgan_small` − `ctgan` | 0.210 | **0.0016** |
| `gaussian_copula_shrink010` − `gaussian_copula` | 0.329 | 0.188 |

Three comparisons survive under both classifiers and can be claimed without qualification. Two flip, in opposite directions: `tabular_diffusion_reg`'s advantage exists only when logistic regression is averaged in, and CTGAN's tuning gain exists only on random forest. **Any claim resting on those two has to name its downstream model**, and the paper should report both columns rather than pick one. This is the same lesson item 6 reached from the other side — there the margin depended on the prediction task, here on the classifier.

## 4. Fidelity and correlation are now beyond argument

All six comparisons win every one of 30 splits (or lose every one, for the arms that are worse), giving p = 2/2³⁰ ≈ 1.9 × 10⁻⁹ and Holm-adjusted p below 10⁻⁸:

| Comparison | Mean diff | Splits won |
|---|---|---|
| `gaussian_copula_shrink010` − `gaussian_copula` (corr.) | −0.0320 | 30/30 |
| `gaussian_copula_shrink010` − `independent_marginals` (corr.) | −0.0669 | 30/30 |
| `ctgan_small` − `independent_marginals` (corr., worse) | +0.0070 | 30/30 |
| `tvae` − `independent_marginals` (corr.) | −0.0447 | 30/30 |
| `gaussian_copula_shrink010` − `tabular_diffusion` (JSD) | −0.0224 | 30/30 |
| `tabular_diffusion` − `tvae` (JSD) | −0.0708 | 30/30 |

CTGAN being worse than the no-dependence floor on correlation in **30 of 30** splits closes that question for good.

## 5. Everything else held

| | 10 splits | 30 splits |
|---|---|---|
| Holdout size | 22–40 rows | 21–40 rows |
| Holdout deaths | 5–10 | 5–14 |
| Split share of TSTR variance | 20–62% | 21–61% |
| Degenerate label draws | 0 | 0 |

The variance decomposition is stable, so the item-2 finding that roughly half the utility spread is the split rather than the generator needs no revision. The widened death range (5–14 against the reported split's 6) makes the "reported split is not typical" point slightly stronger than before.

## 6. A harness bug worth recording

The first attempt at this run crashed after about 380 fits. `split_study.py` defaulted its arm list to `sorted(ARMS)`, and `ARMS` had silently grown from 9 to 13 when item 4 added the differentially private arms — so the run included DP arms, which produce a single-class mortality column on 20–45% of draws, and `tstr_eval` raised sklearn's "needs samples of at least 2 classes". This is precisely the failure the `REPORTED_ARMS` list was introduced to prevent, in a script written before that list existed. Three changes went in: the default is now `REPORTED_ARMS`, a one-class synthetic label column is recorded as `degenerate_label` with NaN utility instead of killing the grid, and the CSV is written every 20 fits. None of the nine reported arms produced a degenerate draw in 810 fits, so the crash was entirely the DP arms.

---

## What this changes for the paper

- **The utility ordering can be claimed after all**, as a paired result over 30 patient-level splits with Holm correction: the neural arms beat the tuned copula (adjusted p ≤ 0.010) and the tuned copula beats the no-dependence floor (adjusted p = 0.0009). State the number of splits and the correction, not a ± over seeds.
- **Do not claim** the copula shrinkage gain (p = 0.329) or, on the report's metric, the CTGAN tuning gain (p = 0.210).
- **"Utility rewards memorization" now has proper support at n = 94**: both copying arms, `tabular_diffusion` and `tvae`, beat the tuned copula with corrected p ≤ 0.010. Read together with item 6, where the coupling vanishes at n = 1,446, the paper gets a scoped, well-evidenced claim instead of a general, weakly-evidenced one.
- Replace §3 of [`stats_report_result.md`](stats_report_result.md) with this table, and the paired-test table in [`split_study_result.md`](split_study_result.md) with this one. The patient bootstrap conclusions in items 1 to 5 of that document — the 18× interval understatement, the measurement floor, the warning about test-set bootstraps — are unaffected, because none of them depend on the split count.

## Limits

30 splits of a 100-patient cohort, 3 generator seeds per split, one dataset. The membership attacks are still not in this grid: they need 8 shadow fits each, which would multiply 810 fits by nine. The splits are not independent — they are redraws from the same 100 patients, so a Wilcoxon over them tests whether a difference survives reshuffling *this* cohort, not whether it generalises to another hospital.
