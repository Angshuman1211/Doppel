# Patients as the Unit of Uncertainty — no utility comparison survives, every fidelity comparison does (2026-10-23)

Post-course extension work (improvement plan item 5; §VII of the paper, "seeds vary the generators, not the patients, and p-values are uncorrected"). The reported benchmark gives mean ± sd over 20 generator seeds on one fixed 80/20 patient-level split, and compares nine arms on several metrics at an uncorrected 5%. This replaces both: patients become the resampled unit, and the family of comparisons is corrected with Holm-Bonferroni.

```bash
python -m research.stats_report --workers 6        # 180 fits, 882 s on 6 workers
python -m research.stats_report --from-cache       # re-aggregate without refitting
```

---

## Method

**Cluster bootstrap over patients.** Patients are drawn with replacement and their admissions follow them, the dependence structure the split procedure already respects (ADR-001). Generators are held fixed: each of the nine arms is fitted once per seed on the real train split, and the bootstrap moves only the cohort the metric is measured against — the 20-patient holdout for utility, the 80-patient train reference for fidelity. 2,000 holdout resamples and 500 train resamples, identical across every arm and seed, so every arm-vs-arm difference is a **paired** bootstrap with its own interval rather than two overlapping marginal ones.

Utility is the report's TSTR AUROC, the mean of logistic regression and random forest. The classifier is fitted once per arm and seed and the bootstrap resamples its predictions, so no resample refits a model. 0.15% of holdout resamples drew none of the 6 patients who carry the only deaths; those have no AUROC and are dropped.

**Intervals.** Reported as basic (pivotal) intervals, `2·estimate − bootstrap quantiles`. The fidelity metrics are two-sample divergences and the synthetic table was fitted on the original cohort, so its divergence from a *redrawn* cohort is systematically larger and the percentile interval can sit entirely above the number being reported. The basic interval keeps the bootstrap's spread while staying centred on the statistic the paper actually prints. For AUROC the two forms agree.

**Correction.** Holm-Bonferroni within each metric family, over a comparison family fixed in the code before any p-value was looked at, containing exactly the contrasts the report or the split study already claims: 6 utility, 4 correlation, 2 marginal-fidelity. The same family is then applied to the paired Wilcoxon tests across the 10 redrawn splits of [`split_study.py`](split_study.py), recomputed from its CSV rather than copied from its write-up.

---

## 1. The reported interval understates cohort uncertainty by up to 18×

Ratio of the patient bootstrap's standard deviation to the standard error the reported seed spread implies:

| Arm | Mean JSD | Corr. diff | TSTR |
|---|---|---|---|
| `independent_marginals` | **18.2** | 11.9 | 1.6 |
| `gaussian_copula` | 15.5 | 8.8 | 2.3 |
| `gaussian_copula_shrink025` | 15.5 | 7.1 | 1.7 |
| `gaussian_copula_shrink010` | 11.2 | 6.9 | 1.9 |
| `ctgan` | 1.7 | **14.6** | 1.1 |
| `ctgan_small` | 3.1 | 10.7 | 1.0 |
| `tvae` | 4.3 | 7.5 | 4.1 |
| `tabular_diffusion` | 1.9 | 1.0 | 4.1 |
| `tabular_diffusion_reg` | 1.4 | 2.4 | 4.0 |

The pattern is the uncomfortable one: the arms with the *tightest* seed intervals are the ones whose reported precision is most misleading. `independent_marginals` reports a mean JSD of 0.0162 ± 0.0007 over seeds, because sampling empirical marginals is a near-deterministic procedure — and that stability says nothing about the 80 patients those marginals came from, where the real spread is 18 times larger. A narrow seed interval measures the fitting procedure's repeatability, not the measurement's precision.

## 2. Five of nine arms cannot be distinguished from a redrawn cohort on marginal fidelity

The right yardstick for a divergence is not zero, it is the divergence a redrawn cohort of the same patients has from the real one. Resampling the 80 train patients and scoring them against the real train split, with no generator involved at all:

| Floor | Mean | 95% |
|---|---|---|
| Mean JSD | 0.0159 | [0.0109, 0.0219] |
| Corr. diff | 0.0843 | [0.0659, 0.1048] |

`p_floor` below is the share of redrawn cohorts at least as far from the real one as the arm is:

| Arm | Mean JSD | p_floor | Corr. diff | p_floor |
|---|---|---|---|---|
| `independent_marginals` | 0.0162 | **0.42** | 0.1629 | 0.000 |
| `gaussian_copula` | 0.0164 | **0.39** | 0.1377 | 0.000 |
| `gaussian_copula_shrink025` | 0.0164 | **0.39** | 0.1099 | 0.010 |
| `gaussian_copula_shrink010` | 0.0167 | **0.36** | 0.1021 | **0.044** |
| `ctgan` | 0.0931 | 0.000 | 0.1649 | 0.000 |
| `ctgan_small` | 0.0727 | 0.000 | 0.1688 | 0.000 |
| `tvae` | 0.1070 | 0.000 | 0.1231 | 0.002 |
| `tabular_diffusion` | 0.0422 | 0.000 | 0.1741 | 0.000 |
| `tabular_diffusion_reg` | 0.3767 | 0.000 | 0.1604 | 0.000 |

On marginals the copula family and the independent floor are **saturated**: their synthetic columns are no further from the real cohort than another draw of the same 80 patients would have been, so the metric cannot separate them and the differences in the fourth decimal of the reported table are not measurements. Every neural arm is far outside the floor, so the metric still discriminates where it matters.

On correlation only `gaussian_copula_shrink010` reaches the floor, and it does so borderline (p = 0.044, just inside a one-sided 2.5% reading). The honest claim is that the tuned copula is **at or very near the limit of what 80 patients can measure** on both fidelity axes — which is a stronger result than "0.1021 beats 0.1629", and it also means no further tuning of this family is measurable on this cohort.

This is also why the basic interval for mean JSD has a negative lower bound for those five arms: the interval's half-width exceeds the estimate. That is the signature of a statistic sitting on its floor, not a divergence below zero.

## 3. Utility: four of six comparisons survive the holdout bootstrap, none survive redrawing the split

| Comparison | Diff | Bootstrap 95% | Holm | Across 10 splits | Holm |
|---|---|---|---|---|---|
| `tabular_diffusion` − `gaussian_copula_shrink010` | +0.076 | [−0.001, 0.170] | 0.104 | +0.074 | 0.195 |
| `tabular_diffusion_reg` − `gaussian_copula_shrink010` | +0.097 | [0.030, 0.163] | **0.020** | +0.101 | 0.059 |
| `tvae` − `gaussian_copula_shrink010` | −0.001 | [−0.094, 0.084] | 0.880 | +0.079 | 0.098 |
| `gaussian_copula_shrink010` − `gaussian_copula` | +0.085 | [0.028, 0.150] | **0.020** | −0.024 | 0.316 |
| `gaussian_copula_shrink010` − `independent_marginals` | +0.191 | [0.101, 0.282] | **0.006** | +0.036 | 0.320 |
| `ctgan_small` − `ctgan` | +0.049 | [0.013, 0.116] | **0.015** | +0.021 | 0.625 |

**Not one utility comparison survives Holm across the 10 redrawn splits available when this was written.** On the paper's TSTR metric all six Holm-adjusted p-values exceed 0.05; the smallest is 0.059.

**That was a power problem, and item 7 settled it.** A signed-rank test on 10 pairs cannot return a p below 2/2¹⁰ = 0.002, so the Holm floor for a six-test family is 0.012, and the observed utility p-values (0.0098 to 0.625) sat at or above the correction threshold rather than far above it. Scaling to 30 splits, **four of the six comparisons do survive** — see [`scale_study_result.md`](scale_study_result.md), which supersedes this section. The neural arms beat the tuned copula and the tuned copula beats the no-dependence floor, at adjusted p ≤ 0.010; the copula shrinkage gain and the CTGAN tuning gain remain null. The column below is kept because the contrast between it and the 30-split table is the lesson: "not significant" at a feasible number of resamples is not the same as "not there".

Everything else in this document is independent of the split count and stands: the interval understatement in §1, the measurement floor in §2, and the warning about test-set bootstraps in §4.

By contrast, all four correlation and both marginal-fidelity comparisons survive **both** procedures, with Holm-adjusted p ≤ 0.016 in the bootstrap and ≤ 0.008 across splits. The ordering of these generators on fidelity is a finding; their ordering on utility is not.

## 4. Bootstrapping the test set is not enough, and can be actively misleading

Both procedures in §3 resample patients, and they disagree. The holdout bootstrap holds the *training* patients fixed and redraws only the cohort the model is tested on; the split study redraws both. For two of the six comparisons the bootstrap's 95% interval **excludes** the across-split mean:

| Comparison | Bootstrap 95% | Across-split mean |
|---|---|---|
| `gaussian_copula_shrink010` − `gaussian_copula` | [0.028, 0.150] | **−0.024** |
| `gaussian_copula_shrink010` − `independent_marginals` | [0.101, 0.282] | **+0.036** |

Both are the copula-shrinkage utility claims the split study already killed. A test-set bootstrap — the standard, cheap move, and the one that needs no refitting — would have made both look solid and significant. The reason is that on a cohort this small nearly all of the utility uncertainty comes from **which patients the generator was fitted on**, not from which patients it is tested on, and a bootstrap that leaves the training set fixed cannot see it. Two of the fidelity comparisons also fall outside, but there direction and significance agree under both procedures, so no conclusion turns on it.

The practical rule: on small cohorts, resample the split, not the test set. If refitting K times is unaffordable, a test-set bootstrap should be reported as a lower bound on uncertainty and never as a confidence interval for the comparison.

## 5. A side effect that confirms a known problem

The bootstrap mean of TSTR sits 0.021 to 0.027 above the reported estimate for every arm whose ranking is better than chance, and at zero for the two CTGAN arms. Subject 41976 contributes 15 of the 35 holdout rows and none of its admissions ended in death; 36% of patient resamples omit that patient entirely, and each of those gains roughly the +0.08 that dropping the patient is already known to produce (ADR-032). The shift is therefore understood rather than mysterious, and it is further evidence for scoring the holdout-side metrics on the clean holdout.

---

## What this changes for the paper

- §VII's "seeds vary the generators, not the patients, and p-values are uncorrected" can be replaced by the correction itself.
- **Utility comparisons must be stated as paired results over splits with the correction attached**, never as a ± over seeds on one split. Which ones are claimable is settled in [`scale_study_result.md`](scale_study_result.md): four of six at 30 splits, not the zero of six this document found at 10.
- The fidelity and correlation orderings can be stated as findings, with Holm-corrected p-values, and should be the paper's quantitative backbone.
- Add the measurement floor. "The tuned copula has reached the limit of what this cohort can measure" is a better result than any of the margins currently reported, and it explains why the copula family's marginal-fidelity differences are not worth discussing.
- Report ± over patients, not over seeds, or report both and say which is which.

## Limits

One dataset and one fixed split for the bootstrap, 20 seeds, and 10 redrawn splits for the Wilcoxon tests, which bounds any p-value there at 0.002 and any Holm-adjusted p at 0.0039. The membership-attack AUROCs are not covered: each bootstrap replicate would need the 8 shadow fits refitted, so privacy still carries seed-only uncertainty. The resampling floor compares a synthetic table of exactly 94 rows against redrawn cohorts whose row count varies with the patients drawn, averaging 94.
