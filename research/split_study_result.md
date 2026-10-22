# Repeated Split Study — half the utility spread is the split, and one reported gain does not survive it (2026-10-22)

Post-course extension work, outside the graded pipeline (item 2 of the improvement plan). The reported benchmark fixes one 80/20 patient-level split and varies only generator seeds, so every utility number carries an unmeasured dependence on which patients landed in the holdout. This redraws the split 10 times by the same procedure (ADR-001), refits all nine arms on each, and scores fidelity against that split's train and utility against that split's holdout.

```bash
python -m research.split_study --splits 10 --seeds 3 --workers 6   # 270 fits, 1,063 s on 6 workers
```

---

## Method

10 splits × 9 arms × 3 generator seeds. Each split samples 80% of the 100 `subject_id`s to train and lets admissions follow their patient, exactly as Track 3's preprocessing does, with a different seed. Metrics are Track 2's unchanged `run_fidelity_report` and `tstr_eval`, and generator output is round-tripped through a CSV read by `S.load_real` so dtypes match the benchmark (P-005). The membership attacks are not rerun here: they need 8 shadow fits each, which this grid cannot afford, so this study speaks to fidelity and utility only.

What varies between splits is both the training set and the evaluation set, which is the point: the question is whether a conclusion survives a different draw of patients, not whether one metric is noisy in isolation.

---

## The reported split is not typical

| | Holdout rows | Holdout deaths |
|---|---|---|
| Reported split | 35 | 6 |
| 10 redrawn splits | 22 to 40 | 5 to 10 |

Holdout size swings by nearly a factor of two because patients carry between 1 and 15 admissions each. The reported split sits near the bottom of the range for deaths, which is the denominator of every utility number in the project.

## Utility across splits, mean ± sd over 10 splits × 3 seeds

| Arm | TSTR RF | TSTR LR | Mean JSD | Corr. diff |
|---|---|---|---|---|
| `tvae` | 0.750 ± 0.111 | 0.633 ± 0.153 | 0.110 | 0.120 |
| `tabular_diffusion` | 0.744 ± 0.086 | 0.629 ± 0.118 | 0.037 | 0.159 |
| `tabular_diffusion_reg` | 0.719 ± 0.099 | 0.709 ± 0.097 | 0.367 | 0.171 |
| `gaussian_copula_shrink025` | 0.670 ± 0.142 | 0.591 ± 0.132 | 0.016 | 0.105 |
| `gaussian_copula` | 0.661 ± 0.160 | 0.611 ± 0.154 | 0.016 | 0.131 |
| `gaussian_copula_shrink010` | 0.661 ± 0.118 | 0.564 ± 0.129 | 0.016 | 0.098 |
| `independent_marginals` | 0.630 ± 0.147 | 0.523 ± 0.169 | 0.016 | 0.165 |
| `ctgan_small` | 0.584 ± 0.158 | 0.499 ± 0.195 | 0.070 | 0.172 |
| `ctgan` | 0.544 ± 0.151 | 0.496 ± 0.169 | 0.091 | 0.168 |

## About half the utility variance is the split, not the generator

Between-split standard deviation against between-seed standard deviation within a split, for TSTR RF:

| Arm | Split sd | Seed sd | Share of variance from the split |
|---|---|---|---|
| `tabular_diffusion` | 0.074 | 0.058 | 62% |
| `gaussian_copula` | 0.133 | 0.115 | 57% |
| `independent_marginals` | 0.121 | 0.106 | 57% |
| `tvae` | 0.090 | 0.083 | 54% |
| `gaussian_copula_shrink010` | 0.092 | 0.094 | 49% |
| `ctgan_small` | 0.083 | 0.164 | 20% |

The benchmark reports ± over seeds only, which by this measure captures roughly half of the variation that actually exists for most arms.

## Paired tests across splits

Same split, mean over three seeds, Wilcoxon signed-rank over the 10 paired differences.

| Comparison | Metric | Mean diff | Splits won | p |
|---|---|---|---|---|
| `gaussian_copula_shrink010` vs `gaussian_copula` | utility | **−0.000** | 4/10 | **1.00** |
| `gaussian_copula_shrink010` vs `gaussian_copula_shrink025` | utility | −0.009 | 4/10 | 0.63 |
| `ctgan_small` vs `ctgan` | utility | +0.040 | 6/10 | 0.084 |
| `tabular_diffusion` vs `gaussian_copula_shrink010` | utility | +0.083 | 9/10 | 0.020 |
| `tvae` vs `gaussian_copula_shrink010` | utility | +0.088 | 10/10 | 0.002 |
| `tabular_diffusion` vs `independent_marginals` | utility | +0.115 | 7/10 | 0.027 |
| `gaussian_copula_shrink010` vs `gaussian_copula` | correlation | −0.033 | **10/10** | **0.002** |
| `gaussian_copula_shrink010` vs `independent_marginals` | correlation | −0.067 | **10/10** | **0.002** |
| `ctgan_small` vs `independent_marginals` | correlation | +0.007 (worse) | **0/10** | **0.002** |

---

## What changes, and what does not

**One reported gain does not survive.** On the reported split the copula's utility rises as shrinkage falls, 0.599 to 0.711 for random forest, which reads as "keeping more correlation makes the data more useful". Across 10 redrawn splits that difference is **exactly zero** (−0.000, p = 1.00, winning 4 splits of 10). It was a property of that split, not of the method, and it should not be claimed in the report. The same caution applies to CTGAN's tuning gain (+0.040, p = 0.084), which is suggestive but not established.

**The structural findings hold, and get stronger.** The copula's correlation advantage is real and wins on **every** split, against both the default shrinkage and the floor. CTGAN is worse than the no-dependence floor on correlation in **all 10** splits, so "it learns no relationships between columns" is not a one-split accident. And the arms that copy their training rows, TVAE and the diffusion model, beat the tuned copula on utility in 9 and 10 splits respectively, which sharpens rather than softens the project's central point: on this data, utility rewards memorization.

**For the write-up.** Fidelity and correlation conclusions can be stated as findings. Utility comparisons should be stated either as paired results across splits or with the explicit caveat that one split cannot separate a 0.05 to 0.10 AUROC difference from the draw of patients. This is the evidence for the improvement plan's item 5: report statistics over patients, not over generator seeds.
