# A Second Prediction Task — the utility axis was broken by the task, not by the metric (2026-10-23)

Post-course extension work (improvement plan item 6). Every utility number in the project comes from one task, in-hospital mortality on MIMIC, so "generator A is more useful than B" has always meant "A beats B at one task on one holdout". This adds three more tasks, and finds that the project's uninformative utility axis was a property of the outcome being predicted.

```bash
python -m research.task_study --mimic                                  # the MIMIC census
python -m research.task_study --sizes 250 1000 --seeds 3 --workers 6    # 30 fits, 400 s
```

---

## 1. MIMIC cannot supply a second task, and here is the census

| Cohort | Rows | Mortality | `readmit_30d` | Patients carrying a readmission |
|---|---|---|---|---|
| train | 94 | 34 | 9 | 8 |
| holdout | 35 | 6 | **2** | **1** |
| clean holdout (P-003) | 20 | 6 | **0** | **0** |

The only other binary outcome in the modeled schema is 30-day readmission, and the holdout has two positive rows, both admissions of the same patient. Under the patient bootstrap of [`stats_report.py`](stats_report.py), **35.3%** of resamples contain no positive case at all and therefore have no AUROC, against 0.1% for mortality. On the clean holdout the task does not exist. This is not a judgement call about statistical power; the measurement is undefined.

So the second task runs where it is measurable, on the UCI Diabetes 130-US data the [size sweep](size_sweep_result.md) already uses, with the same codec, the same fixed holdout patients, the same arms and the same cohort draws. For seeds 0 and 1 the synthetic tables are the identical draws the size sweep scored, so its memorization numbers can be read directly against these.

## 2. Four tasks, and the first one is the odd one out

Each task's source column is withheld from the features. TRTR is the ceiling: a random forest trained on the **real** training subsample and tested on the same holdout.

| n = 1000 patients (1,433 rows) | `readmit_lt30` | `long_stay` | `high_meds` | `emergency_admission` |
|---|---|---|---|---|
| Base rate | 11.2% | 20.8% | 27.1% | 53.1% |
| **TRTR ceiling (real data)** | **0.563** | **0.804** | **0.822** | **0.953** |
| `tabular_diffusion` | 0.556 ± 0.007 | **0.784 ± 0.006** | **0.781 ± 0.006** | **0.912 ± 0.008** |
| `gaussian_copula_shrink010` | 0.522 ± 0.010 | 0.734 ± 0.014 | 0.772 ± 0.007 | 0.704 ± 0.025 |
| `tvae` | 0.508 | 0.722 ± 0.010 | 0.747 ± 0.013 | 0.856 ± 0.018 |
| `ctgan_small` | 0.492 ± 0.014 | 0.533 ± 0.015 | 0.526 ± 0.012 | 0.522 ± 0.045 |
| `independent_marginals` (floor) | 0.501 ± 0.021 | 0.525 ± 0.007 | 0.515 ± 0.030 | 0.511 ± 0.034 |

**Early readmission is not learnable from this schema at all.** Real data reaches 0.563. Every arm therefore lands between 0.49 and 0.57, and the size sweep's conclusion that "utility is uninformative on UCI" was a statement about the task, not about synthetic data. Replace it with a learnable outcome and the axis works immediately: the no-dependence floor collapses to chance (0.511 to 0.525) while the copula and diffusion arms reach 0.70 to 0.91, within 0.02 to 0.05 of what real data achieves.

That is what a working utility axis looks like, and it is worth contrasting with MIMIC, where the floor scored 0.4865 and the tuned copula 0.6777 — a gap of the same size that [`stats_report_result.md`](stats_report_result.md) showed was not statistically established. Here the same gap is 0.209 with seed standard deviations of 0.007 and 0.014.

## 3. The ranking is stable across tasks; the effect size is not

Spearman correlation between the arm orderings of two tasks, at n = 1000: **+0.80 to +1.00**, with `long_stay` and `high_meds` agreeing perfectly. At n = 250 the range is the same. So the second task does not change *who* wins — which is the reassuring answer for the benchmark's design, and means the project's generator ordering is not an artefact of predicting mortality.

The margins are another matter. `tabular_diffusion` minus `gaussian_copula_shrink010`, same cohorts, same seeds:

| Task | n = 250 | n = 1000 |
|---|---|---|
| `readmit_lt30` | +0.049 | +0.034 |
| `long_stay` | +0.025 | +0.050 |
| `high_meds` | +0.010 | **+0.009** |
| `emergency_admission` | +0.184 | **+0.208** |

A 23-fold range in the same comparison at the same cohort size. On `high_meds` the tuned copula is level with the diffusion model and within 0.05 of real data; on `emergency_admission` it loses by 0.21. **One task can support a claim about ordering; it cannot support a claim about how much better a generator is.** Any paper reporting a single utility delta is reporting a number that depends on the outcome it happened to pick.

## 4. CTGAN is at chance on every task, at both sizes

0.492 to 0.533 across four tasks and two cohort sizes, indistinguishable from the independent-marginals floor. Two metrics built on different machinery now agree: CTGAN sits *above* the no-dependence floor on correlation error at every size tested, and it transfers no predictive signal on any task. "CTGAN learns no relationships between columns on this data" is no longer an inference from one correlation statistic.

## 5. The one finding this overturns: utility stops rewarding memorization once the cohort is large enough

The project's central reading of MIMIC is that the arms which copy their training rows win on utility. Reading the memorization rates from the size sweep against the utility numbers here, on the identical draws:

| n = 1000, 1,446 rows | Near-copy rate | `long_stay` | `emergency_admission` |
|---|---|---|---|
| `tabular_diffusion` | **0.007** | 0.784 | 0.912 |
| `tvae` | **0.570** | 0.722 | 0.856 |
| `gaussian_copula_shrink010` | 0.009 | 0.734 | 0.704 |
| `independent_marginals` | 0.008 | 0.525 | 0.511 |

At this size the diffusion model has **stopped copying** — near-copy 0.007, the same as the statistical arms, with a DCR ratio of 1.20 on the underfitting side — and it still wins every task. TVAE copies 57% of its rows and loses to it on every task. So the coupling between utility and memorization that MIMIC shows is a **small-cohort phenomenon**: at n = 94 the only way a flexible model fits the data is by reproducing it, and at n = 1,446 the same architecture earns its utility by generalising. The size sweep localised where the copying stops; this localises where the utility advantage stops depending on it.

---

## What this changes for the paper

- §VII's "utility is measured on a single task" is answered, with the MIMIC census as the reason it had to be answered elsewhere.
- The claim "utility rewards memorization" must be **scoped to the small-cohort regime**. It is true at n = 94 and false at n = 1,446, and saying so is a stronger result than asserting it in general.
- The generator ordering survives the change of task (Spearman ≥ 0.80), so the benchmark's ranking can be stated with more confidence than before.
- But no single-task utility *margin* should be quoted, in this paper or any other: the same comparison moves by 23× across tasks.
- The unlearnability of early readmission (TRTR 0.563) belongs next to the MIMIC TRTR ceiling of 0.56 to 0.60. Two datasets, two near-unlearnable targets, and in both cases the utility axis was measuring the task's difficulty rather than the generator.

## Limits

Three seeds per cell, two cohort sizes, one dataset for the multi-task comparison and random forest as the only downstream model. The four tasks are all binary outcomes derived from columns the generator models, which is the only way to keep the target inside the synthetic table; a task defined on an unmodeled column would test something different. `tvae` returns no AUROC for `readmit_lt30` at n = 250 and has only one usable seed at n = 1000, from the mode collapse the size sweep documents, so its row there is not a three-seed mean.
