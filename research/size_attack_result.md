# Membership Attack Across Cohort Size — the attack confirms the diffusion curve, and separates copied rows from exposed patients for TVAE (2026-10-25)

Post-course extension work. The [size sweep](size_sweep_result.md) measured memorisation on UCI Diabetes 130-US with the near-copy rate, a distance statistic of the kind the paper argues cannot replace an attack. A JBI-style review of the journal draft flagged this as the most serious gap: the cohort-size claim rested on the very tool the paper criticises. This runs the benchmark's shadow-model attack on the same cohorts instead.

```bash
python -m research.size_attack --sizes 50 250 1000 --seeds 2 --workers 3 --out size_attack_small   # 192 fits, 717 s
python -m research.size_attack --sizes 5000 --seeds 2 --workers 3 --out size_attack_5000           # 64 fits, 1,117 s
```

The two outputs are concatenated into `research/output/size_attack.csv`. The first attempt at 6 workers ran out of memory on Windows because every shadow fit re-read the 100k-row file; the script now caches the dataset once per worker process.

---

## Method

The design of `evaluation/membership_inference.py` on MIMIC-III, on the size sweep's cohorts (same fixed holdout, same patient draws for seeds 0 and 1). For each cohort, 8 shadow generators are each fitted on a random half of the cohort's rows and release as many rows as they were fitted on. Every cohort row is described by minus its Gower distance to the nearest released row, and a logistic-regression attack is trained on 7 shadows and scored on the 8th, in turn. Members and non-members both come from the cohort, so there is no distribution shift between them. Reported: mean AUROC over the held-out shadows, and the pooled TPR at 1% FPR (the highest TPR at an FPR no higher than 1%).

Four settings (the floor, the tuned copula, TVAE, the 3000-step diffusion model), four sizes, two cohort draws each, Gower distance only.

## Results

Shadow attack AUROC (mean of two cohort draws; ± is their standard deviation) beside the size sweep's near-copy rate for the same cohorts:

| Patients (training rows) | Floor | Copula λ=0.10 | TVAE | Diffusion |
|---|---|---|---|---|
| 50 (67) | 0.574 ± 0.016 | 0.640 ± 0.021 | 0.667 ± 0.015 | **0.850 ± 0.028** |
| 250 (362) | 0.526 ± 0.007 | 0.538 ± 0.007 | 0.540 ± 0.011 | **0.712 ± 0.006** |
| 1,000 (1,446) | 0.505 ± 0.003 | 0.512 ± 0.003 | 0.514 ± 0.005 | **0.576 ± 0.000** |
| 5,000 (7,145) | 0.500 ± 0.000 | 0.502 ± 0.002 | 0.509 ± 0.002 | **0.516 ± 0.004** |

| Patients (training rows) | Diffusion TPR at 1% FPR | Diffusion near-copy | TVAE near-copy |
|---|---|---|---|
| 50 (67) | **0.503** | 0.723 | 0.699 |
| 250 (362) | 0.102 | 0.069 | 0.415 |
| 1,000 (1,446) | 0.024 | 0.007 | 0.570 |
| 5,000 (7,145) | 0.011 | 0.004 | 0.374 |

## What it shows

**1. The diffusion model's exposure falls with cohort size, measured by an attack.** Its AUROC falls from 0.850 at 67 training rows to 0.712, 0.576 and 0.516 at 7,145, and the share of training records it exposes at 1% FPR from 50.3% to 1.1%. Its excess over the floor is 0.276, 0.186, 0.071 and 0.016. The attack and the near-copy rate agree for this model, so the cohort-size claim no longer rests on a distance statistic alone.

**2. For TVAE the two measures disagree, and the attack is the one to believe.** TVAE's near-copy rate stays between 0.37 and 0.70 at every size, but its attack AUROC falls from 0.667 to 0.540, 0.514 and 0.509, close to the floor from 362 rows on (excess 0.093, 0.014, 0.009, 0.009). The likely explanation is mode collapse: a high share of TVAE's synthetic rows sit close to some training row, but on a few training rows, so most patients are not exposed. This is consistent with two other observations. On MIMIC-III, TVAE's synthetic rows have only 31 of the 94 training admissions as nearest neighbours, with up to 24 synthetic rows on one admission; and in the size sweep TVAE's output lost the target's minority class entirely at three of seven sizes. The coverage measurement below confirms it. Either way, the near-copy rate measures how many synthetic rows look copied, not how many patients are exposed, and the two can diverge.

**3. Everything approaches chance by 5,000 patients.** At 7,145 training rows no setting is more than 0.016 AUROC above the floor. The floor itself goes from 0.574 to 0.500: at small sizes even independent sampling leaks a little through the empirical marginals, a version of the code-set floor seen on MIMIC-III.

## Coverage: the mode-collapse explanation, measured (added 2026-10-25)

`research/uci_coverage.py` fits each setting on the full cohort, releases as many rows as it was fitted on, and assigns every synthetic row its nearest training row by Gower distance (two cohort draws per size, 32 fits, 570 s):

```bash
python -m research.uci_coverage --sizes 50 250 1000 5000 --seeds 2 --workers 3
```

| Training rows | Share of training rows reached: Independent | Copula λ=0.10 | Diffusion | **TVAE** | TVAE: synthetic rows on the top 10% of training rows | TVAE: most synthetic rows on one training row |
|---|---|---|---|---|---|---|
| 67 | 0.485 | 0.531 | 0.611 | **0.228** | 0.82 | 19 |
| 362 | 0.478 | 0.500 | 0.592 | **0.111** | 0.98 | 83 |
| 1,446 | 0.479 | 0.484 | 0.575 | **0.103** | 0.99 | 201 |
| 7,145 | 0.469 | 0.476 | 0.568 | **0.182** | 0.91 | 193 |

TVAE's synthetic rows reach only 10 to 23% of the training rows, against 47 to 61% for the other settings, and from 362 rows on 91 to 99% of them sit on the 10% most-hit training rows, up to 201 on a single row. That is mode collapse, and it explains the disagreement above: a high share of TVAE's synthetic rows lie close to some training row, but on so few training rows that most records are not exposed. The diffusion model reaches the most training rows at every size with the least concentration, consistent with its exposure being spread over many patients on MIMIC-III.

## What this changes for the paper

- The cohort-size result can now be stated with an attack: the 3000-step diffusion model's AUROC falls from 0.850 to 0.516 as the cohort grows from 67 to 7,145 training rows.
- "TVAE copies at every size" must be replaced: TVAE produces near-copies at every size, but its membership exposure approaches the floor from a few hundred rows. "Diffusion's memorisation is a cohort-size effect; TVAE's is a model property" (the size sweep's reading) holds for the near-copy rate and not for the attack.
- The disagreement for TVAE is a second, independent example of the paper's point that a distance statistic is not a substitute for an attack.

## Limits

Two cohort draws per size, four settings, the Gower distance only, and no copy or unseen-record controls on this dataset. The cohorts are the size sweep's draws, so the same caveats about the UCI schema apply. TPR at 5% FPR is below 0.05 for some settings at 5,000 patients because many attack scores are tied; it is not reported for that reason. Coverage uses one release per cohort, of the same size as the cohort.
