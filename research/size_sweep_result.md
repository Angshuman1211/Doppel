# Size Sweep — memorization is a function of cohort size for diffusion, and a property of the model for TVAE (2026-10-22)

Post-course extension work (improvement plan item 1), and the result the project's central claim needed. Until now "capacity relative to cohort size drives memorization" was inferred from a single cohort of 94 training rows. This measures it across 50 to 5,000 patients on a second dataset.

```bash
python -m research.size_sweep --sizes 50 100 250 500 1000 2500 5000 --seeds 2 --workers 6   # 70 fits, 696 s
```

---

## Setup

UCI Diabetes 130-US hospitals, 101,766 encounters over 71,518 patients, 50 columns, fetched with `ucimlrepo` (id 296). The generators are unchanged: only the codec is new (`research/tabular_codec.py`), because `generators/codec.py` is MIMIC-specific. The Phase 1 interface took a second dataset with no change to any generator.

Three choices keep sizes comparable, and each is an assumption worth stating:

- **The codec is fitted once on the full dataset**, so the modeled schema is identical at every size. Fitting per subsample would shrink the frame at small n, confounding the effect being measured. The schema is therefore treated as public.
- **One fixed holdout of 2,000 patients** (2,863 encounters), disjoint from every training draw.
- **Every arm emits 1,500 synthetic rows at every size**, since the synthetic set size is a free parameter.

Categorical columns are capped at 50 levels with the remainder mapped to OTHER, which affects the three diagnosis columns (about a quarter of their values) and nothing else.

---

## Memorization against cohort size

Near-copy rate, mean of 2 seeds. Real unseen rows score about 0.05 by construction, and the selection guard from the course project is 0.10.

| Patients | Train rows | `tabular_diffusion` | `tvae` | `ctgan_small` | `gaussian_copula_shrink010` | `independent_marginals` |
|---|---|---|---|---|---|---|
| 50 | 67 | **0.723** | **0.699** | 0.013 | 0.036 | 0.017 |
| 100 | 140 | **0.334** | **0.685** | 0.019 | 0.020 | 0.010 |
| 250 | 362 | 0.069 | **0.415** | 0.010 | 0.008 | 0.007 |
| 500 | 739 | 0.014 | **0.597** | 0.024 | 0.011 | 0.008 |
| 1,000 | 1,446 | 0.007 | **0.570** | 0.008 | 0.009 | 0.008 |
| 2,500 | 3,603 | 0.005 | **0.480** | 0.004 | 0.009 | 0.004 |
| 5,000 | 7,145 | 0.004 | **0.374** | 0.002 | 0.007 | 0.003 |

**This is two different failure modes, not one.**

**Diffusion memorizes because the cohort is small, and stops when it isn't.** Its near-copy rate falls by two orders of magnitude, 0.723 to 0.004, and crosses the 0.10 guard between 250 and 500 patients. The MIMIC result is consistent with this curve: 94 training rows there produced 0.804, against 0.723 at 67 rows and 0.334 at 140 rows here. The two datasets differ in columns (110 modeled on MIMIC, about 50 here) and in what a near copy means, so this is agreement in direction and rough size, not a prediction the curve makes for MIMIC. It does make it unlikely that the course project's headline was a quirk of MIMIC or of our implementation.

**TVAE memorizes at every size.** It starts at 0.699 and is still at 0.374 with 7,145 training rows, having not crossed the guard at any point, and it does not decline monotonically. Its utility is also undefined at three of the seven sizes because mode collapse dropped the minority class from the synthetic data entirely. For TVAE the problem is the model, not the data budget, which is the opposite diagnosis to diffusion and was invisible at n = 94 where both simply looked like "memorizes".

## The structural fidelity findings hold at every size

Correlation difference, lower is better, against the no-dependence floor:

| Patients | `gaussian_copula_shrink010` | `ctgan_small` | `independent_marginals` (floor) |
|---|---|---|---|
| 50 | 0.063 | 0.166 | 0.162 |
| 500 | 0.043 | 0.117 | 0.117 |
| 5,000 | **0.033** | 0.097 | 0.100 |

**CTGAN never beats the floor**, at any size up to 7,145 rows: 0.097 against 0.100 at the largest. The course finding that it learns no relationships between columns is not a small-data artifact. The copula's advantage, by contrast, widens as data grows, from roughly 2.5× the floor's quality at 50 patients to 3× at 5,000.

One nuance in the other direction: diffusion's DCR ratio climbs past 1 at large n, reaching 1.35 at 5,000 patients, meaning its rows end up *further* from the training data than real unseen rows are. With the step count fixed at 3,000 while the data grows, the same configuration moves from memorizing to underfitting. The sweep varies the data, not the capacity, so "capacity relative to cohort size" is only half-measured here; varying both is the obvious next experiment.

## What this sweep cannot say

**Utility is uninformative on this dataset.** Train-on-real reaches only 0.562 to 0.602 AUROC for early readmission across these sizes, so the task is barely learnable from real data, and every synthetic arm lands in the same 0.45 to 0.59 band. The sweep speaks to fidelity and memorization; a second task is needed before it speaks to utility, which is item 6 of the improvement plan.

Other limits: two seeds per cell, one dataset, categorical columns capped at 50 levels, and the membership attacks were not rerun here, so "memorization" means the distance-based measure rather than an attack result. Given the course finding that the distance guard passed a configuration the attack then broke, the attack-based version of this curve is the one that matters for the method contribution, and it is the natural next step.
