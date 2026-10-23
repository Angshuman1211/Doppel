# Density-Based Attack — DOMIAS is weaker than distance here, and a simpler attacker beats the shadow pipeline (2026-10-22)

Post-course extension work (improvement plan item 3; limitations 3 and 4 of the membership survey; §VII of the paper, "no density-based or release-only attacker was run"). 6 arms × 10 seeds, 96 s.

```bash
python -m research.domias_attack --seeds 10 --workers 6
```

---

## Setup

The generator is fitted on a random 47 of the 94 training admissions, so the other 47 are non-members **from the same distribution** and the holdout's distribution shift never enters. The attacker sees only the released synthetic table (500 rows) and a reference sample, which is the clean holdout with subject 41976 removed. No shadow generators, no retraining, no generator access.

Two scores are computed on the identical split, so they are directly comparable:

- **DOMIAS** (van Breugel et al., 2023): `log p_synth(x) − log p_ref(x)`, both densities Gaussian KDE (Scott's rule) on a 5-component PCA projection fitted only on attacker-visible data.
- **Distance**: minus the Gower distance to the nearest synthetic row, the same family the benchmark uses.

With 47 non-members the smallest non-zero false-positive rate is about 2%, so TPR is reported at 5% rather than 1%.

## Results, mean over 10 seeds

| Arm | DOMIAS AUROC | DOMIAS TPR@5% | Distance AUROC | Distance TPR@5% | Difference |
|---|---|---|---|---|---|
| `tabular_diffusion` | 0.636 ± 0.070 | 0.140 | **0.976 ± 0.017** | **0.875** | −0.339 |
| `tabular_diffusion_reg` | 0.625 ± 0.054 | 0.128 | 0.748 ± 0.028 | 0.392 | −0.123 |
| `gaussian_copula_shrink010` | 0.576 ± 0.044 | 0.119 | 0.678 ± 0.038 | 0.168 | −0.103 |
| `independent_marginals` | 0.556 ± 0.068 | 0.087 | 0.601 ± 0.044 | 0.132 | −0.045 |
| `ctgan_small` | 0.522 ± 0.075 | 0.030 | 0.594 ± 0.041 | 0.089 | −0.072 |
| `tvae` | **0.519 ± 0.062** | 0.081 | 0.749 ± 0.025 | 0.277 | −0.230 |

---

## 1. The density attack is weaker on every arm, and it clears a known copier

DOMIAS loses to the distance attack on all six settings, by 0.045 to 0.339 AUROC. The mechanism is the one the membership survey predicted before it was run: the reference density is estimated from 20 records in a 5-dimensional projection, which is not enough for the ratio to mean much. This is a negative result about **this cohort size**, not about DOMIAS, which was designed for settings with a usable reference sample.

The more useful part is which arm it misses. **TVAE, the known copier, scores 0.519 under DOMIAS, essentially chance** — the lowest of all six arms, below even the independent floor. A density-ratio attack therefore joins the 4-column attack as a second attack that would have cleared a generator we know copies 52 to 71% of its rows. That is direct evidence for the paper's existing rule of reporting the worst of several attacks rather than any single one, and it is worth more than a positive DOMIAS result would have been.

## 2. A simpler attacker beats the benchmark's shadow pipeline on every arm

Comparing the direct release-only distance attack against the shadow-model numbers in Table I of the paper, on the same 47/47 member split:

| Arm | Shadow-model Gower AUROC (paper) | Direct release-only AUROC (here) |
|---|---|---|
| `tabular_diffusion` | 0.787 | **0.976** |
| `tvae` | 0.649 | **0.749** |
| `tabular_diffusion_reg` | 0.704 | 0.748 |
| `gaussian_copula_shrink010` | 0.629 | 0.678 |
| `independent_marginals` | 0.551 | 0.601 |
| `ctgan_small` | 0.543 | 0.594 |

Both designs fit the generator on 47 admissions and use the other 47 as non-members, so the regime is the same. The difference is the attack procedure: the benchmark trains a logistic-regression attack model on distances pooled from seven shadow worlds and applies it to the eighth, while this scores distances **within the single release under attack**.

The likely reason the simpler attack wins is that distances are not calibrated across shadow worlds. Each shadow fits a different generator on different rows, so a decision threshold learned on seven worlds transfers poorly to the eighth, whereas ranking records inside one release needs no cross-world threshold at all. That is a hypothesis consistent with the pattern, not something this experiment isolates; a targeted test would compare pooled-threshold against per-world-ranked scoring on the same shadow outputs.

For the diffusion model the practical reading is stark: at a 5% false-positive budget, a release-only attacker with no shadow models identifies **87.5% of the records the generator was trained on**, against 47.5% for the shadow attack at the same budget in the paper.

## What this changes

- §VII's "no density-based or release-only attacker was run" can be removed; both now exist.
- The paper's headline leakage figures are **conservative**. A weaker attacker than the one assumed does better, which strengthens rather than softens the argument.
- "Use the worst of several attacks" gains a second supporting case, since DOMIAS also clears TVAE.

## Limits

One dataset, 10 seeds, a 47-row fit rather than the 94-row release the main benchmark reports, and a single PCA-KDE configuration for DOMIAS with no bandwidth or dimension tuning. A fairer DOMIAS would use a larger reference sample, which this cohort cannot supply, and a normalizing-flow density of the kind the original paper uses.
