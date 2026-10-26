# Density-Based Attack — DOMIAS is weaker than distance here, and a release-only attacker matches the shadow pipeline once the release size is matched (2026-10-22, corrected 2026-10-24)

Post-course extension work (improvement plan item 3; limitations 3 and 4 of the membership survey; §VII of the paper, "no density-based or release-only attacker was run"). 6 arms × 10 seeds, 96 s.

```bash
python -m research.domias_attack --seeds 10 --workers 6                 # 500-row release, 96 s
python -m research.domias_attack --seeds 10 --workers 6 --n-synth 47    # 47-row release, matches the shadow attack, 206 s
```

> **Correction (2026-10-24).** The first version of this document compared a 500-row release against the benchmark's shadow attack, which releases 47 rows per shadow (`n_samples=len(member_df)` in `evaluation/membership_inference.py`). A larger release gives each member a closer nearest neighbour, so that comparison favoured the release-only attack for a reason unrelated to the attack procedure. The script now takes `--n-synth`, and the run at 47 rows replaces the old §2. The headline "a simpler attacker beats the shadow pipeline on every arm" does not hold at matched size and has been withdrawn.

---

## Setup

The generator is fitted on a random 47 of the 94 training admissions, so the other 47 are non-members **from the same distribution** and the holdout's distribution shift never enters. The attacker sees only the released synthetic table (500 rows, or 47 to match the benchmark) and a reference sample, which is the clean holdout with subject 41976 removed. No shadow generators, no retraining, no generator access.

Two scores are computed on the identical split, so they are directly comparable:

- **DOMIAS** (van Breugel et al., 2023): `log p_synth(x) − log p_ref(x)`, both densities Gaussian KDE (Scott's rule) on a 5-component PCA projection fitted only on attacker-visible data.
- **Distance**: minus the Gower distance to the nearest synthetic row, the same family the benchmark uses.

With 47 non-members the smallest non-zero false-positive rate is about 2%, so TPR is reported at 5% rather than 1%.

## Results with a 500-row release, mean over 10 seeds

| Arm | DOMIAS AUROC | DOMIAS TPR@5% | Distance AUROC | Distance TPR@5% | Difference |
|---|---|---|---|---|---|
| `tabular_diffusion` | 0.636 ± 0.070 | 0.140 | **0.976 ± 0.021** | **0.877** | −0.340 |
| `tvae` | **0.509 ± 0.061** | 0.096 | 0.755 ± 0.017 | 0.315 | −0.245 |
| `tabular_diffusion_reg` | 0.625 ± 0.054 | 0.126 | 0.747 ± 0.027 | 0.385 | −0.122 |
| `gaussian_copula_shrink010` | 0.567 ± 0.044 | 0.130 | 0.678 ± 0.042 | 0.153 | −0.111 |
| `ctgan_small` | 0.519 ± 0.058 | 0.034 | 0.603 ± 0.052 | 0.121 | −0.084 |
| `independent_marginals` | 0.564 ± 0.061 | 0.089 | 0.601 ± 0.043 | 0.121 | −0.037 |

Both tables were rerun on 2026-10-24 with the conservative TPR rule (below), on a second machine; the first version's 500-row AUROCs differ from these by at most 0.010.

---

## 1. The density attack is weaker on every arm, and it clears a known copier

DOMIAS loses to the distance attack on all six settings at both release sizes, by 0.037 to 0.340 AUROC at 500 rows and 0.047 to 0.223 at 47. The mechanism is the one the membership survey predicted before it was run: the reference density is estimated from 20 records in a 5-dimensional projection, which is not enough for the ratio to mean much. This is a negative result about **this cohort size**, not about DOMIAS, which was designed for settings with a usable reference sample.

The more useful part is which arm it misses. With a 500-row release **TVAE, the known copier, scores 0.509 under DOMIAS, essentially chance** — the lowest of all six arms, below even the independent floor. With a 47-row release it scores 0.553, level with the independent floor (0.544) and CTGAN-small (0.550), so DOMIAS still cannot separate it from settings that do not copy. A density-ratio attack therefore joins the 4-column attack as a second attack that would have cleared a generator we know copies 52 to 71% of its rows. That is direct evidence for the paper's existing rule of reporting the worst of several attacks rather than any single one, and it is worth more than a positive DOMIAS result would have been.

## Results with a 47-row release, mean over 10 seeds

TPR is the highest true-positive rate at a false-positive rate no higher than 5%, the rule `evaluation/membership_inference.py` uses. The first version interpolated between ROC points, which credits the attack with thresholds it cannot realise; with 47 non-members the two rules give the same values here to three decimals.

| Arm | DOMIAS AUROC | DOMIAS TPR@5% | Distance AUROC | Distance TPR@5% | Difference |
|---|---|---|---|---|---|
| `tabular_diffusion` | 0.587 ± 0.054 | 0.121 | **0.810 ± 0.039** | **0.477** | −0.223 |
| `tvae` | 0.553 ± 0.074 | 0.111 | 0.662 ± 0.036 | 0.172 | −0.109 |
| `tabular_diffusion_reg` | 0.585 ± 0.062 | 0.096 | 0.670 ± 0.028 | 0.268 | −0.086 |
| `gaussian_copula_shrink010` | 0.582 ± 0.056 | 0.149 | 0.654 ± 0.050 | 0.170 | −0.072 |
| `ctgan_small` | 0.550 ± 0.057 | 0.083 | 0.597 ± 0.052 | 0.119 | −0.047 |
| `independent_marginals` | 0.545 ± 0.041 | 0.051 | 0.596 ± 0.056 | 0.109 | −0.051 |

The release size moves the distance attack a long way: the 3000-step diffusion model falls from 0.976 to 0.810 and TVAE from 0.755 to 0.662. DOMIAS moves much less.

## 2. At matched release size, the release-only attack is about level with the shadow pipeline (corrected)

Comparing the direct release-only distance attack against the shadow-model Gower numbers in Table I of the paper. Both fit the generator on 47 admissions, use the other 47 as non-members and release 47 synthetic rows:

| Arm | Shadow-model Gower AUROC (paper) | Release-only, 47 rows | Release-only, 500 rows (superseded) |
|---|---|---|---|
| `tabular_diffusion` | 0.787 | 0.810 | 0.976 |
| `tabular_diffusion_reg` | 0.704 | **0.670** | 0.747 |
| `tvae` | 0.649 | 0.662 | 0.755 |
| `gaussian_copula_shrink010` | 0.629 | 0.654 | 0.678 |
| `independent_marginals` | 0.551 | 0.596 | 0.601 |
| `ctgan_small` | 0.543 | 0.597 | 0.603 |

At matched size the two attacks are close. The release-only attack is higher on five arms by 0.013 to 0.054 and lower on the regularised diffusion model by 0.034, and the order of the three leaking arms is not preserved (it puts TVAE level with the regularised model). At a 5% false-positive budget it finds 47.7% of the 3000-step diffusion model's training records, against 47.5% for the shadow attack in the paper: the same number.

So the earlier claim that a simpler attacker beats the shadow pipeline on every arm was mostly the release size. What remains is a small, mixed difference, and two other differences between the designs could account for it: this attack ranks distances within one release, while the shadow attack applies a threshold learned on seven other shadow worlds; and this attack's Gower distance is computed on the codec's columns rather than the benchmark's 53 features. Neither has been isolated.

The 500-row run is still informative on its own terms: a release ten times larger than the training set makes a copying generator far easier to attack (0.976 for the 3000-step diffusion model, 87.7% of members found at 5% FPR). Release size is a lever the data holder controls, and that is worth stating, but it is a statement about release size, not about the attack.

## What this changes

- §VII's "no density-based or release-only attacker was run" can be removed; both now exist.
- The paper's leakage figures are **not contradicted** by a release-only attacker: at matched release size it gives about the same numbers (0.810 against 0.787 for the 3000-step diffusion model, 47.7% against 47.5% of members at 5% FPR). Do not claim the shadow figures are conservative; that rested on the 500-row comparison.
- "Use the worst of several attacks" gains a second supporting case, since DOMIAS cannot separate TVAE from non-copying settings at either release size.
- Release size matters: the same generator is far more exposed when it releases more rows than it was trained on.

## Limits

One dataset, 10 seeds, a 47-row fit rather than the 94-row fit the main benchmark's released table comes from, a Gower distance on the codec's columns rather than the benchmark's feature set, and a single PCA-KDE configuration for DOMIAS with no bandwidth or dimension tuning. A fairer DOMIAS would use a larger reference sample, which this cohort cannot supply, and a normalizing-flow density of the kind the original paper uses.
