# Differentially Private Baseline — a real guarantee at n = 94 costs about 12x in fidelity and brings the attack to chance (2026-10-22, rebuilt 2026-10-24)

Post-course extension work (improvement plan item 4). The reported benchmark has no arm with a formal guarantee, so every privacy statement in it is attack-relative: it says an attack failed, not that leakage is bounded. This adds two differentially private mechanisms at two budgets each, scored on the benchmark's own metrics.

```bash
python -m research.dp_study --seeds 20 --workers 3     # 120 runs, 79 s
```

> **Rebuilt (2026-10-24).** The first version of `generators/dp_copula.py` was not differentially private, for three reasons found in review: numeric ranges and category levels were read from the training data rather than a public schema; the outputs were decoded through `FrameCodec`, which is fitted on the training rows and draws rare ICD-9 codes from the training pool by their real counts; and the correlation noise was calibrated per entry, which over a 110 x 110 matrix overspent the stated budget by a factor of thousands. The mechanism has been rewritten, every number below is from the rebuilt version, and the earlier numbers are listed at the end for comparison. The headline of the first version ("the attack cannot tell DP from a non-private baseline") no longer holds; it was largely the code leak.

---

## The mechanism

`generators/dp_copula.py`. **(epsilon, delta)-DP at the level of one admission**, with delta = 1e-5, bounded (replace-one) neighbours and the cohort size n public. Patient-level privacy would follow only by group privacy over a patient's admissions (up to 15 here), so it is not claimed.

- **Public domain, no codec.** The generators work on the raw table against a public domain: the published schema's categorical levels, the public ICD-9 dictionary (14,567 codes, `output/icd9_lookup.csv`) and fixed clinical bounds for age, both stay lengths and all 40 lab columns. Nothing about the training rows is read except through noise.
- **Three Gaussian measurements**, accounted with zero-concentrated DP:
  1. all 50 one-way histograms at once (10 bins per numeric column, an extra cell for missing values or for the masked age 89), L2 sensitivity sqrt(2 x 50);
  2. primary and secondary ICD-9 code counts over the whole dictionary, each admission contributing at most 38 secondary codes, L2 sensitivity sqrt(2 + 2 x 38);
  3. for `dp_copula` only, the latent correlation of 8 core columns (age, both stay lengths, number of codes, death, sex, admission type, care unit), each record's latent vector clipped to L2 norm sqrt(8), Frobenius sensitivity 2 x 8 / n.
- **Everything after is post-processing**: clipping negative counts, keeping only dictionary codes whose noisy count clears 3 sigma, projecting to a correlation matrix, and shrinking it toward independence by an amount set from the public noise level only.

Budget: rho = 0.0208 at epsilon = 1 and rho = 1.049 at epsilon = 8. Resulting noise standard deviation per histogram cell: 61 and 8.6 for the marginals, 73 and 10.3 for the code counts.

## Results, 20 seeds per arm

| Arm | Mean JSD ↓ | Corr. diff ↓ | TSTR LR / RF | Gower MIA ↓ | Synthetic death rate | Runs with an unusable label |
|---|---|---|---|---|---|---|
| `dp_marginals` ε=1 | 0.368 | 0.166 | 0.480 / 0.446 | **0.501** | 0.409 | 35% |
| `dp_marginals` ε=8 | **0.185** | 0.163 | 0.476 / 0.517 | 0.525 | 0.353 | **0%** |
| `dp_copula` ε=1 | 0.375 | 0.167 | 0.524 / 0.537 | **0.501** | 0.415 | 35% |
| `dp_copula` ε=8 | 0.190 | 0.164 | 0.499 / 0.511 | 0.521 | 0.357 | **0%** |
| `gaussian_copula_shrink010` (no DP) | 0.016 | 0.102 | 0.662 / 0.711 | 0.629 | 0.362 | 0% |
| `independent_marginals` (no DP) | 0.016 | 0.163 | 0.493 / 0.480 | 0.551 | 0.361 | 0% |

Utility is averaged over the runs where it is defined. Real training death rate is 0.362.

---

## Four findings

**1. The guarantee costs about 12x in marginal fidelity at ε = 8 and 23x at ε = 1.** Mean JSD goes from 0.016 without DP to 0.185 at ε = 8 and 0.37 at ε = 1. With 94 rows the noise on a histogram cell (sigma 8.6 at ε = 8) is the size of the counts it measures.

**2. The membership attack can now tell DP apart, and at ε = 1 it is at chance.** The Gower attack scores both DP arms at 0.501 at ε = 1, and 0.521 to 0.525 at ε = 8, against 0.551 for the non-private independent floor (Welch p = 0.007 and 0.014 at ε = 8, p < 0.0001 at ε = 1). This is the reverse of the first version, and the reason is the code leak: every non-private setting, the independent floor included, scores above chance because a generator can only emit ICD-9 codes it saw in training (see the paper's attack-anchor section). With codes drawn from a noisy count over the public dictionary instead, that channel closes.

**3. At ε = 8 every run is usable; at ε = 1 a third are not.** At ε = 1, 35% of runs produce a mortality column with a single class, so no classifier can be trained on them. At ε = 8 none do. Reporting only the mean AUROC over surviving runs would hide the ε = 1 failures.

**4. Dependence is not recoverable at this size.** Both DP arms carry no correlation structure (0.163 to 0.167, the floor's 0.163), and their utility is at chance (0.45 to 0.54). For `dp_copula` the correlation measurement is real but its noise (sigma 0.30 per entry at ε = 8, 2.2 at ε = 1) is larger than the correlations it measures, so the public-noise shrinkage pulls it to 70% and 99% independence. Spending budget on correlation at n = 94 buys nothing measurable; the ICD-9 codes fare similarly, since no code occurs often enough among 94 admissions to clear the 3-sigma threshold reliably, so the released codes are mostly drawn from noise.

## Against the first version

| Arm | Mean JSD, old → new | Unusable runs, old → new | Gower MIA, old → new |
|---|---|---|---|
| `dp_marginals` ε=1 | 0.390 → 0.368 | 45% → 35% | 0.535 → 0.501 |
| `dp_marginals` ε=8 | 0.286 → 0.185 | 20% → 0% | 0.533 → 0.525 |
| `dp_copula` ε=1 | 0.395 → 0.375 | 40% → 35% | 0.546 → 0.501 |
| `dp_copula` ε=8 | 0.310 → 0.190 | 25% → 0% | 0.558 → 0.521 |

The rebuilt mechanism is both private and better on every column of this table. Fidelity improved because measuring all histograms in one Gaussian mechanism adds four to five times less noise per cell than splitting a pure-ε Laplace budget across 110 columns, which outweighs the cost of using wide public ranges instead of the data's own. The attack scores fell because the code leak is gone.

## Limits of this baseline

- One-way marginals, plus a small correlation block. A mechanism that spends its budget on a few selected two-way marginals (MST, AIM) would likely keep more structure; this is a straightforward baseline, not the best DP synthesiser available.
- Admission-level DP only, delta = 1e-5, and the public-domain assumptions listed above (schema levels, ICD-9 dictionary, clinical bounds, n).
- The secondary-code cap (38) and the bin count (10) are fixed public choices, not tuned; tuning them on this data would itself use the data.
- Only the Gower membership attack was run against the DP arms, not the full three-attack suite with calibration.
- DP outputs contain ICD-9 codes that never occur in training, so they fail the Stage 2 output contract (`generators.validate`) by design and are scored here rather than through `evaluation.eval_runner`.
