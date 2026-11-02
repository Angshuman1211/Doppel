# Response to the Second Review — training passes, the rare-code channel, the model actually attacked, a synthpop-style baseline, and corrected tests (2026-11-02)

Post-course extension work, in answer to a detailed external review of the journal draft. Each study below tests a specific objection. Two of them change what the paper can claim: the utility comparisons are not significant once split overlap is corrected, and cohort size accounts for only part of the fall in exposure with cohort size.

```bash
python -m research.passes_study --workers 4                 # 176 fits, 8,039 s
python -m research.code_channel_study --workers 4           # 15 runs, 480 shadow fits
python -m research.fit_size_study --workers 4               # 440 tasks
python -m research.learned_attack --seeds 3 --workers 3 --out learned_attack_cart --arms sequential_cart
python -m research.split_study --splits 30 --seeds 3 --workers 3 --out split_study_cart --arms sequential_cart
python -m research.nb_correction
python -m research.tpr_bootstrap --resamples 1000 --workers 4
```

---

## 1. Cohort size at fixed passes per record

The diffusion model trains for a fixed 3,000 steps at batch 64, so a smaller cohort also gets more passes per record. `passes_study.py` holds passes per fitted record fixed, on the same UCI cohorts, member draws and fit seeds as `size_attack.py` (8 shadows, two cohort draws, width 256). A shadow is fitted on half a cohort, so 3,000 steps on 33 member rows is 5,818 passes. The steps follow each draw's member count, so that cell is close to the size attack's configuration and not identical to it (0.844 against 0.850).

| Passes | Rows | Steps | Distance AUROC | Boosted AUROC | Likelihood ratio | Distance TPR@1% | Boosted TPR@1% |
|---|---|---|---|---|---|---|---|
| 100 | 67 | 53 | 0.617 | 0.565 | 0.631 | 0.034 | 0.014 |
| 100 | 362 | 283 | 0.545 | 0.528 | 0.514 | 0.012 | 0.013 |
| 100 | 1,446 | 1,129 | 0.548 | 0.550 | 0.529 | 0.012 | 0.019 |
| 100 | 7,145 | 5,582 | 0.519 | 0.518 | 0.501 | 0.011 | 0.010 |
| 1,000 | 67 | 524 | 0.787 | 0.789 | 0.766 | 0.161 | 0.365 |
| 1,000 | 362 | 2,828 | 0.715 | 0.744 | 0.707 | 0.100 | 0.163 |
| 1,000 | 1,446 | 11,289 | 0.620 | 0.644 | 0.604 | 0.039 | 0.064 |
| 1,000 | 7,145 | 55,821 | 0.540 | 0.551 | 0.524 | 0.016 | 0.019 |
| 5,818 | 67 | 3,046 | 0.844 | 0.858 | 0.812 | 0.481 | 0.542 |
| 5,818 | 362 | 16,454 | 0.801 | 0.819 | 0.784 | 0.414 | 0.492 |
| 5,818 | 1,446 | 65,680 | 0.696 | 0.733 | 0.680 | 0.162 | 0.221 |

The 5,818-pass cell at 7,145 rows would need about 325,000 steps per fit and was not run.

**The objection was right, and cohort size still matters.** From 67 to 1,446 rows the distance attack falls by 0.274 at fixed steps (0.850 to 0.576) and by 0.148 at fixed passes (0.844 to 0.696), so roughly half of the fall in the original design is fewer passes per record. The remainder is a cohort effect that holds at 1,000 and at 5,818 passes. At the top of the range the cohort effect is visible at 1,000 passes (0.787 to 0.540), and more steps can bring exposure back: at 7,145 rows, 55,821 steps score 0.540 by distance and 0.551 under boosting, against 0.517 and 0.520 at 3,000 steps. "Near chance at 7,145 rows" therefore holds at this training budget and not at any budget, although even at 55,821 steps only 1.9% of records are found at 1% FPR.

The paper's wording changes from "cohort size" to "cohort size and the training budget per record".

At 100 passes the 67-row cohort scores 0.617 by distance but 0.565 under boosting after 53 steps. That model is hardly trained, the two draws differ by 0.055 standard deviation, and this cell should not be over-read.

## 2. The rare-code channel

`generators/codec.py` draws secondary codes seen in fewer than five training admissions from a pool of the training rows' own codes, and the primary code is a categorical column over training levels. `code_channel_study.py` refits with one change at decoding: rare secondary and rare primary codes are drawn uniformly from the public ICD-9 dictionary, as the DP baselines do. Codes seen in at least five admissions stay. Same seeds, member draws and generator seeds as `learned_attack.py`, which is the paired baseline; three seeds, 32 shadows.

| Setting | Boosted: shared → public | Likelihood ratio | Distance worst case | Code-set distance | Boosted TPR@1% | Mean divergence (full fit) |
|---|---|---|---|---|---|---|
| Independent | 0.782 → 0.634 | 0.697 → 0.623 | 0.642 → 0.570 | 0.642 → 0.570 | 0.123 → 0.045 | 0.016 → 0.030 |
| Copula λ=0.10 | 0.844 → 0.759 | 0.785 → 0.741 | 0.662 → 0.622 | 0.662 → 0.593 | 0.193 → 0.097 | 0.016 → 0.030 |
| TVAE | 0.848 → 0.759 | 0.835 → 0.775 | 0.684 → 0.629 | 0.684 → 0.584 | 0.247 → 0.122 | 0.106 → 0.115 |
| Diffusion-reg | 0.793 → 0.727 | 0.769 → 0.740 | 0.700 → 0.687 | 0.679 → 0.583 | 0.276 → 0.213 | 0.378 → 0.406 |
| Diffusion (3000 steps) | 0.872 → 0.828 | 0.833 → 0.821 | 0.796 → 0.777 | 0.751 → 0.654 | 0.541 → 0.488 | 0.042 → 0.062 |

Part of the leak in the settings that copy no rows is the shared decoder and not the generator: their boosted TPR at 1% FPR about halves, and the independent setting loses 0.148 AUROC. The diffusion model loses 0.044 and keeps 48.8% TPR, so its exposure is mostly its own, and its AUROC lead over the other settings widens (0.828 against at most 0.759). No setting reaches chance: even the independent setting stays at 0.634, because it still samples training values and training codes seen at least five times.

"Artefact" would be too strong a word. Emitting only codes seen in training is what generators normally do, so the shared decoder's numbers describe real practice. The public vocabulary is not free: mean divergence roughly doubles for the statistical settings, largely because the primary code column changes.

## 3. The model that is attacked, and a multivariate fidelity test

Privacy is measured on fits to 47 admissions while the benchmark's fidelity and utility come from fits to 94. `fit_size_study.py` scores both, 20 seeds each, with each fit and release of the same size. Fidelity is against the rows that were fitted. It also adds a 5-fold random-forest two-sample test on the utility features (no code lists): C2ST is its held-out AUROC and pMSE the mean squared deviation of its probabilities from the base rate.

| Setting | Divergence 94 / 47 | Corr. error 94 / 47 | TSTR 94 / 47 | Deaths 94 / 47 | C2ST 94 / 47 | pMSE 94 / 47 |
|---|---|---|---|---|---|---|
| Independent | 0.016 / 0.031 | 0.163 / 0.207 | 0.486 / 0.531 | 0.361 / 0.352 | 0.626 / 0.354 | 0.006 / 0.006 |
| Copula (LW) | 0.016 / 0.031 | 0.138 / 0.175 | 0.593 / 0.554 | 0.367 / 0.346 | 0.482 / 0.263 | 0.006 / 0.008 |
| Copula λ=0.25 | 0.016 / 0.031 | 0.110 / 0.142 | 0.659 / 0.600 | 0.363 / 0.351 | 0.345 / 0.209 | 0.007 / 0.011 |
| Copula λ=0.10 | 0.016 / 0.031 | 0.102 / 0.134 | 0.686 / 0.611 | 0.362 / 0.354 | 0.316 / 0.201 | 0.007 / 0.012 |
| CTGAN | 0.094 / 0.135 | 0.165 / 0.209 | 0.509 / 0.468 | 0.378 / 0.344 | 0.993 / 0.995 | 0.076 / 0.073 |
| CTGAN-small | 0.072 / 0.097 | 0.169 / 0.215 | 0.532 / 0.490 | 0.389 / 0.385 | 0.981 / 0.947 | 0.055 / 0.033 |
| TVAE | 0.106 / 0.145 | 0.122 / 0.164 | 0.704 / 0.686 | 0.210 / 0.231 | 0.984 / 0.962 | 0.051 / 0.035 |
| Diffusion | 0.042 / 0.065 | 0.174 / 0.186 | 0.762 / 0.729 | 0.446 / 0.444 | 0.646 / 0.500 | 0.031 / 0.032 |
| Diffusion-reg | 0.378 / 0.378 | 0.160 / 0.183 | 0.773 / 0.755 | 0.787 / 0.631 | 0.999 / 0.999 | 0.193 / 0.164 |
| Sequential CART | 0.018 / 0.031 | 0.117 / 0.165 | 0.607 / 0.631 | 0.345 / 0.382 | 0.347 / 0.250 | 0.010 / 0.013 |

The mixed comparison does not change the picture. Utility falls by 0.018 to 0.075 at 47 rows for most settings and rises for the independent setting and CART, which is within the seed spread of this 35-admission test. The order of the settings on utility barely moves, and the diffusion settings still lead. Divergence against a 47-row table about doubles for the statistical settings, as expected for a smaller sample.

The two-sample test separates what correlation error cannot. A classifier tells CTGAN, TVAE and the regularised diffusion model from real rows almost perfectly (0.98 to 0.999), the 3000-step model much less (0.646), and the copulas and CART not at all. References: two random halves of the real training rows score 0.517, and real training rows against real held-out rows 0.656. Scores below 0.5 for the statistical settings are a cross-validation effect of small samples containing repeated training values, and should be read as "cannot be separated", not as closer than real data. The features leave out the code lists.

## 4. A synthpop-style baseline

`generators/sequential_cart.py` implements sequential CART without smoothing and with a minimum leaf of 5, the default `cart` method of synthpop (Nowok et al., 2016), on the shared frame. It is a reimplementation, not the R package. Every generated value is a training value, so exposure was expected.

| Measure | Sequential CART | Copula λ=0.10 |
|---|---|---|
| Distance worst case | 0.690 | 0.662 |
| Boosted / likelihood ratio | 0.860 / 0.804 | 0.844 / 0.785 |
| Boosted TPR at 1% FPR | 21.3% | 19.3% |
| Utility, 30 splits | 0.624 | 0.652 |
| Mean divergence / correlation error | 0.018 / 0.117 | 0.016 / 0.102 |

On every measure here it sits slightly on the wrong side of the tuned copula.

## 5. Corrected tests over the splits

`nb_correction.py` applies the corrected resampled t-test of Nadeau and Bengio (2003) with n_test/n_train = 0.272. It was derived for test error, so it applies directly to utility and conservatively to fidelity; for the attacks the same inflation (0.25) is a sensitivity check.

| Comparison | Diff. | Splits won | Wilcoxon Holm p | Corrected p | Corrected Holm p |
|---|---|---|---|---|---|
| Diffusion − copula λ=0.10, utility | +0.063 | 23/30 | 0.0006 | 0.165 | 0.965 |
| Diffusion-reg − copula, utility | +0.071 | 22/30 | 0.0010 | 0.161 | 0.965 |
| TVAE − copula, utility | +0.046 | 23/30 | 0.0034 | 0.257 | 0.965 |
| Copula − independent, utility | +0.067 | 24/30 | 0.0015 | 0.186 | 0.965 |

**The utility orderings are descriptive.** No utility comparison stays significant. All fidelity orderings do (corrected Holm p below 10^-5), as does every distance-attack comparison (at most 0.011), the gain of both learned attacks over the distance attack on every setting, and the diffusion model's TPR lead under both learned attacks. Its learned-attack AUROC lead over TVAE and the copula does not (corrected Holm 0.12 to 0.18), so "the diffusion model leaks most" rests on TPR at low FPR. Full table: `research/output/nb_correction.csv`.

## 6. TPR intervals

`tpr_bootstrap.py` bootstraps whole records, because the 1,500 pooled non-member scores come from 94 records. On the reported split, boosted TPR at 1% FPR with a 95% record-clustered interval: diffusion 54.1% (51.7 to 56.5), regularised diffusion 27.6% (23.7 to 31.5), TVAE 24.7% (21.8 to 28.7), CART 21.3% (16.4 to 24.3), tuned copula 19.3% (14.0 to 22.5), independent 12.3% (9.5 to 14.6). Over the 10 splits with the corrected variance, the diffusion model's 53.4% has an interval of 50.2 to 56.6%, wider than the earlier uncorrected 51.7 to 55.1%.

## Limits

The fixed-pass grid has two cohort draws and no 5,818-pass cell at 7,145 rows. The code-channel study keeps codes seen at least five times and still reports the shared decoder's numeric values. The two-sample test leaves out the code lists. CART is a reimplementation, without the R package's smoothing options.
