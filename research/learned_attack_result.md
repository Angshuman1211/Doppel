# Learned Membership Attacks — stronger attacks lift every non-private setting, keep the diffusion model on top at low FPR, and leave the DP baselines at chance (2026-10-25)

Post-course extension work, in answer to a review comment (M5). The paper's attacks are a logistic regression on one nearest-neighbour distance per run, three distances, worst case taken, checked against copy and unseen-record releases. The draft argued that this checked worst case bounds what an attacker finds. A reviewer may ask for a learned attack, so we ran two. **The argument does not hold**: both learned attacks find more than the distance attack in every non-private setting.

```bash
python -m research.learned_attack --seeds 3 --workers 4                    # 21 runs, 672 shadow fits, 1,173 s
python -m research.learned_attack --seeds 3 --workers 4 --out learned_attack_dp \
    --arms dp_marginals_eps1 dp_marginals_eps8 dp_copula_eps1 dp_copula_eps8 dp_bayesnet_eps1 dp_bayesnet_eps8   # 56 s
python -m research.learned_attack_ablation                                  # refits attack models only
# added after review, the four remaining non-private settings (LOKY_MAX_CPU_COUNT=1, see below)
python -m research.learned_attack --seeds 3 --workers 6 --out learned_attack_copulas --arms gaussian_copula gaussian_copula_shrink025
python -m research.learned_attack --seeds 3 --workers 5 --out learned_attack_ctgan --arms ctgan ctgan_small
```

---

## Method

Same threat model as the benchmark: the attacker knows the 94 training admissions as candidates, can fit the generator on any subset of them, and sees one 47-row release. For each setting and seed, 32 shadow generators are each fitted on a random 47 of the 94 admissions (the benchmark uses 8). On the same shadows, four attacks, each fitted on 31 shadows and scored on the 32nd in turn:

- **Distance attack** (the benchmark's): logistic regression on the nearest synthetic distance, for the 4-column Euclidean, Gower and ICD-9 Jaccard distances; worst case of the three.
- **Feature logistic regression**: the same model on 18 features. Per distance: the 1st, 2nd and 3rd nearest synthetic distances, the mean of the 5 and of the 10 nearest, and the record's distance to its nearest *other real* record.
- **Gradient boosting** on the same 18 features (`HistGradientBoostingClassifier`, 200 iterations, learning rate 0.05, 15 leaves).
- **LiRA** after Carlini et al. (2022): per record, Gaussians fitted to its log nearest-neighbour distance (three distances) over the shadows where it was in and where it was out, with a variance per distance shared across records; the score is the log likelihood ratio.

The real-neighbour distance is constant for a record across shadows, so it lets the boosted model tell an unusual record from an exposed one and set a threshold per record, which is what LiRA does explicitly. That is allowed in this threat model, because the attacker fits shadows on the candidates themselves. Seeds 42 to 44. Reported: mean AUROC over held-out shadows and the pooled TPR at 1% FPR (about 1,500 non-member scores).

## Results

Mean of 3 seeds. Seed standard deviation of the learned AUROCs: 0.003 to 0.015 for the non-private settings, up to 0.022 for the DP settings and 0.043 for the unseen-record control's boosted model.

| Setting | Distance attack (worst case) | Feature LR | Gradient boosting | LiRA | TPR at 1% FPR: distance | boosting | LiRA |
|---|---|---|---|---|---|---|---|
| Copy control | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Unseen-record control | 0.523 | 0.509 | 0.489 | 0.499 | 0.007 | 0.008 | 0.009 |
| Independent | 0.642 | 0.680 | **0.782** | 0.697 | 0.018 | 0.123 | 0.035 |
| Copula λ=0.10 | 0.662 | 0.761 | **0.844** | 0.785 | 0.028 | 0.193 | 0.090 |
| TVAE | 0.684 | 0.815 | **0.848** | 0.835 | 0.048 | 0.247 | 0.259 |
| Diffusion (3000 steps) | 0.796 | 0.862 | **0.872** | 0.833 | 0.415 | **0.541** | **0.495** |
| Diffusion-reg | 0.700 | 0.764 | **0.793** | 0.769 | 0.239 | 0.276 | 0.254 |
| Copula (Ledoit-Wolf) | 0.646 | 0.705 | **0.803** | 0.728 | 0.022 | 0.123 | 0.053 |
| Copula λ=0.25 | 0.656 | 0.746 | **0.833** | 0.768 | 0.028 | 0.151 | 0.067 |
| CTGAN | 0.626 | 0.668 | **0.737** | 0.635 | 0.020 | 0.079 | 0.025 |
| CTGAN-small | 0.635 | 0.672 | **0.764** | 0.678 | 0.022 | 0.094 | 0.032 |
| DP-marginals ε=1 / ε=8 | 0.512 / 0.532 | 0.504 / 0.536 | 0.491 / 0.517 | 0.492 / 0.503 | ≤ 0.016 | ≤ 0.011 | ≤ 0.014 |
| DP-copula ε=1 / ε=8 | 0.515 / 0.525 | 0.510 / 0.535 | 0.506 / 0.528 | 0.504 / 0.509 | ≤ 0.012 | ≤ 0.016 | ≤ 0.014 |
| DP-Bayes ε=1 / ε=8 | 0.521 / 0.534 | 0.525 / 0.532 | 0.504 / 0.510 | 0.505 / 0.524 | ≤ 0.012 | ≤ 0.015 | ≤ 0.017 |

The distance attack on 32 shadows reproduces the benchmark's 8-shadow scores (0.642, 0.662, 0.684, 0.796, 0.700 against 0.638, 0.656, 0.680, 0.787, 0.706), so the gain below is not a change of harness.

Where the boosted model's gain comes from (`learned_attack_ablation.csv`, same seeds):

| Setting | Full model | Without the real-neighbour feature | Codes features only | Gower features only | 4-column features only | Full model, 8 shadows | Distance attack, 8 shadows |
|---|---|---|---|---|---|---|---|
| Independent | 0.782 | 0.680 | 0.740 | 0.595 | 0.525 | 0.693 | 0.632 |
| Copula λ=0.10 | 0.844 | 0.741 | 0.766 | 0.719 | 0.520 | 0.763 | 0.667 |
| TVAE | 0.848 | 0.790 | 0.744 | 0.780 | 0.530 | 0.807 | 0.681 |
| Diffusion | 0.872 | 0.850 | 0.812 | 0.824 | 0.602 | 0.836 | 0.793 |
| Diffusion-reg | 0.793 | 0.758 | 0.735 | 0.718 | 0.523 | 0.729 | 0.698 |

## What it shows

**1. The distance attack does not bound what an attacker finds.** Gradient boosting raises the AUROC of every non-private setting by 0.08 to 0.18 over the checked worst-case distance attack, and LiRA by 0.04 to 0.15. The controls still behave (copy 1.000, unseen records 0.49 to 0.52), so the gain is membership signal, not attack overfitting. The distance attack is a lower bound on exposure, as every empirical attack is; it is not a ceiling.

**2. The 3000-step diffusion model is still the most exposed where it matters.** At 1% FPR the boosted model finds 54.1% of its training records and LiRA 49.5%, against at most 27.6% for any other setting. Its AUROC lead shrinks, though: under the boosted model it is 0.872 against 0.848 for TVAE and 0.844 for the copula, and under LiRA TVAE matches it (0.835 against 0.833).

**3. The settings that "do not copy rows" are not safe.** Under the distance attack the independent setting and the copula score 0.64 to 0.66 with TPR under 3%. Under the boosted model they score 0.78 and 0.84, with 12.3% and 19.3% of records found at 1% FPR. They store no whole admission, but each training record's rare codes and values move the marginals and the code frequencies when it is in the training set, and a per-record attack picks that shift up. Removing the per-record feature cuts the copula from 0.844 to 0.741 and the independent setting from 0.782 to 0.680, and the codes features alone reach 0.740 and 0.766. The independent setting's 0.638 is therefore not a resampling limit of this cohort; it is the limit of a distance attack.

**4. Some orderings change.** Under both learned attacks TVAE now scores above the regularised diffusion model on AUROC (0.848 against 0.793 boosted, 0.835 against 0.769 LiRA), with similar TPR (25 to 26% against 25 to 28%), and above the copula on TPR. Under the boosted model the copula's AUROC is within 0.004 of TVAE's. The orderings the paper states with the distance attack (diffusion > regularised diffusion > TVAE > copula > independent) hold for the distance attack only.

**5. The DP baselines stay at chance.** Every learned attack scores the six DP settings at 0.49 to 0.54 AUROC with at most 1.7% TPR at 1% FPR, while the same attacks score the non-private independent setting at 0.78. This is the strongest evidence in the benchmark for the DP result: the attacks that lift every non-private setting leave the DP ones where they were.

**6. The gain is the attack, not the shadow count.** With the benchmark's 8 shadows the boosted model still beats the distance attack by 0.03 to 0.13, and the distance attack gains at most 0.010 from 32 shadows. The per-record calibration needs no more shadows than the benchmark already fits.

**7. All nine non-private settings are lifted, and the copulas trade privacy for structure smoothly.** The four settings added after review follow the same pattern: gradient boosting adds 0.111 to 0.177 AUROC over the distance attack. Across the three copulas the boosted score rises as shrinkage falls, 0.803 (λ = 0.668), 0.833 (λ = 0.25) and 0.844 (λ = 0.10), with TPR at 1% FPR 12.3%, 15.1% and 19.3%: the correlation structure the paper credits to low shrinkage is paid for in exposure. CTGAN is the least exposed non-private setting under every learned attack (0.737, 7.9%), below even the independent setting, consistent with it reproducing neither exact training values nor dependence between columns.

Running CTGAN in parallel workers needs `LOKY_MAX_CPU_COUNT=1`: its `DataTransformer` calls joblib `Parallel(n_jobs=-1)`, which otherwise starts one process per core inside every worker; the first attempt exhausted the Windows paging file. The script now sets it.

## Across patient splits (added after review)

```bash
python -m research.learned_attack --splits 10 --workers 5 --out learned_attack_splits --arms exact_copy disjoint_real \
    independent_marginals gaussian_copula_shrink010 tvae tabular_diffusion tabular_diffusion_reg   # 70 runs, 2,240 shadow fits, 5,896 s
python -m research.learned_attack_tests
```

The same 32-shadow design on the 10 redrawn patient splits of [`split_attack.py`](split_attack.py) (split seeds 1000 to 1009, 89 to 107 training admissions, generator seed 42), with both controls on every split. Every pair of the five settings is compared with a paired Wilcoxon test over the splits, Holm-corrected as a family of ten per attack and metric; with 10 splits the smallest Holm-adjusted p is 0.020.

Mean ± sd over 10 splits:

| Setting | Distance (worst case) | Gradient boosting | Likelihood ratio | TPR at 1% FPR: distance / boosting / LR |
|---|---|---|---|---|
| Copy control | 1.000 | 1.000 | 1.000 | 1.000 / 1.000 / 1.000 |
| Unseen-record control | 0.508 ± 0.011 | 0.506 ± 0.019 | 0.494 ± 0.016 | 0.008 / 0.007 / 0.007 |
| Diffusion (3000 steps) | **0.797 ± 0.010** | **0.862 ± 0.013** | **0.827 ± 0.009** | **0.379 / 0.534 / 0.442** |
| TVAE | 0.666 ± 0.015 | 0.835 ± 0.022 | 0.811 ± 0.022 | 0.071 / 0.253 / 0.191 |
| Copula λ=0.10 | 0.640 ± 0.020 | 0.833 ± 0.029 | 0.769 ± 0.028 | 0.039 / 0.172 / 0.082 |
| Diffusion-reg | 0.690 ± 0.015 | 0.780 ± 0.032 | 0.745 ± 0.026 | 0.172 / 0.266 / 0.219 |
| Independent | 0.618 ± 0.026 | 0.768 ± 0.031 | 0.683 ± 0.024 | 0.019 / 0.097 / 0.037 |

**The learned attacks beat the distance attack on every split.** For each of the five non-private settings, gradient boosting exceeds the worst-case distance attack on 10 of 10 splits (by 0.052 to 0.214; mean gain 0.065 for diffusion, 0.090 for regularised diffusion, 0.150 for the independent setting, 0.169 for TVAE and 0.192 for the copula), and the likelihood-ratio attack on 10 of 10 (by at least 0.021). The unseen-record control gains nothing (0.47 to 0.54 under every attack).

**The diffusion model is the most exposed setting over patients, under every attack.** It beats every other setting on AUROC on 9 or 10 of 10 splits under all three attacks, and on TPR at 1% FPR on 10 of 10 (53.4% boosted, 44.2% likelihood ratio), all Holm p = 0.020.

**TVAE above the regularised diffusion model is a robust reversal.** Under both learned attacks TVAE has the higher AUROC on 10 of 10 splits (+0.056 boosted, +0.066 likelihood ratio, Holm p = 0.020), while under the distance attack the regularised model is higher on 10 of 10 (+0.023). On TPR at 1% FPR the two are not separable under either learned attack (5 of 10 splits, Holm p ≥ 0.23).

**AUROC and low-FPR exposure rank the copula and the regularised diffusion model in opposite orders.** The copula has the higher learned-attack AUROC on 10 of 10 splits (+0.053 boosted, +0.024 likelihood ratio), and the regularised diffusion model the higher TPR at 1% FPR on 10 of 10 (+0.094 and +0.137). The copula leaks a little about many records; the regularised model, which collapses onto about 25 admissions, leaks a lot about a few. Which one is "more private" depends on which of these an evaluation weights, and the paper follows Carlini et al. in weighting the second.

**TVAE and the copula tie on boosted AUROC** (+0.003, 6 of 10 splits, Holm p = 0.43), but TVAE is higher on likelihood-ratio AUROC and on TPR at 1% FPR under both learned attacks (9 or 10 of 10 splits, Holm p = 0.020). The copula is above the independent setting under every attack and metric (10 of 10 or 9 of 10, Holm p = 0.020).

## What this changes for the paper

- Withdraw "the checked worst-case distance attack bounds what we found". State that the distance attack gives a lower bound and report the learned attacks beside it.
- Keep the lead finding: the 3000-step diffusion model has the highest exposure at low FPR under every attack (41.5%, 54.1%, 49.5% at 1% FPR).
- Qualify every statement that the settings without row copying sit at a low or shared attack score: that is true of the distance attack, not of a learned one.
- The copula's privacy advantage over the neural settings is attack-dependent: lower TPR than TVAE and diffusion under both learned attacks, but no clear AUROC gap from TVAE under boosting.
- The DP result is strengthened.
- The UCI cohort-size and split results use the distance attack only; the learned attacks were run on the reported MIMIC-III split.

## Limits

Three seeds on the reported split for all nine non-private settings and the six DP settings; the 10-split rerun covers five settings and the two controls, one generator seed per split, and the splits are redraws of the same 100 patients. One boosted-model configuration, not tuned. LiRA uses a shared variance and only the nearest distance per metric. Zhang et al.'s contrastive-learning attack was not run, so stronger attacks than these may exist; the scores here are still lower bounds. TPR at 1% FPR rests on about 15 false positives per pooled curve.
