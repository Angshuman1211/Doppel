# Learned Membership Attacks — stronger attacks lift every non-private setting, keep the diffusion model on top at low FPR, and leave the DP baselines at chance (2026-10-25)

Post-course extension work, in answer to a review comment (M5). The paper's attacks are a logistic regression on one nearest-neighbour distance per run, three distances, worst case taken, checked against copy and unseen-record releases. The draft argued that this checked worst case bounds what an attacker finds. A reviewer may ask for a learned attack, so we ran two. **The argument does not hold**: both learned attacks find more than the distance attack in every non-private setting.

```bash
python -m research.learned_attack --seeds 3 --workers 4                    # 21 runs, 672 shadow fits, 1,173 s
python -m research.learned_attack --seeds 3 --workers 4 --out learned_attack_dp \
    --arms dp_marginals_eps1 dp_marginals_eps8 dp_copula_eps1 dp_copula_eps8 dp_bayesnet_eps1 dp_bayesnet_eps8   # 56 s
python -m research.learned_attack_ablation                                  # refits attack models only
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

## What this changes for the paper

- Withdraw "the checked worst-case distance attack bounds what we found". State that the distance attack gives a lower bound and report the learned attacks beside it.
- Keep the lead finding: the 3000-step diffusion model has the highest exposure at low FPR under every attack (41.5%, 54.1%, 49.5% at 1% FPR).
- Qualify every statement that the settings without row copying sit at a low or shared attack score: that is true of the distance attack, not of a learned one.
- The copula's privacy advantage over the neural settings is attack-dependent: lower TPR than TVAE and diffusion under both learned attacks, but no clear AUROC gap from TVAE under boosting.
- The DP result is strengthened.
- The UCI cohort-size and split results use the distance attack only; the learned attacks were run on the reported MIMIC-III split.

## Limits

Three seeds, the reported split only, five non-private settings and six DP settings (CTGAN, the small CTGAN and the default and λ=0.25 copulas were not run). One boosted-model configuration, not tuned. LiRA uses a shared variance and only the nearest distance per metric. Zhang et al.'s contrastive-learning attack was not run, so stronger attacks than these may exist; the scores here are still lower bounds. TPR at 1% FPR rests on about 15 false positives per pooled curve.
