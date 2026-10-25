# Learned Attacks Across Cohort Size — the cohort-size result survives a learned attack, and the UCI cohorts now have controls (2026-10-25)

Post-course extension work, in answer to the review's top remaining gap. The cohort-size claim ("the diffusion model's exposure falls to near chance as the cohort grows") rested on the Gower distance attack in [`size_attack.py`](size_attack.py). On MIMIC-III a learned attack calibrated per record found more than any distance attack ([learned attack result](learned_attack_result.md)), so the curve had to be retested with one. It holds.

```bash
python -m research.size_learned_attack --sizes 50 250 1000 5000 --seeds 2 --workers 3   # 768 shadow releases, 6,882 s
```

---

## Method

Same cohorts as the size sweep and the size attack (same fixed holdout, same patient draws for seeds 0 and 1). For each cohort, 16 shadow generators are each fitted on a random half of the cohort's rows and release as many rows as they were fitted on. The first 8 use exactly the member draws and fit seeds of `size_attack.py`, and the distance attack on those 8 reproduces every published `size_attack.csv` AUROC to within 2 × 10⁻⁶, so the harness is the same.

Every cohort row is described in each shadow by its 1st, 2nd and 3rd nearest Gower distances to the release, the mean of its 5 and 10 nearest, and its Gower distance to the nearest other cohort row (the per-record feature). On the same 16 shadows, scored leave-one-shadow-out: the distance attack (logistic regression on the nearest distance), gradient boosting on the 6 features, and the likelihood-ratio attack on the log nearest distance. Two control releases run through the same harness at every size: the members themselves (copy) and as many rows drawn from the fixed holdout (unseen records). These controls were missing from the UCI attack before.

Four settings, four sizes, two cohort draws. UCI has no multi-label code set, so the features are Gower only.

## Results

AUROC, mean of two cohort draws (draw standard deviation at most 0.038, typically below 0.01):

| Patients (training rows) | Setting | Distance (16 shadows) | Gradient boosting | Likelihood ratio | TPR at 1% FPR: distance / boosting / LR |
|---|---|---|---|---|---|
| 50 (67) | Copy control | 1.000 | 1.000 | 1.000 | 1.000 / 1.000 / 1.000 |
| | Unseen-record control | 0.498 | 0.489 | 0.527 | 0.010 / 0.009 / 0.017 |
| | Independent | 0.594 | 0.598 | 0.582 | 0.019 / 0.022 / 0.022 |
| | Copula λ=0.10 | 0.641 | 0.650 | 0.652 | 0.044 / 0.049 / 0.043 |
| | TVAE | 0.687 | 0.748 | **0.774** | 0.046 / 0.119 / 0.155 |
| | Diffusion | **0.851** | 0.838 | 0.836 | 0.517 / **0.605** / 0.544 |
| 250 (362) | Unseen-record control | 0.493 | 0.504 | 0.503 | |
| | Independent | 0.528 | 0.510 | 0.508 | |
| | Copula λ=0.10 | 0.536 | 0.518 | 0.517 | |
| | TVAE | 0.544 | 0.536 | 0.567 | 0.017 / 0.016 / 0.023 |
| | Diffusion | 0.713 | **0.753** | 0.719 | 0.102 / **0.161** / 0.155 |
| 1,000 (1,446) | Unseen-record control | 0.507 | 0.502 | 0.499 | |
| | Independent | 0.503 | 0.505 | 0.496 | |
| | Copula λ=0.10 | 0.509 | 0.500 | 0.507 | |
| | TVAE | 0.511 | 0.509 | 0.514 | |
| | Diffusion | 0.574 | **0.595** | 0.567 | 0.022 / 0.025 / 0.024 |
| 5,000 (7,145) | Unseen-record control | 0.500 | 0.501 | 0.501 | |
| | Independent | 0.500 | 0.501 | 0.501 | |
| | Copula λ=0.10 | 0.500 | 0.500 | 0.500 | |
| | TVAE | 0.510 | 0.508 | 0.508 | |
| | Diffusion | 0.517 | **0.520** | 0.507 | 0.012 / 0.012 / 0.012 |

The copy control scores 1.000 at every size under every attack.

## What it shows

**1. The cohort-size result holds under the learned attacks.** At 7,145 training rows no setting scores above 0.520 under any attack, and the diffusion model's best learned score falls from 0.838 at 67 rows to 0.753, 0.595 and 0.520. The share of its training records found at 1% FPR falls from 60.5% to 16.1%, 2.5% and 1.2%. The statement "the diffusion model's exposure approaches chance as the cohort grows" no longer depends on the attack being a distance attack.

**2. The learned attacks add most where the cohort is small.** Their gain over the distance attack is up to 0.087 at 67 rows (TVAE) and 0.040 at 362 rows (diffusion), but at most 0.021 at 1,446 rows and 0.003 at 7,145. A per-record attack needs a record whose presence moves the release, and in a large cohort no single record does.

**3. TVAE's statement is refined, not overturned.** At 362 rows the likelihood-ratio attack scores TVAE at 0.567 against 0.508 for the independent setting, so TVAE is not yet at the independent setting there; from 1,446 rows on it is within 0.018 of the independent setting under each attack (0.008 distance, 0.004 boosting, 0.018 likelihood ratio at 1,446 rows; 0.007 to 0.010 at 7,145). The size attack's "within 0.014 from 362 rows on" was a distance-attack statement.

**4. The learned attacks gain far less on UCI than on MIMIC-III.** On MIMIC-III the boosted attack lifted the independent setting from 0.642 to 0.782, mostly through the ICD-9 code-set features. UCI has no code set, and even at 67 rows the independent setting moves only from 0.594 to 0.598. The MIMIC gain belongs to the rare-code channel, which the cohort-size result does not rely on.

**5. The UCI attacks now have controls.** The copy control scores 1.000 and the unseen-record control 0.489 to 0.527 at every size, so the UCI scale is anchored as the MIMIC one is.

## What this changes for the paper

- The cohort-size claim can be stated under every attack: diffusion's best learned AUROC 0.838 → 0.520 from 67 to 7,145 rows; nothing above 0.520 at 7,145.
- TVAE: "within 0.018 of the independent setting from 1,446 rows on under every attack" (not "within 0.014 from 362 rows on").
- Remove the limitation that the UCI attack had no copy or unseen-record controls, and the one that the cohort-size results may understate exposure because they used distance attacks only.

## Limits

Two cohort draws per size, four settings, 16 shadows (the MIMIC learned attacks used 32), Gower features only. One untuned boosted-model configuration.
