# Final Evaluation Result — nine arms: diffusion leaks most, the tuned copula is the best trade-off, and Track 1's run reproduces in direction (2026-10-21)

Raw evidence for Track 2's evaluation of all nine arms: the five from Phase 3 and the four Track 1 added after tuning (ADR-029), including the blueprint's third family, tabular diffusion. Reproduce from the repo root:

```bash
pip install -r generators/requirements-neural.txt
for arm in independent_marginals gaussian_copula gaussian_copula_shrink025 gaussian_copula_shrink010 \
           ctgan ctgan_small tvae tabular_diffusion tabular_diffusion_reg; do
  for seed in $(seq 42 61); do python -m evaluation.eval_runner --generator $arm --seed $seed; done
done
python -m evaluation.summarize_results
python -m evaluation.pareto
python -m evaluation.mia_calibration --workers 6 --arms gaussian_copula_shrink010 ctgan_small tabular_diffusion tabular_diffusion_reg \
       --out results/calibration/mia_calibration_phase3_arms.json
python -m evaluation.synthetic_coverage
```

This supersedes the five-arm headline in [`full_evaluation_result.md`](full_evaluation_result.md) and [`full_pareto_result.md`](full_pareto_result.md), whose numbers for those five arms are unchanged and still stand. It is Track 2's own run of the comparison Track 1 first made in [`final_comparison_result.md`](final_comparison_result.md); the two are compared at the end.

---

## Method (honest framing)

Nine arms, each fitted on the 94 train rows over 20 seeds (42 to 61), 180 runs. Metrics, attacks and thresholds are those in [`eval_protocol.md`](eval_protocol.md), unchanged: fidelity against real train, utility as train-on-synthetic, test-on-real on `hospital_expire_flag` against the 35-row holdout, membership inference with 8 shadow generators and three attacks scored on the worst case, attribute inference on three targets.

| Arm | Config | Role |
|---|---|---|
| `independent_marginals` | none | no-dependence floor |
| `gaussian_copula` | Ledoit-Wolf shrinkage | Phase 1 statistical baseline |
| `gaussian_copula_shrink025` | shrinkage 0.25 | the copula Track 1 carried into Phase 3 |
| `gaussian_copula_shrink010` | shrinkage 0.10 | Track 1's tuned copula (ADR-029) |
| `ctgan` | 300 epochs | Phase 3 CTGAN |
| `ctgan_small` | 300 epochs, 64×64 networks, embedding 32 | Track 1's tuned CTGAN |
| `tvae` | 300 epochs | memorizes; the membership-attack positive control |
| `tabular_diffusion` | 3000 steps | best neural fidelity in Track 1's tuning, copies 80% of rows there |
| `tabular_diffusion_reg` | 1000 steps, dropout 0.2 | the only diffusion setting that passed Track 1's memorization guard |

**One environment.** The five original arms ran on 2026-09-24 and the four new ones on 2026-10-21. To check they are comparable, seed 42 of `independent_marginals`, `gaussian_copula` and `tvae` was rerun on 2026-10-21: all 190 numbers in each result file were identical to the 24 September files, TVAE included (torch 2.14.0 then, 2.12.0 now). The generator code for the original arms did not change between the two dates (only the diffusion registry entry and the sweep script were added), and the cleaned dataset is byte-identical. Environment: Python 3.11.9, numpy 2.4.6, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.17.1, ctgan 0.12.1, rdt 1.22.0.

**Calibration for the new arms** was run into its own file and reuses the shadow fits the evaluation cached, so its AUROCs equal the evaluation's exactly (for example `tabular_diffusion` Gower 0.787 in both). The seven-arm calibration in [`positive_control_result.md`](positive_control_result.md) is unchanged.

What this does not show:

- **The dataset is tiny.** 94 train rows, a 35-row holdout with 6 positives. Seeds resample generators, not patients, and every p-value is a Welch test over seeds, uncorrected.
- **Utility above the real-data ceiling is not real utility.** Both diffusion arms score above train-on-real, and the reason is below.
- **The neural configs are Track 1's in-train choices**, tuned once, on this one dataset.

**Honest phrasing for the report: "across nine configurations of three families on 94 training rows, the statistical baseline is the only family that is faithful without copying; both neural families fail in one of two ways, by learning no dependence (CTGAN) or by reproducing training patients (TVAE and diffusion), and the diffusion model that copies leaks more than any other arm; utility cannot rank the arms, because the highest scores belong to arms that copy or that generate the wrong rate of deaths."**

---

## Results (mean ± sd over 20 seeds)

| Arm | Mean JSD ↓ | Corr. diff ↓ | TSTR LR ↑ | TSTR RF ↑ | Worst MIA ↓ | Gower TPR at 1% FPR (chance 0.01) |
|---|---|---|---|---|---|---|
| `independent_marginals` | 0.0162 ± 0.0007 | 0.1629 ± 0.0026 | 0.493 ± 0.196 | 0.480 ± 0.174 | 0.638 ± 0.016 | 0.014 |
| `gaussian_copula` | 0.0165 ± 0.0009 | 0.1376 ± 0.0037 | 0.587 ± 0.178 | 0.599 ± 0.202 | 0.645 ± 0.019 | 0.019 |
| `gaussian_copula_shrink025` | 0.0165 ± 0.0010 | 0.1098 ± 0.0045 | 0.634 ± 0.201 | 0.684 ± 0.183 | 0.651 ± 0.020 | 0.038 |
| `gaussian_copula_shrink010` | 0.0161 ± 0.0010 | **0.1020 ± 0.0048** | 0.662 ± 0.193 | 0.711 ± 0.157 | 0.656 ± 0.018 | 0.043 |
| `ctgan` | 0.0937 ± 0.0127 | 0.1653 ± 0.0019 | 0.473 ± 0.153 | 0.546 ± 0.132 | 0.637 ± 0.016 | 0.013 |
| `ctgan_small` | 0.0724 ± 0.0065 | 0.1690 ± 0.0029 | 0.529 ± 0.148 | 0.536 ± 0.163 | **0.632 ± 0.014** | 0.010 |
| `tvae` | 0.1063 ± 0.0039 | 0.1224 ± 0.0028 | 0.670 ± 0.144 | 0.738 ± 0.096 | 0.680 ± 0.018 | 0.058 |
| `tabular_diffusion` | 0.0422 ± 0.0082 | 0.1743 ± 0.0276 | 0.714 ± 0.129 | **0.810 ± 0.067** | **0.787 ± 0.013** | **0.358** |
| `tabular_diffusion_reg` | **0.3778 ± 0.0276** | 0.1603 ± 0.0124 | **0.753 ± 0.110** | 0.793 ± 0.092 | 0.706 ± 0.012 | 0.208 |

Real-data references: train-on-real AUROC 0.578 (logistic regression) and 0.737 (random forest); real train against real holdout, mean JSD 0.099 and correlation diff 0.203.

### Privacy, by attack (shadow-model AUROC)

| Arm | 4 numeric | Gower | ICD-9 codes | Codes seen once (diag.) | Gower TPR at 5% FPR |
|---|---|---|---|---|---|
| Ceiling (exact copy) | 1.000 | 1.000 | 1.000 | 0.992 | 1.000 |
| Floor (unseen real rows) | 0.502 | 0.503 | 0.489 | 0.481 | 0.045 |
| `gaussian_copula_shrink010` | 0.523 | 0.629 | 0.656 | 0.837 | 0.134 |
| `ctgan_small` | 0.507 | 0.543 | 0.632 | 0.836 | 0.062 |
| `tvae` | 0.503 | 0.649 | 0.680 | 0.821 | 0.187 |
| `tabular_diffusion` | 0.574 | **0.787** | **0.739** | 0.841 | **0.475** |
| `tabular_diffusion_reg` | 0.507 | 0.704 | 0.689 | **0.853** | 0.332 |

The other four arms are in [`positive_control_result.md`](positive_control_result.md). Worst-case order (the same groups, in the same order, on Gower and ICD-9): `tabular_diffusion` (0.787) > `tabular_diffusion_reg` (0.706) > `tvae` (0.680) > the copulas (0.645 to 0.656) > `independent_marginals` (0.638) > the CTGAN arms (0.632 to 0.637).

### Where the leakage comes from

| Arm | Distinct train rows nearest to some synthetic row (of 94) | Most synthetic rows on one train row | Per-record exposure vs once-seen codes (Spearman) |
|---|---|---|---|
| `gaussian_copula_shrink010` | 46.9 ± 2.6 | 7.5 ± 1.8 | 0.121 |
| `ctgan_small` | 39.2 ± 3.9 | 12.3 ± 2.9 | 0.371 |
| `tvae` | 31.2 ± 3.6 | 24.0 ± 4.9 | −0.011 |
| `tabular_diffusion` | **50.5 ± 4.1** | 7.3 ± 2.3 | **−0.530** |
| `tabular_diffusion_reg` | **24.9 ± 3.6** | 18.2 ± 5.0 | −0.114 |

### Synthetic mortality rate (20 seeds; real train 0.362, 34 of 94)

| Arm | Share of synthetic rows with `hospital_expire_flag` = 1 |
|---|---|
| `independent_marginals`, the three copulas | 0.361 to 0.367 |
| `ctgan`, `ctgan_small` | 0.378, 0.389 |
| `tvae` | **0.210 ± 0.041** (range 0.15 to 0.30) |
| `tabular_diffusion` | 0.446 ± 0.089 (range 0.24 to 0.59) |
| `tabular_diffusion_reg` | **0.787 ± 0.086** (range 0.62 to 0.91) |

### Attribute inference member gap (read against `independent_marginals`)

| Target | `independent_marginals` | `gaussian_copula_shrink010` | `ctgan_small` | `tvae` | `tabular_diffusion` | `tabular_diffusion_reg` |
|---|---|---|---|---|---|---|
| `gender` | 0.042 | 0.148 (p = 0.008) | −0.018 | 0.183 (p = 0.0004) | **0.286** (p < 0.0001) | 0.038 |
| `first_careunit` | −0.005 | 0.013 | −0.001 | 0.008 (p = 0.042) | **0.142** (p < 0.0001) | 0.011 (p = 0.008) |
| `age_bucket` | −0.019 | 0.058 (p = 0.016) | −0.017 | 0.041 (p = 0.014) | **0.221** (p < 0.0001) | 0.024 |

p-values shown only below 0.05, Welch against `independent_marginals`, uncorrected across 24 tests.

### Pareto frontier

| Arm | Fidelity (JS) | Utility (mean TSTR) | Privacy (worst MIA) | On frontier | Bootstrap frequency | Frequency without the utility axis |
|---|---|---|---|---|---|---|
| `independent_marginals` | 0.0162 | 0.486 | 0.638 | yes | 0.98 | 0.97 |
| `gaussian_copula` | 0.0165 | 0.593 | 0.645 | yes | 0.90 | 0.28 |
| `gaussian_copula_shrink025` | 0.0165 | 0.659 | 0.651 | yes | 0.84 | 0.15 |
| `gaussian_copula_shrink010` | 0.0161 | 0.686 | 0.656 | yes | 0.94 | 0.61 |
| `ctgan` | 0.0937 | 0.509 | 0.637 | **no** | 0.34 | 0.13 |
| `ctgan_small` | 0.0724 | 0.532 | 0.632 | yes | 0.98 | 0.90 |
| `tvae` | 0.1063 | 0.704 | 0.680 | yes | 0.59 | 0.00 |
| `tabular_diffusion` | 0.0422 | 0.762 | 0.787 | yes | 0.96 | **0.00** |
| `tabular_diffusion_reg` | 0.3778 | 0.773 | 0.706 | yes | 0.98 | **0.00** |

The frequencies in the last column come from the sensitivity run (1,000 resamples). With a correlation axis added, `gaussian_copula_shrink010` is on the frontier in every resample (1.00).

| Reading | Evidence |
|---|---|
| The diffusion model that copies leaks more than any other arm | Worst case 0.787 against 0.632 to 0.706 for every other arm, p < 0.0001 against each; Gower TPR at 1% FPR 0.358, 36× chance and 6× TVAE's 0.058; the largest attribute gap of any arm on all three targets (gender 0.286 against the floor's 0.042, p < 0.0001) |
| Passing Track 1's in-train guard did not make diffusion private | `tabular_diffusion_reg` passed the guard (near-copy 0.084) and still has the second-highest worst case, 0.706, above TVAE's 0.680 (p < 0.0001) |
| The two diffusion arms leak in different ways | The 3000-step arm spreads over the most training rows (50.5 of 94) and its exposure is highest on records without rare codes (Spearman −0.53): it reproduces ordinary patients near-exactly. The regularized arm collapses onto the fewest (24.9) |
| Both diffusion arms "beat" real data on utility, for the wrong reasons | RF TSTR 0.810 and 0.793 against the 0.737 ceiling; they generate 0.45 and 0.79 deaths against a real 0.36. AUROC is rank-based, so a wrong prior still ranks the holdout, and on 6 holdout positives the ceiling itself is noisy |
| TVAE's utility is copied signal with the wrong prior too | It generates 0.21 deaths against 0.36, a distortion not reported before, and still matches the RF ceiling (0.738) |
| The tuned copula is the best trade-off | Best correlation of any arm (0.102, better than the 0.25 copula, p < 0.0001); utility 0.686, not distinguishable from TVAE (p = 0.68) or the diffusion arm (p = 0.08); worst case 0.656, below TVAE and both diffusion arms though above both CTGAN arms; marginal fidelity equal to the floor (0.0161 vs 0.0162, p = 0.69) |
| Less copula shrinkage buys correlation at little privacy cost | 0.10 against 0.25: Gower 0.629 vs 0.617 (p = 0.06), worst case 0.656 vs 0.651 (p = 0.38), TPR at 1% 0.043 vs 0.038 (p = 0.51) |
| Tuning CTGAN improved its marginals and nothing else | JSD 0.072 vs 0.094 (p < 0.0001); correlation 0.169 vs 0.165, worse, and above the floor's 0.163 (p < 0.0001), so still no dependence learned; utility and privacy unchanged (p = 0.56, 0.29). It dominates the original CTGAN in 62% of resamples, which is why CTGAN is the one arm off the frontier |
| Eight of nine arms on the frontier does not rank them | The three copying arms (TVAE and both diffusion) fall to 0.00 without the utility axis, the axis their copied signal and wrong priors inflate |
| Every generator that emits codes leaks through codes seen once | 0.821 to 0.853 for all nine arms against a 0.481 floor, CTGAN arms included |

---

## Raw evidence

```
$ python -m evaluation.pareto
arm                          seeds       fidelity JS      utility TSTR   privacy worst MIA  frontier  boot freq
ctgan                           20 0.0937 +/- 0.0127 0.5094 +/- 0.1167   0.6370 +/- 0.0156        no       0.34
ctgan_small                     20 0.0724 +/- 0.0065 0.5323 +/- 0.1287   0.6320 +/- 0.0138       yes       0.98
gaussian_copula                 20 0.0165 +/- 0.0009 0.5932 +/- 0.1639   0.6448 +/- 0.0188       yes       0.90
gaussian_copula_shrink010       20 0.0161 +/- 0.0010 0.6864 +/- 0.1639   0.6564 +/- 0.0181       yes       0.94
gaussian_copula_shrink025       20 0.0165 +/- 0.0010 0.6586 +/- 0.1768   0.6511 +/- 0.0197       yes       0.84
independent_marginals           20 0.0162 +/- 0.0007 0.4861 +/- 0.1684   0.6380 +/- 0.0156       yes       0.98
tabular_diffusion_reg           20 0.3778 +/- 0.0276 0.7730 +/- 0.0793   0.7059 +/- 0.0124       yes       0.98
tabular_diffusion               20 0.0422 +/- 0.0082 0.7619 +/- 0.0857   0.7866 +/- 0.0132       yes       0.96
tvae                            20 0.1063 +/- 0.0039 0.7042 +/- 0.0949   0.6798 +/- 0.0175       yes       0.59

Sensitivity: bootstrap frequency of being on the frontier under other axis choices
arm                            main  + correlation  utility = LR  utility = RF  privacy = 4 numeric  privacy = Gower  privacy = ICD-9  no utility axis
ctgan                          0.33           0.82          0.19          0.61                 0.41             0.66             0.34             0.13
ctgan_small                    0.98           0.98          0.97          0.98                 0.65             0.96             0.99             0.90
gaussian_copula                0.90           0.93          0.89          0.92                 0.64             0.99             0.92             0.28
gaussian_copula_shrink010      0.95           1.00          0.94          0.96                 0.94             0.93             0.94             0.61
gaussian_copula_shrink025      0.83           0.90          0.78          0.89                 0.73             0.92             0.83             0.15
independent_marginals          0.97           0.97          0.98          0.98                 0.83             1.00             0.98             0.97
tabular_diffusion_reg          0.98           0.97          0.95          0.95                 1.00             0.97             0.98             0.00
tabular_diffusion              0.97           0.96          0.81          1.00                 0.97             0.97             0.95             0.00
tvae                           0.57           0.56          0.45          0.68                 0.96             0.59             0.56             0.00
```

```
$ python -m evaluation.synthetic_coverage   (new arms)
gaussian_copula_shrink010                    46.9 +/-  2.6                       7.5 +/-  1.8
ctgan_small                                  39.2 +/-  3.9                      12.3 +/-  2.9
tabular_diffusion                            50.5 +/-  4.1                       7.3 +/-  2.3
tabular_diffusion_reg                        24.9 +/-  3.6                      18.2 +/-  5.0
```

```
Welch tests (two-sided, uncorrected, 20 seeds each)
worst case   tabular_diffusion vs each of the other eight        p < 0.0001
             tabular_diffusion_reg vs tvae                       p < 0.0001
             tvae vs gaussian_copula_shrink010                   p = 0.0002
utility      gaussian_copula_shrink010 vs tvae                   p = 0.68   (RF alone p = 0.51)
             gaussian_copula_shrink010 vs tabular_diffusion      p = 0.078
             gaussian_copula_shrink010 vs gaussian_copula        p = 0.080
             tabular_diffusion vs tvae                           p = 0.050
correlation  gaussian_copula_shrink010 vs shrink025              p < 0.0001
             ctgan_small vs ctgan (worse)                        p < 0.0001
fidelity     ctgan_small vs ctgan                                p < 0.0001
             gaussian_copula_shrink010 vs independent_marginals  p = 0.69
```

---

## Against Track 1's run of the same comparison

[`final_comparison_result.md`](final_comparison_result.md) ran the same nine arms through the same harness on Track 1's machine. The two agree on every conclusion and on most numbers to the second or third decimal:

| Arm | Mean JSD (T1 / T2) | Mean TSTR (T1 / T2) | Worst MIA (T1 / T2) |
|---|---|---|---|
| `independent_marginals` | 0.0162 / 0.0162 | 0.487 / 0.486 | 0.640 / 0.638 |
| `gaussian_copula` | 0.0164 / 0.0165 | 0.593 / 0.593 | 0.646 / 0.645 |
| `gaussian_copula_shrink025` | 0.0164 / 0.0165 | 0.649 / 0.659 | 0.651 / 0.651 |
| `gaussian_copula_shrink010` | 0.0167 / 0.0161 | 0.678 / 0.686 | 0.657 / 0.656 |
| `ctgan` | 0.0931 / 0.0937 | 0.506 / 0.509 | 0.635 / 0.637 |
| `ctgan_small` | 0.0727 / 0.0724 | 0.555 / 0.532 | 0.635 / 0.632 |
| `tvae` | 0.1070 / 0.1063 | 0.677 / 0.704 | 0.681 / 0.680 |
| `tabular_diffusion` | 0.0422 / 0.0422 | 0.754 / 0.762 | 0.787 / 0.787 |
| `tabular_diffusion_reg` | 0.3767 / 0.3778 | 0.775 / 0.773 | 0.706 / 0.706 |

The privacy order is identical, and so are both diffusion arms' worst cases. The two runs do not give identical output, even for the deterministic copulas, so something in the two environments differs; it was not identified, and Track 2's own results do reproduce byte for byte across its two run dates. Two frontier readings move with these small differences and should not be leaned on: TVAE's frontier frequency (0.37 there, 0.59 here) and the tuned copula's frequency without the utility axis (0.04 there, 0.61 here). The second turns on whether its mean JSD is just above or just below the floor's, a gap of 0.0001 that is not significant in either run (p = 0.69 here).
