# Membership Attack Across Patient Splits — the privacy ordering holds over patients, not just over seeds (2026-10-25)

Post-course extension work, in answer to a review comment (pseudo-replication). Every privacy comparison in the benchmark is a Welch test over 20 generator seeds on one patient split, and a seed changes the generator and the shadow subsets but not the patients. Those p-values (down to p < 10^-20) therefore describe generator variability on one cohort, not uncertainty about the cohort. This reruns the benchmark's attack on redrawn patient splits so that the key privacy orderings are tested over patients, as the utility orderings already were.

```bash
python -m research.split_attack --splits 10 --workers 4     # 50 attack runs, 400 shadow fits, 944 s
```

---

## Method

Ten patient-level splits drawn as in [`split_study.py`](split_study.py) (80% of patients to training, admissions follow their patient, split seeds 1000 to 1009); training sets of 89 to 107 admissions. On each split, for five settings, the benchmark's shadow attack is run unchanged on that split's training admissions: 8 shadow generators on random halves, the 4-column, Gower and ICD-9 code-set distances on the same shadows, and the worst case of the three. One generator seed (42) per split. The four comparisons that carry the paper's privacy ordering are tested with a paired Wilcoxon signed-rank test over the 10 splits and Holm-corrected as one family. With 10 pairs the smallest attainable two-sided p is 0.002.

## Results

Worst-case attack AUROC, mean ± sd over 10 splits:

| Setting | Worst case | Gower | ICD-9 set | Gower TPR at 1% FPR |
|---|---|---|---|---|
| Diffusion (3000 steps) | **0.792 ± 0.017** | 0.792 | 0.708 | **0.334** |
| Diffusion-reg | 0.686 ± 0.023 | 0.684 | 0.641 | 0.167 |
| TVAE | 0.667 ± 0.020 | 0.657 | 0.656 | 0.066 |
| Copula λ=0.10 | 0.647 ± 0.026 | 0.638 | 0.631 | 0.046 |
| Independent | 0.610 ± 0.028 | 0.569 | 0.610 | 0.024 |

Paired comparisons on the worst case:

| Comparison | Mean difference | Splits won | Wilcoxon p | Holm p |
|---|---|---|---|---|
| Diffusion − Diffusion-reg | +0.106 | 10/10 | 0.0020 | 0.0078 |
| Diffusion-reg − TVAE | +0.020 | 8/10 | 0.0098 | 0.0098 |
| TVAE − Copula λ=0.10 | +0.020 | 9/10 | 0.0039 | 0.0078 |
| Copula λ=0.10 − Independent | +0.036 | 10/10 | 0.0020 | 0.0078 |

## What it shows

**The whole privacy ordering of the benchmark holds over patients.** Each adjacent step, diffusion above regularised diffusion above TVAE above the tuned copula above the independent setting, survives Holm correction across 10 patient splits. The levels are close to the reported split's (diffusion 0.792 against 0.787, TVAE 0.667 against 0.680, copula 0.647 against 0.656), and the diffusion model's TPR at 1% FPR averages 33.4% against 35.8% there.

**The effect sizes are what to quote, not the seed p-values.** The diffusion model leaks 0.106 more AUROC than the next setting and the other steps are 0.020 to 0.036. Across splits the attack's standard deviation is 0.017 to 0.028, about the same as its seed standard deviation on one split (0.012 to 0.020), so the patients add little uncertainty to the privacy scores; it is the p-values over seeds, not the orderings, that overstated certainty.

**The independent setting moves with the split.** Its worst case is 0.610 here against 0.638 on the reported split, because the code-set leak depends on how many rare codes the split's training patients carry. This is why every setting is read against it on the same split.

## Limits

Ten splits rather than thirty, one generator seed per split, five of the nine settings, and the splits are redraws of the same 100 patients, so they overlap and the Wilcoxon tests describe robustness to reshuffling this cohort. The copy and unseen-record controls were not rerun per split.
