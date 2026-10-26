# Final Comparison Result — all nine arms through the evaluation harness (2026-10-16)

Raw evidence for the report's Results section. Run by Track 1 because the merged Phase 3 evaluation predates the diffusion family and the tuned configurations, so the headline three-family comparison was missing a family. The harness, protocol and thresholds are Track 2's and unchanged; only `evaluation/arms.py` gained four entries (ADR-029). Rayyan owns the evaluation narrative, so this is evidence for him to fold into [`full_evaluation_result.md`](full_evaluation_result.md), not a replacement for it.

Reproduce from the repo root:

```bash
pip install -r generators/requirements-neural.txt
for arm in independent_marginals gaussian_copula gaussian_copula_shrink025 gaussian_copula_shrink010 \
           ctgan ctgan_small tvae tabular_diffusion tabular_diffusion_reg; do
  for seed in $(seq 42 61); do python -m evaluation.eval_runner --generator $arm --seed $seed; done
done
python -m evaluation.summarize_results
```

---

## Method (honest framing)

Nine arms × 20 seeds (42 to 61) = 180 runs, each fitting the generator plus 8 shadow generators for the membership attack. Every run on one machine and one environment (Python 3.11.5, numpy 2.4.6, torch 2.14.0+cpu, ctgan 0.12.1), so the whole table is internally comparable.

What the numbers are: fidelity against the real **train** split, utility as train-on-synthetic / test-on-real against the 35-row holdout, privacy as the worst of three membership attacks. What they are not: tuned on the holdout (selection happened inside train, ADR-021), and not a claim about these architectures in general, since the training set is 94 rows against 110 modeled columns.

---

## Results (mean ± sd over 20 seeds)

180 runs, no failures. Fidelity and correlation are against real train, utility is TSTR on the 35-row holdout, privacy is the worst of three membership attacks.

| Arm | Mean JSD ↓ | Corr. diff ↓ | TSTR AUROC ↑ | Worst MIA ↓ |
|---|---|---|---|---|
| `independent_marginals` (floor) | 0.0162 ± 0.0007 | 0.1629 ± 0.0026 | 0.4865 ± 0.1641 | 0.6395 ± 0.0198 |
| `gaussian_copula` (Ledoit-Wolf) | 0.0164 ± 0.0008 | 0.1377 ± 0.0035 | 0.5931 ± 0.1449 | 0.6460 ± 0.0205 |
| `gaussian_copula_shrink025` | 0.0164 ± 0.0008 | 0.1099 ± 0.0045 | 0.6485 ± 0.1738 | 0.6512 ± 0.0223 |
| **`gaussian_copula_shrink010`** | 0.0167 ± 0.0011 | **0.1021 ± 0.0046** | 0.6777 ± 0.1638 | 0.6568 ± 0.0151 |
| `ctgan` | 0.0931 ± 0.0112 | 0.1649 ± 0.0021 | 0.5060 ± 0.1263 | **0.6345 ± 0.0133** |
| `ctgan_small` (tuned) | 0.0727 ± 0.0058 | 0.1688 ± 0.0028 | 0.5553 ± 0.1370 | 0.6349 ± 0.0174 |
| `tvae` | 0.1070 ± 0.0063 | 0.1231 ± 0.0037 | 0.6772 ± 0.0858 | 0.6810 ± 0.0154 |
| `tabular_diffusion` | 0.0422 ± 0.0079 | 0.1741 ± 0.0271 | 0.7539 ± 0.0829 | **0.7870 ± 0.0131** |
| `tabular_diffusion_reg` | **0.3767 ± 0.0271** | 0.1604 ± 0.0121 | **0.7747 ± 0.0808** | 0.7059 ± 0.0123 |

- **Fidelity**: the copula family sits an order of magnitude ahead of every neural arm (0.0164 against 0.042 to 0.377), and is indistinguishable from the independent floor on marginals, as expected since both sample empirical marginals.
- **Correlation**: only the copula beats the floor's 0.1629, and it improves monotonically as shrinkage falls, to 0.1021 at λ = 0.10. Both CTGAN arms sit *above* the floor, meaning no dependence was learned at either size.
- **Utility**: rises with correlation inside the copula family on this split (0.487 → 0.678). A later study of 10 redrawn patient-level splits found that difference is exactly zero on average (−0.000, p = 1.00, winning 4 splits of 10), so it is a property of this split rather than of shrinkage, and the report should not claim it ([`research/split_study_result.md`](../research/split_study_result.md)). The highest scores belong to arms that are copying or mis-calibrated, see below.

> **Post-course note (2026-10-23), after the extension studies.** Two findings narrow the utility column further, and neither changes a number in the table above.
> 1. For fidelity the ± column here is 1.5 to 4 times too narrow for the copula family and the floor, because it varies the generator and not the cohort; for utility it is already wider than a bootstrap over the held-out patients ([`research/stats_report_result.md`](../research/stats_report_result.md)). Replacing it with 30 redrawn patient-level splits and Holm correction, the **ordering does hold where it matters**: both diffusion arms and TVAE beat `gaussian_copula_shrink010` and it beats the floor, at adjusted p ≤ 0.010 ([`research/scale_study_result.md`](../research/scale_study_result.md)). But two of the claims in this table do **not** survive — the copula's utility gain from lower shrinkage (p = 0.329) and CTGAN's tuning gain (p = 0.210) — and `tabular_diffusion_reg`'s advantage exists only when logistic regression is averaged in. Quote the corrected paired result, never the single-split number or a ± over seeds.
> 2. The reading in point 2 of the conclusions below — that the best utility belongs to arms that copy — is specific to n = 94. On a 1,446-row cohort the diffusion model stops copying (near-copy 0.007) and still wins every prediction task, while TVAE copies 57% of its rows and loses to it ([`research/task_study_result.md`](../research/task_study_result.md)). The coupling is a small-cohort effect, which is a sharper claim than the general one.
- **Privacy**: ordering is identical on all three attacks: `tabular_diffusion` (0.787) > `tabular_diffusion_reg` (0.706) > `tvae` (0.681) > copula family (0.646 to 0.657) > `independent_marginals` (0.640) > CTGAN arms (0.634). The attack's floor on this dataset is about 0.64, not 0.50, since even a generator with no dependence at all scores 0.640, so the diffusion leak is 0.787 against a floor of 0.64.

## The frontier is held up by its weakest axis

Eight of the nine arms are on the Pareto frontier, which by itself says little. The sensitivity table is the informative part:

| Arm | Frontier freq. (main axes) | With correlation axis | **With no utility axis** |
|---|---|---|---|
| `independent_marginals` | 0.98 | 0.97 | 0.98 |
| `gaussian_copula_shrink010` | 0.82 | **1.00** | 0.04 |
| `ctgan_small` | 0.96 | 0.95 | 0.74 |
| `tabular_diffusion` | 0.97 | 0.95 | **0.00** |
| `tabular_diffusion_reg` | 0.99 | 0.98 | **0.00** |
| `tvae` | 0.37 | 0.38 | 0.00 |

Both diffusion arms hold their frontier place **entirely through the utility axis**: drop it and they fall off completely. That is the axis distorted by the label prior below, so the frontier's headline should not be read as "the diffusion arms are competitive". Add the correlation axis instead and the tuned copula is on the frontier in every bootstrap sample.

TVAE is the only arm that is dominated outright (0.37), which is what a generator that copies without reproducing the distribution looks like.

---

## The diffusion arms generate the wrong number of deaths

Checked after the utility numbers came out implausibly high, over 5 seeds each:

| Arm | Synthetic mortality rate | Real train |
|---|---|---|
| `tabular_diffusion_reg` | **0.81** | 0.36 |
| `tabular_diffusion` | 0.46 | 0.36 |

This matters for how the utility number is read. AUROC is rank-based, so a wrong class prior does not stop a classifier trained on that data from ranking holdout patients well, which is why the regularized arm posts the best TSTR in the project (0.775 ± 0.081) while having the worst fidelity of any arm (mean JSD 0.377). The data is still unusable for anything prior-sensitive: prevalence, calibration, absolute risk. A report that quotes only TSTR would call this the best generator in the study.

Both diffusion arms also score above the train-on-real ceiling (0.737 for random forest). On a holdout with 6 positives that is a statement about the ceiling's noise, not evidence that synthetic data beats real data, and it should be written that way.

---

## The in-train guard predicted the holdout privacy result

Track 1 picked configurations with a memorization check that never touches the holdout (ADR-021). Comparing that check against the membership attack it never saw:

| Arm | In-train near-copy rate | Worst membership AUROC |
|---|---|---|
| `tabular_diffusion` | 0.804 | 0.787 |
| `tvae` | 0.543 | 0.681 |
| `tabular_diffusion_reg` | 0.084 | 0.706 |
| `gaussian_copula_shrink010` | 0.035 | 0.657 |
| `ctgan_small` | 0.002 | 0.635 |

The ordering matches on four of five arms, so a cheap in-train diagnostic anticipated the expensive holdout-side attack. The exception matters: `tabular_diffusion_reg` passes the in-train guard (0.084, under the 0.10 threshold) and still carries the second-highest membership risk. **The guard is necessary but not sufficient**, and the report should say so rather than presenting the in-train check as a privacy guarantee.

## Robustness: the holdout without subject 41976

Subject 41976 holds 15 of the 35 holdout rows, all male, all Puerto Rican, and contributes none of the 6 deaths (Track 3's finding, P-003 and P-011). `generators.schema.clean_holdout()` drops them, leaving 20 rows. Every arm was rerun against it with `--clean-holdout`, into `results/clean_holdout/`, so both variants exist side by side. Fidelity and membership inference are computed against train and are bit-identical between the two; only utility and attribute inference move.

**Utility: conclusions hold.** Every arm shifts up by about the same amount and the spread narrows, because the 15 excluded rows were uniformly hard to rank.

| Arm | TSTR, 35 rows | TSTR, 20 clean rows | Shift |
|---|---|---|---|
| `independent_marginals` | 0.486 ± 0.164 | 0.567 ± 0.141 | +0.080 |
| `gaussian_copula_shrink010` | 0.678 ± 0.164 | 0.758 ± 0.116 | +0.081 |
| `ctgan_small` | 0.555 ± 0.137 | 0.549 ± 0.139 | −0.007 |
| `tvae` | 0.677 ± 0.086 | 0.780 ± 0.077 | +0.102 |
| `tabular_diffusion` | 0.754 ± 0.083 | 0.844 ± 0.056 | +0.090 |
| `tabular_diffusion_reg` | 0.775 ± 0.081 | 0.863 ± 0.047 | +0.088 |

The top two arms are unchanged, and the only reordering is TVAE and the tuned copula swapping third and fourth, which their standard deviations already cover. No finding in this document depends on which variant is used.

**Attribute inference: the 35-row holdout is measuring the skew.** On the full holdout the attacker scores *below* chance on non-members, which is not a privacy property but the 12 F / 23 M imbalance that one patient creates:

| Arm | Gender uplift on non-members, 35 rows | 20 clean rows |
|---|---|---|
| `independent_marginals` | −0.015 | −0.006 |
| `gaussian_copula_shrink010` | −0.063 | +0.004 |
| `tvae` | −0.073 | +0.004 |
| `tabular_diffusion` | −0.067 | −0.019 |

Because the member gap is members minus non-members, that negative baseline inflates it. On the clean holdout the gaps fall accordingly: the tuned copula from 0.161 to 0.094, TVAE from 0.223 to 0.145, the diffusion model from 0.284 to 0.236. The ordering is unchanged, so the diffusion model still leaks most, but the full-holdout numbers overstate attribute leakage by roughly 0.05 to 0.08.

**What to report.** Utility against the 35-row holdout, with this table as the robustness check, since that is what every published number and both dashboards use and nothing changes. Attribute inference against the clean holdout, because the full one measures a known artifact. Fidelity and membership inference are unaffected either way (ADR-032).

## Reading guide for the report

1. **Fidelity**: the copula arms are an order of magnitude ahead of every neural arm.
2. **Utility**: the highest numbers belong to arms that are either copying (TVAE, `tabular_diffusion`) or generating the wrong label prior (`tabular_diffusion_reg`), so utility must never be read alone.
3. **Privacy**: the memorizing arms carry the highest membership risk, which is the expected direction and confirms the attack is doing its job.
4. **The trade-off**: across families, fidelity and privacy move against each other, and the classical baseline escapes the trade-off only because it never had the capacity to memorize.
