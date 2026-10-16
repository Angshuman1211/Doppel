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
- **Utility**: rises with correlation inside the copula family (0.487 → 0.678), which is the one place the three axes behave as the blueprint assumed. The highest scores belong to arms that are copying or mis-calibrated, see below.
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

## Reading guide for the report

1. **Fidelity**: the copula arms are an order of magnitude ahead of every neural arm.
2. **Utility**: the highest numbers belong to arms that are either copying (TVAE, `tabular_diffusion`) or generating the wrong label prior (`tabular_diffusion_reg`), so utility must never be read alone.
3. **Privacy**: the memorizing arms carry the highest membership risk, which is the expected direction and confirms the attack is doing its job.
4. **The trade-off**: across families, fidelity and privacy move against each other, and the classical baseline escapes the trade-off only because it never had the capacity to memorize.
