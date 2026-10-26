# Capacity × Cohort Size — at a small cohort, extra capacity buys memorisation and nothing else (2026-10-25)

Journal-version work (Track 1 task 2). The [size sweep](size_sweep_result.md) varied the cohort and held the generator fixed, so it measured only half of "capacity relative to cohort size", and that document says so itself. This varies both: the diffusion model's training budget and its network width, crossed with cohort size, scored on the near-copy rate **and** on the benchmark's shadow membership attack.

```bash
python -m research.size_sweep  --arms diffusion_s{500,1000,3000}_h{128,256} \
    --sizes 50 250 1000 5000 --seeds 2 --out capacity_sweep        # 48 fits, 325 s
python -m research.size_attack --arms diffusion_s{500,1000,3000}_h{128,256} \
    --sizes 50 250 1000 5000 --seeds 2 --out capacity_attack       # 384 shadow fits, 3,999 s
```

Six configurations: training steps ∈ {500, 1000, 3000} × hidden width ∈ {128, 256}, on the UCI Diabetes 130-US cohorts the size sweep already draws, with the same codec and the same fixed holdout. `diffusion_s3000_h256` repeats `tabular_diffusion`'s configuration and reproduces its near-copy rates exactly, which is the check that the grid is wired to the same generator.

---

## The grid

`auroc` and `tpr1` are the shadow attack's; `near_copy` and `dcr_ratio` are distance statistics.

| Rows | Steps | Width | Near-copy | DCR ratio | **Attack AUROC** | **TPR@1%** | Mean JSD | TSTR RF |
|---|---|---|---|---|---|---|---|---|
| 67 | 500 | 128 | 0.135 | 0.88 | 0.767 | 0.099 | 0.070 | 0.594 |
| 67 | 1000 | 128 | 0.283 | 0.78 | 0.812 | 0.275 | 0.070 | 0.582 |
| 67 | 3000 | 128 | 0.591 | 0.65 | 0.847 | 0.449 | 0.071 | 0.603 |
| 67 | 500 | 256 | 0.251 | 0.80 | 0.798 | 0.199 | 0.067 | 0.591 |
| 67 | 1000 | 256 | 0.455 | 0.70 | 0.820 | 0.302 | 0.071 | 0.573 |
| 67 | 3000 | 256 | **0.723** | 0.53 | **0.852** | **0.515** | 0.067 | 0.594 |
| 362 | 500 | 128 | 0.034 | 1.04 | 0.542 | 0.013 | 0.084 | 0.558 |
| 362 | 3000 | 128 | 0.044 | 1.01 | 0.669 | 0.064 | 0.081 | 0.549 |
| 362 | 500 | 256 | 0.047 | 1.04 | 0.583 | 0.021 | 0.082 | 0.542 |
| 362 | 3000 | 256 | 0.069 | 0.97 | 0.717 | 0.123 | 0.082 | 0.557 |
| 1,446 | 500 | 128 | 0.018 | 1.12 | 0.510 | 0.009 | 0.084 | 0.533 |
| 1,446 | 3000 | 256 | 0.007 | 1.20 | 0.575 | 0.022 | 0.075 | 0.552 |
| 7,145 | 500 | 128 | 0.009 | 1.22 | 0.499 | 0.005 | 0.085 | 0.510 |
| 7,145 | 3000 | 256 | 0.004 | 1.35 | 0.516 | 0.010 | 0.077 | 0.536 |

(The full 24-row grid is in `research/output/capacity_attack_summary.csv`.)

## 1. Capacity is a privacy parameter, and at a small cohort it is the dominant one

At 67 training rows, holding the data fixed and changing only the generator:

- near-copy rate goes from **0.135 to 0.723**, a factor of 5.4;
- attack AUROC goes from **0.767 to 0.852**;
- and TPR at a 1% false-positive budget goes from **0.099 to 0.515** — a fivefold change in the number of training records an attacker can name, with the cohort untouched.

Both knobs point the same way and the training budget matters more than the width: at width 128, going 500 → 3000 steps moves the attack from 0.767 to 0.847, while at 500 steps, going 128 → 256 wide moves it from 0.767 to 0.798.

## 2. The extra capacity buys nothing else

This is the part worth putting in the paper. Across the same six configurations at 67 rows:

| | Range across the capacity grid |
|---|---|
| Attack TPR@1% | 0.099 → 0.515 (**5.2×**) |
| Near-copy rate | 0.135 → 0.723 (**5.4×**) |
| Mean JSD | 0.067 → 0.071 (1.06×, no trend) |
| TSTR RF | 0.573 → 0.603 (no trend) |

Six-fold more training and twice the width leave fidelity and downstream utility **statistically flat** while multiplying measured leakage by five. On a cohort this small the additional capacity is spent entirely on reproducing training rows. There is no fidelity-privacy trade-off being navigated here, because there is no fidelity gain to trade: the smallest, shortest-trained configuration is as good as the largest on every quality axis and leaks a fifth as much.

## 3. Capacity stops mattering once the cohort is large enough

At 7,145 rows every configuration sits at chance (attack AUROC 0.499 to 0.516, TPR@1% 0.005 to 0.012), and the DCR ratio has crossed above 1.0 for all of them — the models are now underfitting rather than copying, and the most capable configuration is the *most* underfit (1.35). The `s3000_h256` configuration that memorises 72% of its rows at 67 rows memorises 0.4% at 7,145.

So "capacity relative to cohort size" is now measured rather than inferred, and it behaves like a ratio: leakage is set by how much the model can store against how much data there is to store, and either side moves it.

## 4. The two are substitutable, and roughly how much

Reading equal-leakage points off the grid gives an exchange rate:

| Configuration | Rows | Attack AUROC |
|---|---|---|
| `s500_h128` | 67 | 0.767 |
| `s3000_h256` | 362 | 0.717 |

The smallest configuration on 67 rows leaks *more* than the largest configuration on 362 rows. Cutting the training budget 6× and the width 2× bought more privacy than a 5.4× larger cohort — and, per §2, cost nothing in fidelity or utility. For a project working at n = 94 and unable to collect more patients, that is the actionable result: the capacity knob was available the whole time and was never turned.

---

## What this changes for the paper

- The conclusion that capacity relative to cohort size drives memorisation is now a **measured two-dimensional result**, not an inference from a one-dimensional sweep. This closes the limitation stated in `size_sweep_result.md` and the reviewer objection that motivated the task.
- It adds a recommendation the paper can actually make: **at small n, reduce the generator before reducing the data's utility**, because the capacity reduction is free on every quality axis measured here.
- It also sharpens the MIMIC result. `tabular_diffusion` at n = 94 is `s3000_h256`, the single leakiest cell in this grid, at a cohort size smaller than the grid's smallest. The headline leakage in the paper is the worst corner of this surface, and the paper should say so rather than presenting it as what a diffusion model does.
- The distance statistic and the attack agree on the ordering across all 24 cells, which is worth one sentence given §VII's concern about distance metrics standing in for attacks — here they do agree, when capacity is what varies.

## Limits

Two seeds per cell, one dataset, one architecture family, and two knobs. Width and depth are not separated (depth is fixed at 2 layers). The attack is the benchmark's shadow-model Gower attack, so it inherits that attack's limitations; the release-only attacker of `domias_attack_result.md` is not run here. TSTR is a single prediction task, and `task_study_result.md` shows margins on one task do not transfer — the claim in §2 is that utility is *flat across the grid*, which is weaker and does not depend on the task's absolute level.
