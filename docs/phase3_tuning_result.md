# Phase 3 Tuning Result — the diffusion arm exists, and tuning doesn't rescue either neural family (2026-10-08)

Raw evidence for the Phase 3 tuning decision (ADR-029). Reproduce with `python -m generators.sweep --seeds 0 1 2` (18 configs, 270 fits, 637 s on 6 workers) and `python -m generators.sweep --grid diffusion_refine --seeds 0 1 2` (8 configs, 120 fits).

---

## Method (honest framing)

Same harness as Phase 2 (ADR-021), unchanged: 5 patient folds, fidelity against the fold's fit rows with Track 2's `run_fidelity_report`, memorization against the unseen fifth. The holdout is never read. Selection guard: `dcr_ratio ≥ 0.95` and `near_copy_rate ≤ 0.10`, where real unseen rows score 1.0 and 0.059.

These are in-sample selection numbers, not benchmark numbers. The benchmark is Track 2's evaluation against the holdout ([`B3_full_evaluation.md`](B3_full_evaluation.md)).

The grid tested the three questions ADR-024 left open: whether the copula keeps improving below shrinkage 0.25, whether the neural defaults are simply too big for 75 rows, and how the new diffusion arm behaves.

---

## Results, mean ± sd over 3 seeds

| Generator | Config | DCR ratio | Near-copy | Mean JSD | Corr. diff | Guard |
|---|---|---|---|---|---|---|
| `independent_marginals` | (floor) | 1.099 ± 0.009 | 0.005 | 0.020 | 0.175 | pass |
| `gaussian_copula` | Ledoit-Wolf | 1.073 ± 0.007 | 0.007 | 0.020 | 0.148 | pass |
| `gaussian_copula` | λ = 0.25 | 1.037 ± 0.014 | 0.021 | 0.020 | 0.119 | pass |
| `gaussian_copula` | λ = 0.15 | 1.030 ± 0.015 | 0.029 | 0.020 | 0.114 | pass |
| `gaussian_copula` | **λ = 0.10** | 1.025 ± 0.011 | 0.035 | 0.020 | **0.111** | pass |
| `ctgan` | 300 epochs (default net) | 1.175 ± 0.011 | 0.001 | 0.103 | 0.177 | pass, below floor |
| `ctgan` | **300 epochs, 64×64, emb 32** | 1.139 ± 0.019 | 0.002 | **0.076** | 0.182 | pass, below floor |
| `ctgan` | 1000 epochs, 64×64, emb 32 | 1.153 ± 0.017 | 0.003 | 0.076 | 0.180 | pass, below floor |
| `tvae` | 50 epochs | 0.607 ± 0.006 | 0.949 | 0.233 | 0.169 | **fail** |
| `tvae` | 100 epochs | 0.625 ± 0.008 | 0.844 | 0.193 | 0.149 | **fail** |
| `tvae` | 300 epochs, 64×64 | 0.680 ± 0.007 | 0.647 | 0.130 | 0.142 | **fail** |
| `tvae` | 300 epochs, l2scale 1e-3 | 0.716 ± 0.012 | 0.543 | 0.117 | 0.134 | **fail** |
| `tabular_diffusion` | 300 steps | 1.597 ± 0.052 | 0.000 | 0.398 | 0.174 | pass, underfit |
| `tabular_diffusion` | 1000 steps | 0.867 ± 0.021 | 0.291 | 0.121 | 0.172 | **fail** |
| `tabular_diffusion` | 3000 steps | 0.532 ± 0.010 | 0.804 | **0.036** | 0.156 | **fail** |
| `tabular_diffusion` | 1000 steps, hidden 64 | 2.261 ± 0.294 | 0.001 | 0.425 | 0.222 | pass, underfit |
| `tabular_diffusion` | 3000 steps, hidden 64 | 1.735 ± 0.129 | 0.001 | 0.443 | 0.246 | pass, underfit |
| `tabular_diffusion` | 3000 steps, dropout 0.3 | 1.182 ± 0.046 | 0.075 | 0.416 | 0.150 | pass, underfit |

**Copula.** The trade-off from Phase 2 continues all the way down: λ = 0.10 gives the best correlation of any arm in the project (0.111) while near-copy stays at 0.035, below the 0.059 that real unseen rows score. Marginals stay exact at every setting. Below 0.10 is untested, and λ = 0 is not available, since the matrix is singular at d > n.

**CTGAN.** Shrinking the network was the right guess for marginals and the wrong one for structure. Cutting the generator and discriminator to 64×64 and the embedding to 32 improves mean JSD from 0.103 to 0.076, a quarter better, and more training on top of that adds nothing (0.076 at both 300 and 1000 epochs). Correlation does not move: 0.177 to 0.182 against the independent floor's 0.175. After tuning, CTGAN still produces data with no learned dependence between columns.

**TVAE.** Every setting fails, and the failure is not a training-length artifact. Training *less* makes it worse, not better: at 50 epochs 94.9% of rows are near-copies. The strongest regularization tested (l2scale 1e-3) only brings it to 0.543, five times the guard. Its correlation numbers (0.134 to 0.169) are produced by copying, which is the case the guard exists to catch.

**Diffusion.** The new arm has the sharpest trade-off of the four. At 3000 steps it reaches mean JSD 0.036, by far the best of any neural arm and within touching distance of the copula's 0.020, but 80.4% of its rows are near-copies. Dropout 0.3 pushes it to the clean side of the guard (near-copy 0.075) but fidelity collapses to 0.416. Shrinking the network to hidden 64 underfits without helping memorization, the same shape of answer CTGAN gave. The in-between region is probed in the refinement sweep below.

---

## Diffusion refinement

The main grid left the arm with nothing between "memorizes" (3000 steps, near-copy 0.804) and "underfits" (dropout 0.3, JSD 0.416). Eight configs probed that gap: milder dropout, middling training length, a wider-but-not-huge network, and weight decay as an alternative regularizer.

| Config | DCR ratio | Near-copy | Mean JSD | Corr. diff | Guard |
|---|---|---|---|---|---|
| 1000 steps, dropout 0.2 | 1.159 ± 0.047 | **0.084** | 0.374 | 0.168 | **pass** |
| 3000 steps, hidden 128, dropout 0.2 | 1.504 ± 0.110 | 0.004 | 0.423 | 0.157 | pass, underfit |
| 3000 steps, dropout 0.2 | 1.026 ± 0.034 | 0.207 | 0.371 | 0.159 | fail |
| 1000 steps, dropout 0.1 | 0.997 ± 0.047 | 0.200 | 0.303 | 0.160 | fail |
| 3000 steps, hidden 128 | 1.005 ± 0.010 | 0.198 | 0.213 | 0.178 | fail |
| 3000 steps, dropout 0.1 | 0.721 ± 0.026 | 0.509 | 0.228 | 0.159 | fail |
| 1500 steps | 0.697 ± 0.007 | 0.552 | **0.071** | 0.174 | fail |
| 3000 steps, weight decay 0.01 | 0.545 ± 0.009 | 0.796 | 0.036 | 0.156 | fail |

The gap doesn't contain a usable config. Across all 14 diffusion settings tested, every config that clears the memorization guard scores mean JSD 0.374 or worse, and every config with fidelity better than 0.23 copies more than a fifth of its rows. The best guard-passing config, 1000 steps with dropout 0.2, has worse marginals than independent sampling (0.374 vs 0.020) and correlation at the no-dependence floor (0.168 vs 0.175). Weight decay at 0.01 changed nothing at all against the unregularized run (0.545 / 0.796 / 0.036 vs 0.532 / 0.804 / 0.036), so dropout is the only lever of the two that moves this model.

---

## What this means

On 94 training rows, all three neural families fail, in two distinct ways, and tuning does not move either failure:

- **CTGAN underfits.** Smaller networks improved its marginals by a quarter, and its correlation stayed at the no-dependence floor at every size and length tested.
- **TVAE memorizes.** Every setting, including the strongest regularization and the shortest training, leaves at least 54% of rows as near-copies.
- **Diffusion does both, depending on where you stand on one knob.** It produces the best neural fidelity in the project (0.036) while copying 80% of its rows, and the only settings that stop the copying destroy the fidelity.

The copula is unbeaten on both axes at the same time, and its best setting (λ = 0.10, correlation 0.111) is also its most aggressive one tested. This is a small-data result, not a claim about these architectures in general: 94 rows and 110 modeled columns is well outside the regime any of them were designed for.


---

## Raw evidence

```
$ python -m generators.sweep --seeds 0 1 2
270 tasks on 6 workers
...
wrote output/sweeps/sweep_20261007T193339Z.csv and _summary.csv  (637s)

$ python -m generators.sweep --grid diffusion_refine --seeds 0 1 2
120 tasks on 6 workers
...
wrote output/sweeps/sweep_20261007T202137Z.csv and _summary.csv  (454s)
```

Single full-train run of the new arm, seed 42, for reference: mean JSD 0.042, correlation difference 0.189, 39 of 65 primary ICD-9 codes present, `n_diagnoses` mean 12.9 against 13.0 real, mortality 0.30 against 0.36 real, and fold-0 memorization `dcr_ratio` 0.561 with near-copy 0.779.
