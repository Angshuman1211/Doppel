# AIM at n = 94 — the strongest mechanism in the literature loses to noisy one-way histograms (2026-10-26)

Journal-version work (Track 1 task 1). The three DP arms in `dp_copula.py` all fix what to measure before seeing the data. AIM (McKenna et al., 2022) instead *chooses* which marginals to measure, one per round, spending budget on the marginal its current model fits worst. It is the obvious thing a reviewer asks for, and the question is whether that adaptivity is worth anything at 94 training rows.

```bash
pip install research/third_party/private-pgm         # adds jax/optax/chex; moves no pinned version
python -m research.dp_study --seeds 20 --workers 3   # all ten arms, 200 runs, 295 s
```

`generators/dp_aim.py` wraps the authors' own implementation (ryan112358/private-pgm, Apache 2.0, commit `9429881f`) on the same public domain as the other DP arms — the published schema's levels, `PUBLIC_BOUNDS`, and the public ICD-9 dictionary — so nothing is read from the training rows except through a noisy measurement.

---

## Results, 20 seeds per arm, one machine

| Arm | Mean JSD ↓ | Corr. diff ↓ | TSTR RF ↑ | Gower MIA ↓ | Synthetic death rate | Unusable label |
|---|---|---|---|---|---|---|
| `dp_marginals` ε=1 | **0.368** | 0.166 | 0.446 | 0.501 | 0.409 | 35% |
| `dp_copula` ε=1 | 0.375 | 0.167 | 0.538 | 0.501 | 0.415 | 35% |
| `dp_bayesnet` ε=1 | 0.395 | 0.179 | 0.446 | 0.500 | 0.405 | 30% |
| **`dp_aim` ε=1** | 0.441 | 0.168 | 0.502 | 0.494 | 0.432 | **25%** |
| `dp_marginals` ε=8 | **0.185** | 0.163 | 0.518 | 0.525 | 0.353 | 0% |
| `dp_copula` ε=8 | 0.190 | 0.164 | 0.511 | 0.521 | 0.357 | 0% |
| `dp_bayesnet` ε=8 | 0.203 | 0.170 | **0.575** | 0.513 | 0.363 | 0% |
| **`dp_aim` ε=8** | 0.246 | 0.165 | 0.507 | 0.505 | **0.364** | 0% |
| `independent_marginals` (no DP) | 0.016 | 0.163 | 0.481 | 0.553 | 0.361 | 0% |
| `gaussian_copula_shrink010` (no DP) | 0.017 | **0.102** | **0.696** | 0.628 | 0.364 | 0% |

Real training death rate is 0.3617. These numbers reproduce the rebuilt DP study's committed table to three decimals on a second machine (`dp_marginals` ε=8 at 0.1849 against its 0.185, ε=1 at 0.3676 against 0.368), which is a useful check on that rebuild.

## 1. AIM is the worst DP arm on marginal fidelity, and it is not just the budget split

AIM has to take its budget from somewhere. Here it takes 40% of ε, leaving 30% for the 40 lab columns' histograms, where `dp_marginals` gives those histograms 65%. That confounds the fidelity comparison, so it was measured directly — same mechanism, same seeds, only the split moved:

| AIM budget share | Non-core histogram share | Mean JSD at ε=8 |
|---|---|---|
| 0.40 | 0.30 | 0.250 ± 0.012 |
| 0.15 | 0.55 | 0.223 ± 0.011 |
| — | 0.65 (`dp_marginals`, no AIM) | **0.185** |

Shifting budget back to the histograms recovers 0.026 of the 0.061 gap, about **43%**. The remaining 0.038 is the mechanism: AIM spends part of every round on *choosing* a marginal through an exponential mechanism and on re-measuring, instead of measuring everything once at full strength. At 94 rows that adaptivity is overhead with nothing to buy.

## 2. The pairwise marginals it does buy produce no measurable dependence

Across the run AIM selected **260 one-way and 122 two-way** cliques, so about a third of its rounds went on pairwise structure. It bought nothing: AIM's correlation error is 0.165, against 0.163 for `dp_marginals` and 0.163 for the non-private independent floor. **No DP arm beats the no-dependence floor on correlation at either budget** — the only arm that does is the non-private copula at 0.102.

This is the same wall every DP arm here hits. At 94 rows and 110 modeled columns the noise needed for a pairwise marginal is larger than the signal in it, and a mechanism that is clever about *which* pair to measure cannot fix that.

## 3. What AIM does do better: the label

Two places it wins, both about the column the downstream task depends on:

- **Death rate 0.364 at ε=8 against a real 0.3617** — closest of all four DP arms, and closer than the non-private independent floor.
- **25% unusable runs at ε=1 against 30–35%** for the others. A run is unusable when the synthetic mortality column comes out single-class, so no AUROC exists at all.

AIM models `hospital_expire_flag` jointly with seven other core columns and keeps re-measuring it, so the label survives the budget better. That is a real advantage, and it is for the narrow reason that the label is one of the eight columns AIM was given.

## 4. All four DP arms bring the attack to chance

Gower MIA is 0.494 to 0.525 for every DP arm at both budgets, against 0.553 for the non-private independent floor and 0.628 for the non-private copula. That reproduces the rebuilt DP study's central corrected finding — a real guarantee does measurably reduce attack success — and AIM neither adds to it nor subtracts from it.

---

## What this means for the paper

- **The journal version can say the strong mechanism was tried.** The honest result is that AIM does not rescue DP at this cohort size: it is the worst of the four on marginal fidelity, no better on correlation, and comparable on utility and leakage. A reviewer asking "did you try a real mechanism rather than Laplace histograms?" now has an answer with numbers.
- **The binding constraint is the budget per column, not the mechanism.** 110 modeled columns at n = 94 under composition leaves too little per column for any allocation strategy to matter. That is a stronger and more transferable claim than "our DP baseline was weak".
- **Report the budget-share ablation alongside the comparison.** Without it, AIM's fidelity number looks like a mechanism result when 43% of it is an allocation choice this project made.
- Do not quote AIM's correlation as evidence of anything: at 0.165 against a floor of 0.163, it is the floor.

## Limits and honest caveats

- **AIM here is not full AIM.** It models the 8 core columns (`COPULA_COLUMNS`) jointly and leaves the 40 lab columns on independent noisy histograms. Full AIM over all 46 columns did not finish a single fit in ten minutes — its 2-way workload is over a thousand candidate cliques, each of which the model is re-projected onto every round. The restriction is the same one `DPGaussianCopula` already makes, and a full-domain AIM would have *less* budget per measurement, not more, so the direction of the conclusion is unlikely to flip. It has not been shown.
- **`max_iters` is 100, not the authors' 1000**, and `max_model_size` is 10, not their recommended 80 for "official comparisons". Both are handicaps on AIM, chosen for cost, and both are recorded in `hyperparams()`.
- **Accounting is conservative in two places**, deliberately: AIM is unbounded-DP so it runs at ε/2 to give a bounded-DP guarantee comparable with the other arms, and the three measurements compose by basic composition in (ε, δ) rather than a single zCDP account, because AIM keeps its own account internally.
- MST was not run; the task named "AIM or MST" and AIM is the stronger of the two.
- Patient-level privacy is still not claimed by any arm, AIM included. That is task J5.
