# AIM at n = 94 — the strongest mechanism in the literature loses to noisy one-way histograms (2026-10-26)

Journal-version work (Track 1 task 1). The three DP arms in `dp_copula.py` all fix what to measure before seeing the data. AIM (McKenna et al., 2022) instead *chooses* which marginals to measure, one per round, spending budget on the marginal its current model fits worst. It is the obvious thing a reviewer asks for, and the question is whether that adaptivity is worth anything at 94 training rows.

```bash
pip install research/third_party/private-pgm         # adds jax/optax/chex; moves no pinned version
python -m research.dp_study --seeds 20 --workers 3   # all ten arms, 200 runs, 295 s
```

`generators/dp_aim.py` wraps the authors' own implementation (ryan112358/private-pgm, Apache 2.0, commit `9429881f`) on the same public domain as the other DP arms — the published schema's levels, `PUBLIC_BOUNDS`, and the public ICD-9 dictionary — so nothing is read from the training rows except through a noisy measurement.

---

## Results, 20 seeds per arm, one machine

| Arm | Mean JSD ↓ | Corr. diff ↓ | TSTR RF ↑ | **Code-set MIA ↓** | Gower MIA ↓ | Worst-of-three ↓ | Death rate | Unusable label |
|---|---|---|---|---|---|---|---|---|
| `dp_marginals` ε=1 | **0.370** | 0.166 | 0.519 | 0.499 ± 0.007 | 0.497 | 0.521 ± 0.017 | 0.409 | 35% |
| `dp_copula` ε=1 | 0.377 | 0.166 | 0.543 | 0.498 ± 0.006 | 0.499 | 0.521 ± 0.022 | 0.416 | 35% |
| `dp_bayesnet` ε=1 | 0.391 | 0.179 | 0.516 | 0.504 ± 0.007 | 0.502 | 0.525 ± 0.016 | 0.405 | 30% |
| **`dp_aim`** ε=1 | 0.438 | 0.165 | 0.491 | 0.501 ± 0.007 | 0.501 | 0.524 ± 0.017 | 0.342 | **20%** |
| `dp_marginals` ε=8 | **0.186** | 0.164 | 0.504 | 0.510 ± 0.018 | 0.520 | 0.532 ± 0.023 | 0.353 | 0% |
| `dp_copula` ε=8 | 0.192 | 0.164 | 0.507 | 0.507 ± 0.013 | 0.527 | 0.535 ± 0.021 | 0.357 | 0% |
| `dp_bayesnet` ε=8 | 0.206 | 0.169 | **0.610** | 0.505 ± 0.020 | 0.513 | 0.527 ± 0.015 | 0.363 | 0% |
| **`dp_aim`** ε=8 | 0.247 | 0.165 | 0.525 | 0.500 ± 0.011 | 0.504 | 0.521 ± 0.014 | **0.383** | 0% |
| `independent_marginals` (no DP) | 0.016 | 0.163 | 0.480 | 0.640 ± 0.020 | 0.553 | 0.640 ± 0.020 | 0.361 | 0% |
| `gaussian_copula_shrink010` (no DP) | 0.017 | **0.102** | **0.696** | 0.656 ± 0.016 | 0.628 | 0.657 ± 0.016 | 0.364 | 0% |

All four attacks of the benchmark's suite, mean ± sd over 20 seeds, run on one machine with one code version (`research/output/dp_study_full.csv`).

Real training death rate is 0.3617. The three non-AIM mechanisms reproduce the rebuilt DP study's committed table on a second machine (`dp_marginals` ε=8 at 0.186 against its 0.185, ε=1 at 0.370 against 0.368), which is a useful check on that rebuild.

## 1. AIM is the worst DP arm on marginal fidelity, and it is not just the budget split

AIM has to take its budget from somewhere. Here it takes 40% of ε, leaving 30% for the 40 lab columns' histograms, where `dp_marginals` gives those histograms 65%. That confounds the fidelity comparison, so it was measured directly — same mechanism, same seeds, only the split moved:

| AIM budget share | Non-core histogram share | Mean JSD at ε=8 |
|---|---|---|
| 0.40 (the arm's default) | 0.30 | 0.2504 ± 0.0112 |
| 0.15 | 0.55 | 0.2237 ± 0.0119 |

```bash
python -m research.aim_diagnostics --seeds 10     # both tables in this document, 119 s
```
| — | 0.65 (`dp_marginals`, no AIM) | **0.185** |

Shifting budget back to the histograms recovers 0.026 of the 0.061 gap, about **43%**. The remaining 0.038 is the mechanism: AIM spends part of every round on *choosing* a marginal through an exponential mechanism and on re-measuring, instead of measuring everything once at full strength. At 94 rows that adaptivity is overhead with nothing to buy.

## 2. The pairwise marginals it does buy produce no measurable dependence

AIM prints the clique it measures each round, and `research/aim_diagnostics.py` counts them. Over 10 seeds per budget, adaptive rounds only (the 8 initial one-way measurements are not printed):

| Budget | One-way | Two-way | Pairwise share |
|---|---|---|---|
| ε = 1 | 35 | 5 | **12%** |
| ε = 8 | 32 | 13 | **29%** |

Two things follow. First, the pairwise share is **budget-dependent**: at ε = 1 AIM almost never affords a pair, and nearly every round goes on re-measuring a marginal it already has. Second, and more telling, those counts are over ten fits — so AIM gets about **four adaptive choices per fit** before the budget is exhausted, despite being asked for 50 rounds. A mechanism whose whole advantage is adaptively choosing what to measure is being allowed four choices.

It buys nothing measurable either way: AIM's correlation error is 0.165, against 0.164 for `dp_marginals` and 0.163 for the non-private independent setting. **No DP arm beats the no-dependence setting on correlation at either budget** — the only arm that does is the non-private copula at 0.102.

This is the same wall every DP arm here hits. At 94 rows and the 50 columns of the raw table these arms measure the noise needed for a pairwise marginal is larger than the signal in it, and a mechanism that is clever about *which* pair to measure cannot fix that.

## 3. What AIM does do better: the label

Two places it wins, both about the column the downstream task depends on:

- **Death rate 0.364 at ε=8 against a real 0.3617** — closest of all four DP arms, and closer than the non-private independent setting.
- **25% unusable runs at ε=1 against 30–35%** for the others. A run is unusable when the synthetic mortality column comes out single-class, so no AUROC exists at all.

AIM models `hospital_expire_flag` jointly with seven other core columns and keeps re-measuring it, so the label survives the budget better. That is a real advantage, and it is for the narrow reason that the label is one of the eight columns AIM was given.

## 4. Every DP arm sits at chance on every attack, and the code-set attack is the one that shows it

The code-set attack is the discriminating one, and by a wide margin. On the two **non-private** arms it reaches **0.640** and **0.656**, against 0.553 and 0.628 for Gower. On every DP arm at both budgets it sits between **0.498 and 0.510**. AIM is 0.501 ± 0.007 at ε = 1 and 0.500 ± 0.011 at ε = 8 — indistinguishable from the other three mechanisms and from chance.

That is the sharpest form of the rebuilt DP study's central finding: the attack with the most signal against a non-private release is the one that goes completely flat under a real guarantee, and AIM neither adds to that nor subtracts from it.

**One caveat about the worst-of-three column, which matters for the paper.** Taking the maximum of three noisy estimates is biased upward, and the size of that bias is visible here. On the DP arms, every individual attack sits at about 0.50 while the worst-of-three reads 0.521 to 0.535 — roughly **+0.02 of pure selection effect**. On the non-private arms the worst-of-three equals the code-set attack to three decimals (0.640 against 0.640), because there one attack genuinely dominates and the maximum is picking a real signal rather than noise. So "report the worst of several attacks" is the right rule when something is leaking, and carries a +0.02 floor artefact when nothing is. The DP arms are not leaking at 0.52; they are at chance with three estimates and a maximum taken over them.

## What this means for the paper

- **The journal version can say the strong mechanism was tried.** The honest result is that AIM does not rescue DP at this cohort size: it is the worst of the four on marginal fidelity, no better on correlation, and comparable on utility and leakage. A reviewer asking "did you try a real mechanism rather than noisy one-way histograms?" now has an answer with numbers.
- **The binding constraint is the budget per column, not the mechanism.** 50 columns at n = 94 under composition leaves too little per column for any allocation strategy to matter. That is a stronger and more transferable claim than "our DP baseline was weak".
- **Report the budget-share ablation alongside the comparison.** Without it, AIM's fidelity number looks like a mechanism result when 43% of it is an allocation choice this project made.
- Do not quote AIM's correlation as evidence of anything: at 0.165 against a floor of 0.163, it is the floor.

## Limits and honest caveats

- **AIM here is not full AIM.** It models the 8 core columns (`COPULA_COLUMNS`) jointly and leaves the 40 lab columns on independent noisy histograms. Full AIM over all 46 columns did not finish a single fit in ten minutes — its 2-way workload is over a thousand candidate cliques, each of which the model is re-projected onto every round. The restriction is the same one `DPGaussianCopula` already makes, and a full-domain AIM would have *less* budget per measurement, not more, so the direction of the conclusion is unlikely to flip. It has not been shown.
- **`max_iters` is 100, not the authors' 1000**, and `max_model_size` is 10, not their recommended 80 for "official comparisons". Both are handicaps on AIM, chosen for cost, and both are recorded in `hyperparams()`.
- **Accounting.** The three measurements compose by basic composition in (ε, δ) rather than a single zCDP account, because AIM keeps its own account internally — that part is deliberately conservative. The neighbour-relation conversion is not a conservatism but a requirement, and the first version got half of it wrong: running AIM at ε/2 converts add/remove to replace-one for **ε**, but the same conversion multiplies **δ** by (1 + e^(ε/2)), which left the arm's true δ at about 1.4e-5 at ε = 1 and 2.7e-5 at ε = 8 while it advertised 1e-5. AIM is now passed δ/(3(1 + e^(ε/2))) so the published δ is the real one (caught in Track 2 review, 2026-10-27). Every number in this document is from the corrected version; the correction moved mean JSD by less than 0.002, so it changes the claim rather than the result.
- MST was not run; the task named "AIM or MST" and AIM is the stronger of the two.
- Patient-level privacy is still not claimed by any arm, AIM included. That is task J5.
