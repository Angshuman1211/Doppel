# Validation Against the Official TabDDPM — they do not agree, and the honest answer is that we cannot yet say why (2026-10-25)

Journal-version work (Track 1 task 3). `generators/diffusion.py` is our own implementation of a TabDDPM-style model, and every conclusion the paper draws about the diffusion family rests on it. A reviewer is entitled to ask whether it behaves like the published model. This runs the authors' own code against ours on the same cohort with the same metrics.

```bash
python -m research.tabddpm_check --size 1000 --seeds 3                                  # matched config
python -m research.tabddpm_check --size 1000 --seeds 3 --implementations official \
    --config authors --out tabddpm_check_authors                                        # authors' regime
```

The official model is `tab_ddpm.GaussianMultinomialDiffusion` with `MLPDiffusion` from yandex-research/tab-ddpm at commit `b476257d` (MIT), imported directly. Only their *model* is used: their training pipeline, data loaders and CatBoost evaluation are not involved, because the object under test is the generative model and not their harness. Their model package depends on nothing but numpy and torch, so no part of the pinned environment changed.

---

## Result

UCI Diabetes 130-US, 1,000 patients (1,433 rows, 9 numeric and 37 categorical columns, 306 one-hot width), 3 seeds.

| Implementation | Config | Failures | Mean JSD | Corr. diff | Near-copy | DCR ratio | TSTR RF | Seconds |
|---|---|---|---|---|---|---|---|---|
| **ours** | matched | **0 / 3** | **0.076** | **0.037** | 0.007 | 1.20 | 0.556 | 49 |
| official | matched | **2 / 3** | 0.240 | 0.155 | 0.006 | 1.32 | 0.513 | 88 |
| official | authors' | **2 / 3** | 0.319 | 0.156 | 0.000 | 2.15 | 0.539 | 591 |

The official model raised `FoundNANsError` during sampling on two of three seeds, under both configurations. Figures for it are therefore from the single seed that completed, and are not means.

**What was matched** so the comparison is about the implementation and not the configuration: cosine beta schedule, 1,000 diffusion timesteps, epsilon parametrization with MSE on the numeric block, x0 parametrization on the categorical block, a 2-layer MLP of width 256, AdamW at lr 1e-3 with weight decay 1e-5, batch 64, 3,000 steps.

**The "authors' regime" control** exists because the matched configuration is *our* default, and its batch of 64 with 3,000 steps is far from anything in the authors' configs (batches of 1024 to 4096, tens of thousands of steps). If the official model were simply undertrained or starved of batch size, raising it to 20,000 steps at batch 256 should have helped. It did not: the same two seeds failed, and the surviving seed got **worse** on every fidelity axis, with a DCR ratio of 2.15 — the model moving further from the data, not closer.

## Where the failure is

Running the official model on each block alone, at 600 steps on the same cohort:

- **categorical-only** (37 columns, 306 one-hot width): completes, no NaNs.
- **numeric-only**: does not run, because their `GaussianMultinomialDiffusion.__init__` concatenates over an empty `num_classes` array; their pipeline always passes a placeholder there.

So the multinomial half of their model is stable on this data and the instability involves the Gaussian half, which matches the error being raised during sampling rather than training — training completed in every run. Their `sample` calls `gaussian_p_sample(..., clip_denoised=False)`, so nothing bounds the numeric reverse process over 1,000 steps; ours is bounded implicitly because its numeric values are mapped back through each column's own fitted marginal.

## What this does and does not establish

**It does not establish that the official TabDDPM is worse.** We wrote the driver that feeds it, today, and a defect there is the single most likely explanation for a well-cited model diverging. The known asymmetries are listed in the script's docstring: the multinomial loss (ours is cross-entropy to the clean class, theirs the full stochastic variational bound), the numeric transform (ours per-column marginal-to-normal, theirs a quantile transform), and where the time embedding enters the network. Any of those could matter, and only the first is a deliberate choice on our side.

**Two further candidates, and the evidence pointing at them** (Track 2 review, 2026-10-27). The authors' training loop keeps an **exponential moving average of the weights** and samples from the EMA copy; this driver samples from the live weights, which is exactly the kind of omission that shows up as an unstable sampler. Their sampler also runs with `clip_denoised=False`, which is theirs, not ours, but it only bites when the model is poorly converged. The supporting observation is the correlation error on the one completed seed: **0.155, against 0.099 for independent marginals on the same cohort** — so the official model's correlation structure is not merely unlearned, it is *worse than modelling the columns independently*. A model that has genuinely learned nothing lands at the no-dependence level; landing above it means something is actively wrong, which points at the harness rather than at the method. Testing EMA is the cheapest next step and the first thing to try.

**It does establish that the validation failed.** We cannot currently say our implementation behaves like the official one on this dataset, because the official one did not produce three usable runs to compare against. That is the answer to the reviewer's question, and it is a different answer from the one this task was hoping for.

## What this changes for the paper

- **Do not write "we reimplement TabDDPM".** Write that the generator is a TabDDPM-*style* model and name the choices: Gaussian diffusion on the numeric block with a cosine schedule and epsilon parametrization, multinomial diffusion (Hoogeboom et al., 2021) on the categorical block trained with cross-entropy to the clean class rather than the full variational bound. That description is accurate, checkable, and does not claim an equivalence that has not been demonstrated.
- The limitations section should record that a head-to-head against the authors' code was attempted and did not reproduce agreement, with this document as the reference. A limitation that is stated and measured costs far less with a reviewer than one they find themselves.
- The cross-entropy substitution is now the first thing to test if anyone picks this up again: it is the one difference we chose, it is the one in the block that stayed stable, and our docstring's claim that "most TabDDPM implementations" do it remains unverified.

## Limits

One dataset, one cohort size, three seeds, CPU only, and a driver written for this comparison rather than taken from the authors' pipeline. A fairer rematch would run their `scripts/pipeline.py` end to end on data converted to their format, which is a larger job and needs their 2021 dependency set (numpy 1.21, scikit-learn 1.0.2) in a separate interpreter — that path was not taken here.
