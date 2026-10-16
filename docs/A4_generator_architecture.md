# Generation Stage — Final Architecture Reference

**Phase:** Phase 4 — Finalization & Delivery (Weeks 12–14)
**Owner:** Angshuman (Track 1 / Track A)
**Status:** Final reference for Stage 2 of the pipeline. Supersedes nothing: [`A1`](A1_generative_modeling.md) records the Phase 1 design and literature, [`A2`](A2_generator_training.md) the Phase 2 build, [`A3`](A3_diffusion_and_tuning.md) the Phase 3 diffusion arm and tuning. This document is the single place to read how the stage works, for the report and for anyone picking it up afterwards.

---

## What the stage does

One command turns the real training split into a synthetic dataset with the same schema:

```
output/mimic_demo_clean.csv  --(train rows only)-->  FrameCodec  -->  modeling frame (94 x 110)
                                                                            |
                                                                       Generator.fit
                                                                            |
                                                                       Generator.sample
                                                                            |
      output/synthetic/<arm>_seed<n>.csv  <--  contract validator  <--  FrameCodec.decode
      output/synthetic/<arm>_seed<n>.manifest.json
```

Nothing downstream needs to know which generator ran. Stage 3 reads a CSV with exactly the columns of the real one, plus a manifest describing how it was produced.

---

## 1. The data contract

The generators only ever see the `train` split (ADR-001). Everything the stage emits satisfies these invariants, enforced by `generators/validate.py`, which `generate.py` runs before writing. A file that fails is never written, so a failed Job leaves nothing behind for the next stage to pick up.

1. Same columns, same order, same dtype kinds as `output/mimic_demo_clean.csv`.
2. `split == "synthetic"` on every row.
3. `subject_id` from 10,000,000 and `hadm_id` from 20,000,000, never colliding with a real ID, `hadm_id` unique. One admission per synthetic patient (ADR-011).
4. Every categorical level and every ICD-9 code was seen in **train**. Anything else is a holdout leak or an invented value, so this doubles as a leakage tripwire.
5. Binary columns in {0, 1}; `age` in [0, 89]; `age_89_plus` rows have `age == 89`.
6. Continuous columns finite and non-negative, `_abnormal_frac` ≤ 1, NaN only in the three labs that are nullable in train, and a lab's `_mean` and `_abnormal_frac` missing together.
7. `icd9_codes` is a JSON list of strings, no duplicates, first element equal to `icd9_primary`, length equal to `n_diagnoses`.

`icd9_primary` must be read as a string everywhere: parsed as a number, `0389` becomes `389`, a different diagnosis, on 25 of the 129 rows (P-005, P-019).

## 2. The modeling frame

`generators/codec.py` maps the real CSV to one 110-column frame with no missing values, and back. Every generator trains on this exact frame and is decoded by the same code, so a difference in the benchmark is the model and not the preprocessing (ADR-008).

| Kind | Columns | Count |
|---|---|---|
| categorical | `admission_type`, `ethnicity`, `first_careunit`, `gender`, `icd9_primary` | 5 |
| binary | `hospital_expire_flag`, `readmit_30d`, `age_89_plus` | 3 |
| binary | `miss_<itemid>` for the 3 labs with gaps in train | 3 |
| binary | `icd_<code>` multi-hot, one per secondary code in ≥ 5 train admissions | 55 |
| integer | `age`, `n_tail_codes` | 2 |
| continuous | `los_hospital_days`, `los_icu_days`, 40 lab columns | 42 |

Rebuilt on decode rather than generated: `subject_id`, `hadm_id`, `split`, `icd9_codes`, `n_diagnoses`.

**ICD-9 codes (ADR-009).** The primary code is a categorical column. Secondary codes in at least 5 training admissions (55 codes, 49% of secondary mentions) are modeled jointly as multi-hot. The remaining 395 form a tail pool: only the per-admission count is modeled, and codes are drawn from the pool by training frequency on decode, independently of the rest of the row. Individual rare codes stay realistic while rare *combinations*, which are what identify a patient, are never reproduced. `n_diagnoses` is derived from the decoded list, so the list and the count cannot disagree.

**The defining constraint.** 110 modeled columns against 94 training rows. Any model flexible enough to fit this frame well is flexible enough to memorize it, which is what the results show.

## 3. The generators

All five implement `Generator` (`fit(frame, spec, rng)` / `sample(n, rng)`), registered in `generate.GENERATORS`.

| Name | Family | Method |
|---|---|---|
| `independent_marginals` | reference floor | empirical marginals, zero dependence. Not a benchmarked family; it makes every other number readable |
| `gaussian_copula` | statistical | empirical marginals joined by a Gaussian copula; the latent correlation is shrunk toward the identity, intensity from Ledoit-Wolf or given (ADR-010) |
| `ctgan` | GAN | CTGAN from the `ctgan` library on the shared frame, CPU, one torch thread (ADR-020) |
| `tvae` | VAE | TVAE from the same library, same conditions |
| `tabular_diffusion` | diffusion | TabDDPM-style: Gaussian diffusion on the numeric block, multinomial diffusion per discrete column, one MLP denoiser, simplified MSE + cross-entropy objective (ADR-028) |

Two implementation notes that matter for results rather than style: the copula samples through a Cholesky factor, because with d > n the shrunk matrix has 17 identical eigenvalues and `eigh` returns a build-dependent basis (P-013); and the neural arms pin one torch thread, which is both faster at this size and makes float reduction order identical across machines (ADR-020).

## 4. Choosing hyperparameters without touching the holdout

The holdout belongs to Stage 3 (ADR-001), so selection happens inside train (ADR-021). `generators/sweep.py` splits train into 5 patient-level folds, fits on four, and scores against the fifth:

- **Fidelity** to the rows it fit, with Track 2's `run_fidelity_report`.
- **Memorization** with `generators/diagnostics.py`: `dcr_ratio`, the median Gower distance from synthetic rows to the fit rows over the same for real unseen rows, and `near_copy_rate`, the share of synthetic rows closer than the nearest 5% of real unseen rows.

Calibrated before use: real unseen rows score 1.000 and 0.059, exact copies score 0.000 and 1.000. A config counts only if `dcr_ratio ≥ 0.95` and `near_copy_rate ≤ 0.10`. Fidelity alone would pick whichever model copies hardest, which is exactly what the guard catches.

## 5. Reproducibility

- **Seeded end to end.** One seed spawns three streams (fit, sample, decode), so the same seed gives the same file, byte for byte.
- **Across environments.** All five generators at seed 42 are byte-identical on Python 3.11.5 / numpy 2.4.6 (the image) and Python 3.12 / numpy 2.5.3.
- **Every run is traceable.** The manifest records the generator, hyperparameters, seed, row counts, SHA-256 of the training CSV, the codec settings, and the git commit. `input_dataset_id` is content-addressed (`mimic_demo_clean@<sha256[:12]>`), so a change to the input is visible in every downstream result.
- **Pinned.** The root `requirements.txt` covers the copula arms; `generators/requirements-neural.txt` adds torch, ctgan and rdt (ADR-023).

## 6. Extending it

A sixth generator is one file and one line: subclass `Generator`, add `"<name>": "<module>:<Class>"` to `generate.GENERATORS`, and optionally a grid entry in `sweep.GRID`. The codec, contract, validator and manifest need no changes. The diffusion arm was added this way in Phase 3 and required no edits to any of them, which is the clearest evidence the contract was worth fixing first.

## 7. Commands

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r generators/requirements-neural.txt

python -m generators.generate --generator tabular_diffusion --seed 42
python -m generators.generate --generator gaussian_copula --seed 42 --param shrinkage=0.10
python -m generators.validate output/synthetic/gaussian_copula_seed42.csv
python -m generators.sweep --seeds 0 1 2                          # tuning grid
python -m generators.sweep --grid diffusion_refine --seeds 0 1 2
```

In Kubernetes the same command runs inside the generator image, with `--real-csv` and `--out-dir` pointed at the shared volume.

## 8. Known limits

- **94 training rows.** Everything here is a small-data result. The neural families are far outside the regime they were designed for, and the report should say so rather than implying these architectures are weak in general.
- **Rare code combinations are not modeled**, by design (ADR-009). Sweeping `--min-count` is the ablation that would quantify the cost.
- **Code order inside `icd9_codes` is not modeled** beyond the primary code.
- **Synthetic patients have one admission each**, so `readmit_30d` is generated as an attribute rather than derived from a second admission.
- **The in-train selection numbers are in-sample** and are not the benchmark; Stage 3 against the holdout is.
