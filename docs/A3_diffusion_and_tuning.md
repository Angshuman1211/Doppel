# Diffusion Generator & Final Tuning

**Phase:** Phase 3 — Integration & Evaluation (Weeks 8–11)
**Owner:** Angshuman (Track 1 / Track A)
**Status:** The diffusion generator is built, so all three blueprint families exist. All four arms were tuned over 390 fits across two grids. The outcome is an honest negative for the neural arms: no diffusion config meets the A2 acceptance bar, CTGAN still learns no dependence after shrinking, and TVAE memorizes at every setting. Track 3's manifest dtype bug is fixed; Track 4's container and Job gaps are reported, not fixed (P-020). See [`problems_and_decisions.md`](problems_and_decisions.md) ADR-028, ADR-029, P-019, P-020, and the numbers in [`phase3_tuning_result.md`](phase3_tuning_result.md).

---

## Objective

Close the two things Phase 2 left open. The blueprint's third generator family, the diffusion model, didn't exist, so Track 2's evaluation covered two families out of three. And the neural configs were untuned, so "the classical baseline beats both neural generators" was open to the obvious objection that the defaults were simply too big for 75 training rows. This phase builds the diffusion arm and answers that objection with a sweep.

---

## What was built

### Diffusion generator (ADR-028)

```
generators/diffusion.py    TabularDiffusion: Gaussian diffusion on numeric columns, multinomial
                           diffusion on the 66 categorical and binary ones, one MLP denoiser
```

It follows the design recorded in A1, after TabDDPM (Kotelnikov et al., 2023):

- **Numeric columns** reuse the copula's rank-to-normal transform, then standard DDPM noise prediction on a cosine schedule.
- **Categorical and binary columns** each get multinomial diffusion: the forward process mixes the one-hot toward uniform, and the model predicts the clean class.
- **One denoiser** (MLP with a sinusoidal timestep embedding) outputs the noise for the numeric block and per-column logits for the discrete blocks.
- **Training** uses the simplified objective: MSE plus cross-entropy, as most TabDDPM implementations do.
- **Determinism** matches the other neural arms: CPU, one torch thread, seeded from the `rng`. Same seed gives the same SHA-256.

It fits in about 46 s (40 s training, 6 s sampling), comparable to CTGAN, and passes the Stage 2 contract unchanged, so the codec, validator and manifest needed no changes to take a third family.

### Tuning sweep

`generators/sweep.py` gained named grids (`--grid phase3`, `--grid diffusion_refine`) and a cost model that orders tasks by measured seconds per fit, so the longest run first across the worker pool. The Phase 3 grid tested the three questions ADR-024 left open, over 18 configs × 3 seeds × 5 folds.

### Configs for the final comparison (ADR-029)

| Arm | Config | Role |
|---|---|---|
| `independent_marginals` | none | no-dependence floor |
| `gaussian_copula` | Ledoit-Wolf | untuned reference |
| `gaussian_copula_shrink010` | shrinkage 0.10 | tuned statistical baseline, best correlation in the project (0.111) |
| `ctgan_small` | 300 epochs, 64×64, embedding 32 | tuned GAN arm, mean JSD 0.103 → 0.076 |
| `tvae` | 300 epochs | unchanged, memorization control |
| `tabular_diffusion` | 3000 steps | best neural fidelity (0.036), memorizes 80% of rows |
| `tabular_diffusion_reg` | 1000 steps, dropout 0.2 | the only diffusion setting that clears the guard |

Both diffusion configs go forward so the benchmark shows the trade-off rather than one chosen point from it.

### Track 3 fix (P-019)

Their dataset manifest declared `icd9_primary` as an integer, which writes the P-005 trap into the data contract: read that way, `0389` becomes 389, a different code. `generate_manifest.py` and `validate_dataset.py` now pin the dtype, the manifest says `string` at schema_version 1.1, and their validator gained a check comparing `icd9_primary` against the first entry of `icd9_codes`. The guard reports 25 bad rows on the buggy read and is silent on the correct one.

### Dashboard: upload-driven (ADR-030)

Asked for by the course instructor: the dashboard should start empty and work on uploaded data rather than whatever happens to sit in `output/`. `dashboard/app.py` now reads nothing from disk. Synthetic CSVs and an optional real cleaned CSV are uploaded in the sidebar, evaluation reports in the evaluation tab, and a new "Compare to Real Data" tab scores an uploaded pair: contract validation, mean JS divergence, the correlation gap, per-column divergence, and a copy check.

The image still ships only `app.py` with streamlit and pandas, so the comparison tab computes fidelity itself, matching `evaluation.fidelity_metrics` exactly on all four generators at seed 42 (0.0156, 0.0854, 0.1091, 0.0423). Started from a repo checkout it uses Track 2's metric and Track 1's validator instead and says which is in use. The file belongs to Track 4, so Anshuman needs to know it changed.

### Track 4 integration findings (P-020, reported not fixed)

Checked the Phase 2 containers against the generation stage. Four issues, all in Track 4's files:

1. **No `.dockerignore`** with `COPY . /app` in both generator Dockerfiles, so a local `.venv` (1.0 GB), `.git` and generated output land in the build context and image.
2. **No volume mounts on the Jobs**, so the synthetic CSV and manifest are written inside the container and vanish when the Job ends. Stage 3 can never read them. `--out-dir` and `--real-csv` exist for this.
3. **The Jobs don't run the selected configs**: `ctgan-job.yaml` passes `epochs=1` and the statistical Job uses default shrinkage instead of the carried-forward setting.
4. **No resource requests or limits** on either Job.

Docker wasn't running on this machine, so the images couldn't be built and the "same bytes inside the container" check is still outstanding.

---

## Commands

```bash
# Setup (Python 3.11, the Docker image's version)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r generators/requirements-neural.txt

# The diffusion arm, like any other generator
python -m generators.generate --generator tabular_diffusion --seed 42
python -m generators.validate output/synthetic/tabular_diffusion_seed42.csv

# Tuning sweeps (write output/sweeps/)
python -m generators.sweep --seeds 0 1 2                            # the Phase 3 grid, 270 fits
python -m generators.sweep --grid diffusion_refine --seeds 0 1 2    # the diffusion refinement

# Track 3's data contract, after the dtype fix
python preprocessing/generate_manifest.py && python preprocessing/validate_dataset.py
```

---

## Key Decisions

**Why implement the simplified diffusion loss rather than the full variational bound?** Two coupled processes already make this the most intricate generator in the project, and the simplified objective is what most TabDDPM implementations use. On 94 rows the loss form is not what decides the outcome. See ADR-028.

**Why reuse the copula's numeric transform?** Every generator has to see the same encoding (ADR-008), otherwise a fidelity difference could come from preprocessing. The rank-to-normal mapping is already written, already invertible, and already used by a shipped arm.

**Why keep tuning CTGAN and TVAE at all, given Phase 2?** Because "the neural models are untuned" was the strongest objection to the Phase 2 result, and Track 2's write-up flags it explicitly. The sweep answers it directly rather than leaving it open.

**Why not tune on the holdout?** Unchanged from ADR-021: the holdout is Track 2's test set, so hyperparameters are chosen inside train with a memorization guard.

---

## Outputs

| Output | Value |
|---|---|
| Diffusion generator | `generators/diffusion.py`, registered as `tabular_diffusion` |
| Named sweep grids | `generators/sweep.py` (`--grid phase3`, `--grid diffusion_refine`) |
| Track 3 fix | `generate_manifest.py`, `validate_dataset.py`, regenerated manifest (P-019) |
| Upload-driven dashboard | `dashboard/app.py` (ADR-030), Track 4's file |
| Track 4 report | P-020, for Anshuman |
| Final-comparison arms | ADR-029; Track 2 needs them added to `evaluation/arms.py` |
| Result writeup | [`phase3_tuning_result.md`](phase3_tuning_result.md) |
