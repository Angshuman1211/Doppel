# C3 — MIMIC-IV Demo Replication: what Track 3 needs to build

**For Track 3 (data engineering). Written by Track 1, 2026-10-26.** This is a request and a spec, not a finished design — the decisions marked **D1–D5** are yours to make, and Track 1 needs to know what you decide because three of them change code on our side.

## What this is for

The journal version needs the nine-arm benchmark repeated on a second clinical dataset, so the paper's conclusions are not "a property of MIMIC-III demo". The target is the **MIMIC-IV Clinical Database Demo** on PhysioNet — openly downloadable, no credentialing, same as the MIMIC-III demo we already use.

**Set expectations now: this is a replication, not a scale-up.** The MIMIC-IV demo is also about 100 patients. It will not fix n = 94. What it tests is whether the same generators behave the same way on a different hospital extract with a different schema, which is a question reviewers do ask.

## The deliverable

One file, `output/mimic_iv_demo_clean.csv`, that satisfies the **same contract** as `output/mimic_demo_clean.csv`, plus the ICD lookup equivalent to `output/icd9_lookup.csv`. If the contract holds, Track 1 reruns the nine arms with no code changes at all. That is the whole point of the contract and it has already paid off twice (the UCI sweep needed zero generator changes).

Required columns, names and semantics exactly as now — this is `generators/schema.py`, not a wish list:

| Column(s) | Type | Notes |
|---|---|---|
| `subject_id`, `hadm_id` | int | identifiers, never modelled |
| `split` | str | `train` / `holdout`, patient-level 80/20, seed 42 (ADR-001) |
| `age` | int | clipped at 89, see **D4** |
| `age_89_plus` | bool | |
| `gender` | str | |
| `admission_type`, `ethnicity`, `first_careunit` | str | categorical, see **D2** and **D3** |
| `hospital_expire_flag`, `readmit_30d` | 0/1 int | `readmit_30d` = next admission within 30 days of discharge |
| `los_hospital_days`, `los_icu_days` | float | ICU LOS summed per admission, null if no ICU stay |
| `icd9_primary` | **str** | `seq_num == 1`. Must stay a string — see P-005 below |
| `icd9_codes` | str | JSON list of code strings, in `seq_num` order |
| `n_diagnoses` | int | count of codes for the admission |
| `lab_<itemid>_mean` | float | 20 labs, see **D5** |
| `lab_<itemid>_abnormal_frac` | float | fraction of that lab's results flagged abnormal |

Keep the column **names** identical even where the source column changed name. `ethnicity` stays `ethnicity` even though MIMIC-IV calls it `race`; `icd9_primary` keeps its name even if it holds an ICD-10 code. Renaming breaks `generators/schema.py`, `generators/codec.py`, both evaluation suites and every research script.

## What actually differs, table by table

`preprocessing/preprocess_mimic_demo.py` reads seven files from `RAW_DIR`. In MIMIC-IV they move, change case, and are gzipped:

| Now | MIMIC-IV |
|---|---|
| `PATIENTS.csv` | `hosp/patients.csv.gz` |
| `ADMISSIONS.csv` | `hosp/admissions.csv.gz` |
| `ICUSTAYS.csv` | `icu/icustays.csv.gz` |
| `DIAGNOSES_ICD.csv` | `hosp/diagnoses_icd.csv.gz` |
| `D_ICD_DIAGNOSES.csv` | `hosp/d_icd_diagnoses.csv.gz` |
| `LABEVENTS.csv` | `hosp/labevents.csv.gz` |
| `D_LABITEMS.csv` | `hosp/d_labitems.csv.gz` |

`RAW_DIR` is already an environment variable, so paths are the easy part. The column changes are not:

**Age — `compute_age()` has to be rewritten, not adjusted.** MIMIC-IV has no `dob`. `patients` carries `anchor_age`, `anchor_year` and `anchor_year_group`, and age at admission is `anchor_age + (year(admittime) − anchor_year)`. The whole comment in our current function about DOB being shifted ~300 years into the past stops applying: MIMIC-IV de-identifies the over-89s by setting `anchor_age` to 91 instead. So the overflow workaround is unnecessary and the `> 89` test means something different.

**`ethnicity` → `race`** in `admissions`. Rename on read.

**Diagnoses are the big one.** MIMIC-IV's `diagnoses_icd` has `icd_code` and `icd_version` (9 or 10), not `icd9_code`. The demo contains both. See **D1**.

**`admission_type` values differ.** MIMIC-III demo has four (`ELECTIVE`, `EMERGENCY`, `NEWBORN`, `URGENT`); MIMIC-IV uses a longer set including `EW EMER.`, `OBSERVATION ADMIT`, `SURGICAL SAME DAY ADMISSION`, and has no newborns. See **D2**.

**`icustays`** keeps `los` and `first_careunit`, but the care-unit strings differ. `labevents` keeps `itemid`, `valuenum`, `hadm_id` and `flag`; check whether `flag` still uses the literal `"abnormal"`, because `build_lab_features` tests for that exact string and will silently produce all-zero `abnormal_frac` columns if it changed.

## Decisions we need from you

**D1 — ICD-9 or ICD-10 or both.** This is the one that matters most. Options: keep only `icd_version == 9` rows (clean, comparable to MIMIC-III, but may discard most of the demo, since MIMIC-IV's later admissions are ICD-10); keep only ICD-10 (internally consistent, not comparable); or keep both and let the code string carry whichever. Whatever you choose, please report **how many admissions survive** — if ICD-9-only leaves 40 admissions, the replication is not worth running and we should know before anyone builds it. Track 1 cannot pick this for you; it is a data judgement.

**D2 — `admission_type` level set.** Collapse to the MIMIC-III four, or keep MIMIC-IV's native set? Either is defensible. It changes `LEVELS["admission_type"]` in `generators/dp_copula.py`, which is a **public domain** for the DP arms and therefore must be declared from the schema, never measured off the data.

**D3 — race/ethnicity collapse.** We currently take the top 6 by frequency plus `OTHER` (`collapse_ethnicity`). Note this is measured off the data, which is fine for the non-DP arms but is exactly the kind of thing that broke the first DP implementation. Prefer a fixed published list if MIMIC-IV gives you one.

**D4 — age convention.** Simplest for comparability: clip at 89 and set `age_89_plus = age > 89`, matching `S.AGE_CAP`. If you instead carry MIMIC-IV's 91 convention, tell us, because `PUBLIC_BOUNDS["age"]` and the age cell in the DP domain both assume 89.

**D5 — the 20 labs.** We take the top 20 `itemid`s by distinct admissions. MIMIC-IV reuses many MIMIC-III itemids, but the top-20 set will not be identical. **This breaks the DP arms**: `generators/dp_copula.py` has a `PUBLIC_BOUNDS` table keyed by MIMIC-III itemid (`"50882"` bicarbonate, `"51301"` white blood cells, and 18 more) and `_domain()` does a direct lookup, so any new itemid raises `KeyError`. Please send Track 1 the final top-20 list and we will add clinical bounds for the new ones. The non-DP arms are unaffected.

## Traps that already cost us time

- **P-005: `icd9_primary` must be read and written as a string.** `0389` (septicemia) parsed as an integer becomes `389`, a different code. `S.load_real()` pins `dtype={PRIMARY_COL: str}`; write it so a naive `read_csv` does not silently break it.
- **LF line endings everywhere.** The dataset sha256 gate in the Kubernetes Job depends on byte-identical output across OSes. The current script already handles this; keep it.
- **Patient-level split, never row-level.** Admissions follow their patient (ADR-001). A row-level split leaks the same patient into train and holdout and invalidates every privacy number.
- **Check the holdout's class balance before declaring victory.** On MIMIC-III the holdout has 6 deaths in 35 rows and `readmit_30d` has 2 positives *both from one patient*, which is why the second prediction task is impossible there (`research/task_study_result.md`). Please report deaths and readmissions per split so Track 1 knows up front whether utility is measurable at all.

## Acceptance

The dataset is done when `preprocessing/validate_dataset.py` passes against it and `generators/validate.py`'s contract check passes. Then add it to the manifest with `preprocessing/generate_manifest.py` so the hash gate covers it.

Please also report, in whatever form suits you: rows, patients, admissions per patient (max and distribution), deaths and readmissions per split, and the D1 survival count. Those numbers decide whether Track 1 runs the full nine arms or only the subset that is measurable.

## What Track 1 does next

Once the CSV exists and validates: rerun the nine arms through `evaluation/eval_runner.py`, then the parts of the extension that are dataset-specific — the patient bootstrap, the split study, the DP arms and the membership attacks. That is our side and it is mostly a matter of compute, not new code, provided the contract holds.

One ask: tell us about D1 and D5 **before** building the rest. Those two decide whether the DP arms run at all and whether the diagnoses columns are comparable, and both are much cheaper to settle now than after the pipeline is written.
