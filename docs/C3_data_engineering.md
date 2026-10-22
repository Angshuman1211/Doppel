# Data Engineering — Phase 3

**Phase:** Phase 3 — Integration & Evaluation
**Owner:** Anoushka (Track 3 / Track C)
**Status:** Local preprocessing and validation checks were run on 2026-10-22. The preprocessing Kubernetes Job has **not** been run in a cluster. The optional hash gate is on `track3-phase3-hash-gate`, not in `origin/main` at `887cedc`; the preprocessing image has not been rebuilt with it.

---

## Objective

Make the processed dataset reproducible and identifiable by its bytes, close the raw-input path mismatch in the preprocessing Job, and document the known holdout class-coverage limitation without changing the patient-level split.

## What was built

- The preprocessing script writes all three CSV outputs with LF line endings. A local preprocessing run from the repository's raw CSVs, directed to a temporary output directory, reproduced the committed clean CSV's SHA-256.
- `validate_dataset.py` has an optional `EXPECTED_DATASET_SHA256` check. When set, it compares the clean CSV bytes to the expected digest. The hash-gate branch's Job sets the expected digest to `9e263f45fd4fa9ed7c1b53b656c77aceef84eda9162819730a6dcafbc26ad83e`.
- `generators/schema.py` on main exposes `HOLDOUT_EXCLUDED_SUBJECTS` and `clean_holdout(df)` for downstream analyses that need a holdout without subject 41976. The helper does not resplit or change the source CSV.
- Main's preprocessing Job now sets `RAW_DIR=/app/raw` and mounts the shared PVC for raw and output paths. This was checked in the Job specification only; the Job itself was not run.

## Commands

Run from the repository root in PowerShell:

```powershell
$env:EXPECTED_DATASET_SHA256 = "9e263f45fd4fa9ed7c1b53b656c77aceef84eda9162819730a6dcafbc26ad83e"
python preprocessing\validate_dataset.py
```

This real-data hash-gate check was run on `track3-phase3-hash-gate`, not on main. To reproduce the preprocessing result without overwriting tracked outputs, set `RAW_DIR` to the repository root and `OUT_DIR` to a new temporary directory, then run `preprocessing\preprocess_mimic_demo.py`.

## Key Decisions

- Keep the patient-level split from ADR-001; do not move subject 41976 between splits to make the admission counts or demographics look more balanced.
- Use `clean_holdout()` only when a downstream analysis requires the balance-sensitive comparison set. The helper is not automatically applied by preprocessing or validation.
- Leave P-018's validator unchanged. A strict coverage check over every categorical feature would reject legitimate sparse ICD-9 primary codes; measured coverage and the proposed scope are recorded below and in P-025.
- Treat the expected SHA-256 check as a byte-level gate, not as proof that the Kubernetes pipeline ran. The Job writes the CSV before validation, so a mismatch fails the Job but does not roll back the file already written to the volume.

## Outputs

| Output | Result |
|---|---|
| Canonical clean dataset | `output/mimic_demo_clean.csv`; SHA-256 reproduced locally |
| Hash validator and Job pin | Implemented on `track3-phase3-hash-gate`; not merged to main as of `887cedc` |
| Holdout exclusion helper | `generators/schema.py`, merged to main in PR #18 |
| Preprocessing Job paths | Corrected on main; only inspected statically |
| Cluster execution | **Not run**; no minikube, kubectl, or Docker executable was available in this shell |
| Open follow-up | P-018 remains open; see measured ICD-9 coverage below |

## Raw evidence

Dataset counts, split counts, class coverage, age cap, lab nulls, ICD-9 sparsity, and line-ending hashes were measured from the repository file at `output/mimic_demo_clean.csv` with this PowerShell/Python command:

```powershell
@'
from pathlib import Path
from collections import Counter
import hashlib, json
import pandas as pd
p = Path('output/mimic_demo_clean.csv')
b = p.read_bytes()
df = pd.read_csv(p, dtype={'icd9_primary': str})
print('sha256=', hashlib.sha256(b).hexdigest())
print('shape=', df.shape, 'patients=', df.subject_id.nunique())
for name, group in df.groupby('split', sort=True):
    print(name, 'admissions=', len(group), 'patients=', group.subject_id.nunique(), 'gender=', group.gender.value_counts().sort_index().to_dict(), 'deaths=', int(group.hospital_expire_flag.sum()), 'readmit_30d_positive=', int(group.readmit_30d.sum()))
clean = df[df.split.eq('holdout') & ~df.subject_id.eq(41976)]
train = df[df.split.eq('train')]
print('subject_41976=', int(df.subject_id.eq(41976).sum()), 'deaths=', int(df.loc[df.subject_id.eq(41976), 'hospital_expire_flag'].sum()), 'gender=', sorted(df.loc[df.subject_id.eq(41976), 'gender'].unique()), 'ethnicity=', sorted(df.loc[df.subject_id.eq(41976), 'ethnicity'].unique()))
print('clean_holdout admissions=', len(clean), 'patients=', clean.subject_id.nunique(), 'gender=', clean.gender.value_counts().sort_index().to_dict())
for col in ['admission_type','ethnicity','first_careunit','gender','icd9_primary']:
    print('unseen', col, 'all=', len(set(df.loc[df.split.eq('holdout'),col].dropna())-set(train[col].dropna())), 'clean=', len(set(clean[col].dropna())-set(train[col].dropna())))
lab = [col for col in df.columns if col.startswith('lab_')]
print('age_89_plus_rows=', int(df.age_89_plus.sum()), 'lab_features=', len(lab), 'lab_null_cells=', int(df[lab].isna().sum().sum()), 'rows_any_lab_null=', int(df[lab].isna().any(axis=1).sum()))
secondary = Counter(code for row in train.icd9_codes.map(json.loads) for code in row[1:])
frequent = {code: n for code, n in secondary.items() if n >= 5}
print('secondary_codes=', len(secondary), 'at_least_5=', len(frequent), 'mentions=', sum(secondary.values()), 'covered_mentions=', sum(frequent.values()), 'coverage_percent=', round(100*sum(frequent.values())/sum(secondary.values()), 2))
crlf = b.replace(b'\n', b'\r\n')
print('LF_count=', b.count(b'\n'), 'CRLF_count=', b.count(b'\r\n'), 'simulated_CRLF_sha256=', hashlib.sha256(crlf).hexdigest())
'@ | python -
```

```text
sha256= 9e263f45fd4fa9ed7c1b53b656c77aceef84eda9162819730a6dcafbc26ad83e
shape= (129, 56) patients= 100
holdout admissions= 35 patients= 20 gender= {'F': 12, 'M': 23} deaths= 6 readmit_30d_positive= 2
train admissions= 94 patients= 80 gender= {'F': 47, 'M': 47} deaths= 34 readmit_30d_positive= 9
subject_41976= 15 deaths= 0 gender= ['M'] ethnicity= ['HISPANIC/LATINO - PUERTO RICAN']
clean_holdout admissions= 20 patients= 19 gender= {'F': 12, 'M': 8}
unseen admission_type all= 0 clean= 0
unseen ethnicity all= 1 clean= 0
unseen first_careunit all= 0 clean= 0
unseen gender all= 0 clean= 0
unseen icd9_primary all= 18 clean= 13
age_89_plus_rows= 9 lab_features= 40 lab_null_cells= 16 rows_any_lab_null= 3
secondary_codes= 450 at_least_5= 55 mentions= 1132 covered_mentions= 551 coverage_percent= 48.67
LF_count= 130 CRLF_count= 0 simulated_CRLF_sha256= 02f67dd6f9fad88dff84473c1efe4d15051e0c9a145553e349d33b1f3bbafaef
```

The optional hash gate was also run against the real CSV:

```text
VALIDATION PASSED — 129 rows, 56 columns, all checks green.
```

A separate preprocessing run read the seven required raw CSVs from the repository root and wrote to a temporary directory. This was the command:

```powershell
@'
from pathlib import Path
import os, subprocess, sys, tempfile, hashlib
root = Path(r'C:\Users\Anoushka Sarkar\Desktop\SEM VII\BDA\BDA Project\mimic_demo_raw\mimic-iii-clinical-database-demo-1.4\Doppel')
with tempfile.TemporaryDirectory(prefix='doppel-preprocess-') as temp:
    env = dict(os.environ, RAW_DIR=str(root), OUT_DIR=temp)
    run = subprocess.run([sys.executable, str(root / 'preprocessing' / 'preprocess_mimic_demo.py')], cwd=root, env=env, text=True, capture_output=True)
    print('preprocess_exit=', run.returncode)
    print(run.stdout.strip())
    if run.stderr:
        print('stderr=', run.stderr.strip())
    if run.returncode:
        raise SystemExit(run.returncode)
    output = Path(temp) / 'mimic_demo_clean.csv'
    print('temp_csv_sha256=', hashlib.sha256(output.read_bytes()).hexdigest())
    print('matches_expected=', hashlib.sha256(output.read_bytes()).hexdigest() == '9e263f45fd4fa9ed7c1b53b656c77aceef84eda9162819730a6dcafbc26ad83e')
'@ | python -
```

```text
preprocess_exit= 0
Wrote 129 admission rows, 56 columns -> C:\Users\ANOUSH~1\AppData\Local\Temp\doppel-preprocess-cl_1o5t8\mimic_demo_clean.csv
Train/holdout split: {'train': 94, 'holdout': 35}
Lab item lookup -> C:\Users\ANOUSH~1\AppData\Local\Temp\doppel-preprocess-cl_1o5t8\lab_item_lookup.csv
ICD-9 description lookup -> C:\Users\ANOUSH~1\AppData\Local\Temp\doppel-preprocess-cl_1o5t8\icd9_lookup.csv
temp_csv_sha256= 9e263f45fd4fa9ed7c1b53b656c77aceef84eda9162819730a6dcafbc26ad83e
matches_expected= True
```

The expected digest therefore matches a local regeneration, but this says nothing about the container image, PVC contents, or cluster execution. The simulated CRLF conversion shows why the canonical LF line endings matter: it changes only the line terminators yet changes the raw SHA-256.
