"""
Doppel - export the generated synthetic datasets for the web dashboard's Data section.

Ports every tab of the Streamlit "Dataset Explorer" to a page that needs no upload, for all 100 evaluated
datasets (5 generators x 20 seeds under output/synthetic/eval/), not just whatever CSVs sit in output/synthetic/:

  - a preview, column types, distinct-value counts and the generator manifest (Streamlit's Dataset Explorer tab)
  - per-column fidelity against the real training rows, with a correlation gap and a numeric/categorical
    column drill-down (Streamlit's Compare to Real Data tab), using Track 2's own fidelity_metrics code so the
    numbers match the Compare section's aggregates exactly, rather than a second, slightly different definition

The evaluation report upload tab is not ported: the Compare and Trade-offs sections already show the full
20-seed aggregate and Pareto frontier, which a one-off uploaded report cannot improve on.

    python web/scripts/export_datasets.py

The output goes to web/public/data/datasets/ and is git-ignored, like output/synthetic/ itself.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.fidelity_metrics import (  # noqa: E402
    BINARY_COLS, CATEGORICAL_COLS, CONTINUOUS_COLS, NUMERIC_COLS,
    correlation_preservation, dimension_wise_js, infer_lab_cols,
)
from generators import schema as S  # noqa: E402

SRC = ROOT / "output" / "synthetic" / "eval"
REAL_CSV = ROOT / "output" / "mimic_demo_clean.csv"
OUT = ROOT / "web" / "public" / "data" / "datasets"
TOP_CATEGORIES = 6  # categorical column detail: this many named shares, the rest folded into "Other"

LABELS = {
    "independent_marginals": "Independent marginals",
    "gaussian_copula": "Gaussian copula",
    "gaussian_copula_shrink025": "Gaussian copula, shrink 0.25",
    "gaussian_copula_shrink010": "Gaussian copula, shrink 0.10",
    "ctgan": "CTGAN",
    "ctgan_small": "CTGAN, small networks",
    "tvae": "TVAE",
    "tabular_diffusion": "Tabular diffusion",
    "tabular_diffusion_reg": "Tabular diffusion, regularized",
}


def to_native(v):
    """A pandas/numpy cell -> a JSON-safe Python value, missing -> None."""
    if pd.isna(v):
        return None
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def column_detail(series, kind):
    """What the Data section's column-detail card needs for one column: summary stats for a numeric
    column, or value shares for a categorical one -- the two views Streamlit's "Column detail" picker offers."""
    if kind == "numeric":
        d = series.describe()
        return {"stats": {k: round(float(d[k]), 4) for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]}}
    shares = series.dropna().astype(str).value_counts(normalize=True)
    top = shares.head(TOP_CATEGORIES)
    rest = float(shares.iloc[TOP_CATEGORIES:].sum())
    out = {str(k): round(float(v), 4) for k, v in top.items()}
    if rest > 0:
        out["Other"] = round(rest, 4)
    return {"shares": out}


def fidelity_cols(real_train):
    """The columns Track 2's fidelity metric scores, and the real-side half of their column detail (the
    synthetic half is computed per dataset; the real half is the same for every one, so it is written once)."""
    lab_cols = infer_lab_cols(real_train)
    categorical = [c for c in CATEGORICAL_COLS + BINARY_COLS if c in real_train.columns]
    numeric = [c for c in CONTINUOUS_COLS + lab_cols if c in real_train.columns]
    corr_cols = [c for c in NUMERIC_COLS + lab_cols if c in real_train.columns]
    kinds = {c: "categorical" for c in categorical} | {c: "numeric" for c in numeric}
    detail = {c: column_detail(real_train[c], kinds[c]) for c in kinds}
    return kinds, corr_cols, detail


def export_one(csv_path, real_train, kinds, corr_cols):
    # icd9_primary: pin as string or a leading zero is read off and a code silently becomes a different one (P-005).
    df = pd.read_csv(csv_path, dtype={"icd9_primary": str})
    columns = [
        {"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()), "distinct": int(df[c].nunique())}
        for c in df.columns
    ]
    rows = [[to_native(v) for v in row] for row in df.itertuples(index=False, name=None)]

    js_scores = dimension_wise_js(real_train, df, [c for c, k in kinds.items() if k == "categorical"],
                                   [c for c, k in kinds.items() if k == "numeric"])
    fidelity = {
        "mean_js": round(float(np.mean(list(js_scores.values()))), 4) if js_scores else None,
        "corr_diff": round(correlation_preservation(real_train, df, corr_cols), 4),
        "columns": sorted(
            [{"name": c, "kind": kinds[c], "js": round(float(v), 4)} for c, v in js_scores.items()],
            key=lambda r: -r["js"],
        ),
    }
    column_detail_synth = {c: column_detail(df[c], k) for c, k in kinds.items() if c in df.columns}

    manifest_path = csv_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    return {
        "file": csv_path.name,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "missing": int(sum(c["missing"] for c in columns)),
        "columns": columns,
        "rows": rows,
        "manifest": manifest,
        "fidelity": fidelity,
        "column_detail": column_detail_synth,
    }


def main():
    if not SRC.exists():
        sys.exit(f"{SRC} does not exist. Generate the evaluation datasets first (see docs/B3).")
    if not REAL_CSV.exists():
        sys.exit(f"{REAL_CSV} does not exist. Run the Stage 1 cleaning step first (see docs/A... / README).")

    real = pd.read_csv(REAL_CSV, dtype={"icd9_primary": str})
    real_train = real[real[S.SPLIT_COL] == S.TRAIN_SPLIT].reset_index(drop=True)
    kinds, corr_cols, real_detail = fidelity_cols(real_train)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real.json").write_text(
        json.dumps({"n_train": len(real_train), "columns": real_detail}, separators=(",", ":")), encoding="utf-8"
    )

    index = []
    for arm, label in LABELS.items():
        files = sorted((SRC / arm).glob("*_seed*.csv"), key=lambda p: int(p.stem.rsplit("seed", 1)[1]))
        seeds = []
        for f in files:
            seed = int(f.stem.rsplit("seed", 1)[1])
            (OUT / f"{arm}_seed{seed}.json").write_text(
                json.dumps(export_one(f, real_train, kinds, corr_cols), separators=(",", ":")), encoding="utf-8"
            )
            seeds.append(seed)
        if seeds:
            index.append({"id": arm, "label": label, "seeds": seeds})
    (OUT / "index.json").write_text(json.dumps({"arms": index}, indent=1), encoding="utf-8")
    total = sum(len(a["seeds"]) for a in index)
    print(f"wrote {total} datasets (+ real.json) for {len(index)} generators to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
