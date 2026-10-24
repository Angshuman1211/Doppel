"""Rescore utility and attribute inference on the clean holdout, from the benchmark's own synthetic files.

Post-course extension work. ADR-032 moves attribute inference to the clean holdout (subject 41976 removed)
and keeps utility on the full holdout with the clean one as a robustness check. Track 1's numbers for that
were computed on another machine. This rescoring uses the exact synthetic tables of the 180-run benchmark
(`output/synthetic/eval/<arm>/`), so it needs no refitting, and it first rescores the full holdout to check
that those files reproduce the reported numbers.

    python -m research.clean_holdout_rescore
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.arms import REPORTED_ARMS, resolve  # noqa: E402
from evaluation.attribute_inference import run_attribute_targets  # noqa: E402
from evaluation.utility_eval import tstr_eval  # noqa: E402
from generators import schema as S  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
SEEDS = range(42, 62)
TARGETS = ("gender", "first_careunit", "age_bucket")


def synthetic_path(arm, seed):
    generator, _ = resolve(arm)
    return S.SYNTH_DIR / "eval" / arm / f"{generator}_seed{seed}.csv"


def main():
    warnings.filterwarnings("ignore")
    real = S.load_real()
    train = real[real[S.SPLIT_COL] == S.TRAIN_SPLIT].reset_index(drop=True)
    full = real[real[S.SPLIT_COL] != S.TRAIN_SPLIT].reset_index(drop=True)
    clean = S.clean_holdout(real)
    print(f"holdout: full {len(full)} rows ({int(full['hospital_expire_flag'].sum())} deaths), "
          f"clean {len(clean)} rows ({int(clean['hospital_expire_flag'].sum())} deaths)")

    rows = []
    for arm in REPORTED_ARMS:
        for seed in SEEDS:
            synth = S.load_real(synthetic_path(arm, seed))
            row = {"arm": arm, "seed": seed}
            for name, holdout in (("full", full), ("clean", clean)):
                u = tstr_eval(synth, holdout)
                row[f"tstr_{name}"] = float(np.mean([u["logistic_regression"], u["random_forest"]]))
                targets = run_attribute_targets(synth, train, holdout)
                for t in TARGETS:
                    row[f"gap_{t}_{name}"] = targets[t]["member_gap"]
                    row[f"nonmember_uplift_{t}_{name}"] = targets[t]["uplift_nonmembers"]
            rows.append(row)
        print(f"  {arm} done", flush=True)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "clean_holdout_rescore.csv", index=False)
    summary = df.groupby("arm").mean(numeric_only=True).drop(columns="seed").reindex(REPORTED_ARMS)
    summary["tstr_shift"] = summary.tstr_clean - summary.tstr_full
    cols = ["tstr_full", "tstr_clean", "tstr_shift", "gap_gender_full", "gap_gender_clean",
            "nonmember_uplift_gender_full", "nonmember_uplift_gender_clean",
            "gap_first_careunit_clean", "gap_age_bucket_clean"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print("\n" + summary[cols].to_string())
    summary.to_csv(OUT_DIR / "clean_holdout_rescore_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
