"""Repeated patient-level splits: how much of the benchmark's spread is the split, not the generator?

Post-course extension work, outside the graded pipeline. The reported benchmark uses one 80/20
patient-level split whose holdout holds 6 deaths in 35 rows, 15 of them from a single patient. This
redraws the split K times by the same procedure (ADR-001: sample 80% of subject_ids, admissions follow
their patient), refits every arm on each split, and scores fidelity against that split's train and
utility against that split's holdout.

Output: one row per (split, arm, seed) in research/output/split_study.csv, plus a summary that separates
split variance from seed variance.

    python -m research.split_study --splits 10 --seeds 3 --workers 6
"""

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.arms import REPORTED_ARMS, resolve  # noqa: E402
from generators import schema as S  # noqa: E402
from generators.generate import synthesize  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
TRAIN_FRAC = 0.8


def patient_split(real: pd.DataFrame, seed: int):
    """Track 3's procedure with a different seed: 80% of patients to train, admissions follow the patient."""
    subjects = np.sort(real[S.ID_COLS[0]].unique())
    rng = np.random.default_rng(seed)
    train_subjects = set(rng.choice(subjects, size=int(round(TRAIN_FRAC * len(subjects))), replace=False))
    is_train = real[S.ID_COLS[0]].isin(train_subjects)
    return real[is_train].reset_index(drop=True), real[~is_train].reset_index(drop=True)


def _quiet():
    warnings.filterwarnings("ignore")


def _run(arm, gen_seed, split_seed):
    from evaluation.fidelity_metrics import run_fidelity_report
    from evaluation.utility_eval import tstr_eval

    real = S.load_real()
    train, holdout = patient_split(real, split_seed)
    generator, params = resolve(arm)

    start = time.time()
    _, _, synth = synthesize(train, generator, gen_seed, params)
    # eval_runner round-trips generator output through a CSV read by S.load_real, which pins icd9_primary
    # as a string (P-005). Do the same here so these numbers sit on the same footing as the benchmark.
    tmp = OUT_DIR / "tmp" / f"{arm}_{gen_seed}_{split_seed}.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    synth.to_csv(tmp, index=False)
    synth_csv = S.load_real(tmp)
    tmp.unlink()

    fidelity = run_fidelity_report(train, synth_csv)
    # A draw whose mortality column has one class has no TSTR AUROC at all. Record it rather than
    # crashing the grid: with 30 splits this happens where 10 splits never showed it.
    degenerate = synth_csv["hospital_expire_flag"].nunique() < 2
    utility = ({"logistic_regression": np.nan, "random_forest": np.nan} if degenerate
               else tstr_eval(synth_csv, holdout))

    return {
        "split_seed": split_seed, "arm": arm, "gen_seed": gen_seed,
        "n_train": len(train), "n_holdout": len(holdout), "degenerate_label": int(degenerate),
        "holdout_deaths": int(holdout["hospital_expire_flag"].sum()),
        "mean_js": fidelity["mean_js_divergence"], "corr_diff": fidelity["correlation_diff"],
        "tstr_lr": utility["logistic_regression"], "tstr_rf": utility["random_forest"],
        "seconds": time.time() - start,
    }


def variance_split(df: pd.DataFrame, metric: str) -> dict:
    """Between-split vs within-split (seed) variance of a metric, per arm and pooled."""
    out = {}
    for arm, g in df.groupby("arm"):
        per_split = g.groupby("split_seed")[metric].mean()
        within = g.groupby("split_seed")[metric].var(ddof=1).mean()
        seeds_per_split = g.groupby("split_seed")[metric].count().mean()
        # The variance of per-split means also contains within/k of seed noise (k seeds per split), so
        # subtract it: one-way random-effects (method of moments) estimate of the split component.
        between = max(per_split.var(ddof=1) - within / seeds_per_split, 0.0)
        total = between + within
        out[arm] = {"between_split_sd": float(np.sqrt(between)), "within_split_sd": float(np.sqrt(within)),
                    "split_share": float(between / total) if total else float("nan")}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=3)
    # REPORTED_ARMS, not every key of ARMS: the DP extension arms produce a single-class mortality
    # column on 20-45% of draws, which is a finding in research/dp_study.py and noise here.
    parser.add_argument("--arms", nargs="+", default=list(REPORTED_ARMS))
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args(argv)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    tasks = [(arm, s, 1000 + k) for arm in args.arms for s in range(args.seeds) for k in range(args.splits)]
    cost = {"tabular_diffusion": 46, "tabular_diffusion_reg": 20, "ctgan": 27, "ctgan_small": 20, "tvae": 8}
    tasks.sort(key=lambda t: -cost.get(t[0], 1))
    print(f"{len(tasks)} fits on {args.workers} workers", flush=True)

    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = {pool.submit(_run, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 20 == 0 or i == len(tasks):
                print(f"[{i:4d}/{len(tasks)}] {time.time() - start:6.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(OUT_DIR / "split_study.csv", index=False)   # crash insurance

    df = pd.DataFrame(rows).sort_values(["arm", "split_seed", "gen_seed"], ignore_index=True)
    df.to_csv(OUT_DIR / "split_study.csv", index=False)

    holdouts = df.groupby("split_seed")[["n_holdout", "holdout_deaths"]].first()
    print(f"\nholdout size {holdouts.n_holdout.min()}-{holdouts.n_holdout.max()} rows, "
          f"deaths {holdouts.holdout_deaths.min()}-{holdouts.holdout_deaths.max()} "
          f"(reported split: 35 rows, 6 deaths)")

    summary = df.groupby("arm").agg(tstr_rf_mean=("tstr_rf", "mean"), tstr_rf_sd=("tstr_rf", "std"),
                                    tstr_lr_mean=("tstr_lr", "mean"), tstr_lr_sd=("tstr_lr", "std"),
                                    mean_js=("mean_js", "mean"), corr_diff=("corr_diff", "mean"))
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n" + summary.sort_values("tstr_rf_mean", ascending=False).to_string())

    print("\nvariance of TSTR (RF): between splits vs between seeds within a split")
    for arm, v in sorted(variance_split(df, "tstr_rf").items(), key=lambda kv: -kv[1]["split_share"]):
        print(f"  {arm:28s} split sd {v['between_split_sd']:.3f} | seed sd {v['within_split_sd']:.3f} | "
              f"split share {v['split_share']:.0%}")
    summary.to_csv(OUT_DIR / "split_study_summary.csv")
    print(f"\nwrote {OUT_DIR}/split_study.csv and _summary.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
