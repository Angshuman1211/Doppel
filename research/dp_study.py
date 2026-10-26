"""Differentially private baselines at two budgets (improvement plan item 4).

Post-course extension work. The reported benchmark has no arm with a formal guarantee, so every privacy
claim in it is attack-relative: it says an attack failed, not that leakage is bounded. This scores the DP
arms on the same metrics as the benchmark.

Run separately from `evaluation.eval_runner` for one reason: at these budgets the noisy mortality
histogram sometimes contains a single class, and the utility classifier then has nothing to fit. That is
a property of the mechanism at this sample size, so it is recorded as a degenerate run rather than
raised as an error, which is what the runner does.

    python -m research.dp_study --seeds 20 --workers 6
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

from evaluation.arms import resolve  # noqa: E402
from generators import schema as S  # noqa: E402
from generators.generate import synthesize  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
ARMS = ["dp_marginals_eps1", "dp_marginals_eps8", "dp_copula_eps1", "dp_copula_eps8",
        "dp_bayesnet_eps1", "dp_bayesnet_eps8", "dp_aim_eps1", "dp_aim_eps8",
        "gaussian_copula_shrink010", "independent_marginals"]
TARGET = "hospital_expire_flag"
N_SHADOW = 8
CAP_SEED = 0            # fixed, so every arm and seed sees the same capped cohort


def cap_admissions(train, max_per_patient: int, seed: int = CAP_SEED):
    """Keep at most `max_per_patient` admissions per patient.

    This is what turns a per-admission guarantee into a per-patient one. With a cap of 1 each patient
    contributes exactly one row, so replacing a patient *is* replacing a row and the mechanisms' existing
    bounded-DP guarantee already holds at patient level, with no extra noise. The cap is public by
    construction, which matters: reading the real maximum off the data would itself leak.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for idx in train.groupby(S.ID_COLS[0]).indices.values():
        idx = np.asarray(idx)
        keep.extend(idx if len(idx) <= max_per_patient
                    else rng.choice(idx, max_per_patient, replace=False))
    return train.iloc[np.sort(np.asarray(keep))].reset_index(drop=True)


def _quiet():
    warnings.filterwarnings("ignore")


def _run(arm, seed, cap=None):
    from evaluation.eval_runner import cached, real_generator_fn
    from evaluation.fidelity_metrics import run_fidelity_report
    from evaluation.membership_inference import NUMERIC_COLS, gower_distances, run_membership_inference
    from evaluation.utility_eval import tstr_eval

    real = S.load_real()
    train = real[real[S.SPLIT_COL] == S.TRAIN_SPLIT].reset_index(drop=True)
    if cap:
        train = cap_admissions(train, cap)
    holdout = real[real[S.SPLIT_COL] != S.TRAIN_SPLIT].reset_index(drop=True)
    generator, params = resolve(arm)

    start = time.time()
    _, _, synth = synthesize(train, generator, seed, params)
    tmp = OUT_DIR / "tmp" / f"dp_{arm}_{seed}.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    synth.to_csv(tmp, index=False)
    synth = S.load_real(tmp)
    tmp.unlink()

    fidelity = run_fidelity_report(train, synth)
    row = {"arm": arm, "seed": seed, "cap": cap or 0, "train_rows": len(train),
           "mean_js": fidelity["mean_js_divergence"],
           "corr_diff": fidelity["correlation_diff"],
           "synth_death_rate": float(synth[TARGET].mean()),
           "degenerate_label": synth[TARGET].nunique() < 2}

    if row["degenerate_label"]:
        row["tstr_lr"] = row["tstr_rf"] = np.nan
    else:
        u = tstr_eval(synth, holdout)
        row["tstr_lr"], row["tstr_rf"] = u["logistic_regression"], u["random_forest"]

    gower = run_membership_inference(train, cached(real_generator_fn(arm)), NUMERIC_COLS,
                                     n_shadow=N_SHADOW, seed=seed, distance_fn=gower_distances)
    row["mia_gower"] = gower["mean_attack_auroc"]
    row["seconds"] = time.time() - start
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    # A fixed output path means a run over a subset of arms silently replaces the full study's results
    # instead of sitting beside them.
    parser.add_argument("--out", default="dp_study", help="stem for the CSVs under research/output")
    parser.add_argument("--cap", type=int, default=None,
                        help="max admissions per patient; only 1 is supported, and it makes the "
                             "arms' existing bounded-DP guarantee a per-patient one")
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    if args.cap is not None and args.cap != 1:
        # A cap of m > 1 lets one patient move m rows, so every mechanism's L2 sensitivity would have to
        # be multiplied by m (and AIM's epsilon divided by it). None of that is implemented, so running
        # it would produce a per-admission guarantee on a capped cohort and look like a per-patient one.
        parser.error(f"--cap {args.cap} is not supported: only --cap 1 gives a per-patient guarantee "
                     "without rescaling every mechanism's sensitivity, which is not implemented.")

    tasks = [(a, 42 + s, args.cap) for a in args.arms for s in range(args.seeds)]
    print(f"{len(tasks)} runs on {args.workers} workers", flush=True)
    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = [pool.submit(_run, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 20 == 0 or i == len(tasks):
                print(f"[{i:3d}/{len(tasks)}] {time.time() - start:5.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values(["arm", "seed"], ignore_index=True)
    df.to_csv(OUT_DIR / f"{args.out}.csv", index=False)

    summary = df.groupby("arm").agg(
        train_rows=("train_rows", "mean"), mean_js=("mean_js", "mean"), corr_diff=("corr_diff", "mean"),
        tstr_rf=("tstr_rf", "mean"), tstr_rf_sd=("tstr_rf", "std"),
        mia_gower=("mia_gower", "mean"),
        death_rate=("synth_death_rate", "mean"),
        degenerate=("degenerate_label", "mean"), n=("seed", "count"))
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n" + summary.to_string())
    print("\nreal train death rate: 0.3617 | non-private copula JSD about 0.016")
    summary.to_csv(OUT_DIR / f"{args.out}_summary.csv")
    print(f"wrote {OUT_DIR}/{args.out}.csv and _summary.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
