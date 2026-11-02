"""Fidelity and utility of the models the attacks actually target, and a multivariate fidelity test, on MIMIC-III.

Post-course extension work, in answer to two review comments. Privacy is measured on shadow generators fitted on
47 admissions, while the benchmark's fidelity and utility come from fits on all 94. This scores both fit sizes
with the same metrics, so the trade-off can be read on one model. It also adds a classifier two-sample test
and the pMSE: a 5-fold random forest separates real from synthetic rows on the utility features (without
the code lists), and both statistics are computed from its held-out probabilities, because correlation error
sees only numeric columns.

Two references for the test: real training rows against real held-out rows, and two disjoint random halves of
the training rows (47 against 47), the level at which real data cannot be told from real data.

    python -m research.fit_size_study --workers 4
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
ARMS = REPORTED_ARMS + ["sequential_cart"]
TARGET = "hospital_expire_flag"


def _quiet():
    warnings.filterwarnings("ignore")


def two_sample(real, other, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from evaluation.utility_eval import build_features

    both = pd.concat([real, other], ignore_index=True)
    X, *_ = build_features(both, fit=True)
    y = np.r_[np.zeros(len(real)), np.ones(len(other))]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p = cross_val_predict(RandomForestClassifier(200, random_state=seed), X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p)), float(np.mean((p - y.mean()) ** 2))


def _context():
    real = S.load_real()
    return (real[real[S.SPLIT_COL] == S.TRAIN_SPLIT].reset_index(drop=True),
            real[real[S.SPLIT_COL] != S.TRAIN_SPLIT].reset_index(drop=True))


def _reference(seed):
    train, holdout = _context()
    idx = np.random.default_rng(seed).permutation(len(train))
    a, b = train.iloc[idx[:47]], train.iloc[idx[47:94]]
    half = two_sample(a, b, seed)
    shift = two_sample(train, holdout, seed)
    return [{"arm": "real_halves", "fit_rows": 47, "seed": seed, "c2st": half[0], "pmse": half[1]},
            {"arm": "real_holdout", "fit_rows": 94, "seed": seed, "c2st": shift[0], "pmse": shift[1]}]


def _run(arm, seed, fit_rows):
    from evaluation.fidelity_metrics import run_fidelity_report
    from evaluation.utility_eval import tstr_eval

    train, holdout = _context()
    if fit_rows < len(train):
        # the draw learned_attack.py's first shadow of this seed uses
        fit = train.iloc[np.random.default_rng(seed).choice(len(train), size=fit_rows, replace=False)].reset_index(drop=True)
    else:
        fit = train
    generator, params = resolve(arm)
    start = time.time()
    _, _, synth = synthesize(fit, generator, seed, params)
    tmp = OUT_DIR / "tmp" / f"fit_size_{arm}_{seed}_{fit_rows}.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    synth.to_csv(tmp, index=False)
    synth = S.load_real(tmp)
    tmp.unlink()
    fidelity = run_fidelity_report(fit, synth)
    degenerate = synth[TARGET].nunique() < 2
    u = {"logistic_regression": np.nan, "random_forest": np.nan} if degenerate else tstr_eval(synth, holdout)
    c2st, pmse = two_sample(fit, synth, seed)
    return [{"arm": arm, "fit_rows": fit_rows, "seed": seed, "mean_js": fidelity["mean_js_divergence"],
             "corr_diff": fidelity["correlation_diff"], "tstr_lr": u["logistic_regression"], "tstr_rf": u["random_forest"],
             "death_rate": float(synth[TARGET].mean()), "degenerate_label": int(degenerate), "c2st": c2st, "pmse": pmse,
             "seconds": time.time() - start}]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="fit_size_study")
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        os.environ[var] = "1"

    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [42 + s for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = [pool.submit(_reference, s) for s in seeds]
        futures += [pool.submit(_run, a, s, f) for a in args.arms for s in seeds for f in (94, 47)]
        for i, fut in enumerate(as_completed(futures), 1):
            rows += fut.result()
            if i % 20 == 0 or i == len(futures):
                print(f"[{i:3d}/{len(futures)}] {time.time() - start:6.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / f"{args.out}.csv", index=False)
    df = pd.DataFrame(rows)
    df["tstr_mean"] = df[["tstr_lr", "tstr_rf"]].mean(axis=1)
    cols = ["mean_js", "corr_diff", "tstr_mean", "death_rate", "degenerate_label", "c2st", "pmse"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n" + df.groupby(["arm", "fit_rows"])[cols].mean().to_string())
    print(f"\nwrote {OUT_DIR}/{args.out}.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
