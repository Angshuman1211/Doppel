"""Membership attack against cohort size on UCI Diabetes 130-US.

Post-course extension work. The size sweep (`research/size_sweep.py`) measures memorisation with the
near-copy rate, a distance statistic of the kind the paper argues is not a substitute for an attack. This
runs the benchmark's shadow-model attack at several cohort sizes instead, on the same cohorts the size
sweep draws (same holdout, same patient draws for seeds 0 and 1).

Design, as in `evaluation/membership_inference.py` on MIMIC-III: for each cohort, 8 shadow generators are
each fitted on a random half of the cohort's rows and release as many rows as they were fitted on; every
cohort row is described by minus its Gower distance to the nearest released row; a logistic-regression
attack is trained on 7 shadows and scored on the 8th, in turn. Reported: mean AUROC over the held-out
shadows and the pooled TPR at 1% and 5% FPR (highest TPR at an FPR no higher than the target).

    python -m research.size_attack --sizes 50 250 1000 5000 --seeds 2 --workers 6
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

from generators.codec import KIND_CONTINUOUS, KIND_INTEGER  # noqa: E402
from generators.diagnostics import _nearest_gower  # noqa: E402
from generators.generate import load_generator  # noqa: E402
from functools import lru_cache  # noqa: E402

from research.size_sweep import ARMS  # noqa: E402
from research.size_sweep import load_context as _load_context  # noqa: E402

# One read of the 100k-row file per worker process, not one per shadow fit (the first run ran out of memory).
load_context = lru_cache(maxsize=1)(_load_context)

OUT_DIR = ROOT / "research" / "output"
N_SHADOW = 8
FPR_LEVELS = (0.01, 0.05)


def cohort(pool, size, seed):
    """The size sweep's cohort draw for this size and seed."""
    rng = np.random.default_rng(10_000 + seed)
    patients = np.sort(pool["patient_nbr"].unique())
    chosen = set(rng.choice(patients, size=min(size, len(patients)), replace=False))
    return pool[pool["patient_nbr"].isin(chosen)].reset_index(drop=True)


def _quiet():
    warnings.filterwarnings("ignore")


def _shadow(arm, size, seed, shadow_id):
    """Fit one shadow generator on a random half of the cohort; return each cohort row's distance score."""
    _, codec, _, pool = load_context()
    rows = cohort(pool, size, seed)
    rng = np.random.default_rng([seed, shadow_id, size])
    members = np.zeros(len(rows), dtype=int)
    members[rng.choice(len(rows), size=len(rows) // 2, replace=False)] = 1
    member_df = rows[members == 1].reset_index(drop=True)

    generator, params = ARMS[arm]
    fit_rng, sample_rng = (np.random.default_rng(s) for s in np.random.SeedSequence([seed, shadow_id]).spawn(2))
    start = time.time()
    model = load_generator(generator)(**params).fit(codec.encode(member_df), codec.spec, fit_rng)
    synth = codec.decode(model.sample(len(member_df), sample_rng))

    numeric = [c for c, k in codec.spec.items() if k in (KIND_INTEGER, KIND_CONTINUOUS)]
    discrete = [c for c in codec.spec if c not in numeric]
    cand, syn = codec.encode(rows), codec.encode(synth)
    scale = (cand[numeric].max() - cand[numeric].min()).replace(0, 1).to_numpy(float)
    dist = _nearest_gower(cand, syn, numeric, discrete, scale)
    return {"arm": arm, "size": size, "seed": seed, "shadow": shadow_id, "rows": len(rows),
            "score": -dist, "member": members, "seconds": time.time() - start}


def attack(shadows):
    """Leave-one-shadow-out logistic regression on the distance score, as in the MIMIC benchmark."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve

    aucs, labels, scores = [], [], []
    for held in shadows:
        train = [s for s in shadows if s is not held]
        x = np.concatenate([s["score"] for s in train]).reshape(-1, 1)
        y = np.concatenate([s["member"] for s in train])
        model = LogisticRegression().fit(x, y)
        p = model.predict_proba(held["score"].reshape(-1, 1))[:, 1]
        aucs.append(roc_auc_score(held["member"], p))
        labels.append(held["member"]); scores.append(p)
    fpr, tpr, _ = roc_curve(np.concatenate(labels), np.concatenate(scores))
    return {"auroc": float(np.mean(aucs)), **{f"tpr{int(l * 100)}": float(tpr[fpr <= l].max()) for l in FPR_LEVELS}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 250, 1000, 5000])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--arms", nargs="+", default=["independent_marginals", "gaussian_copula_shrink010",
                                                      "tvae", "tabular_diffusion"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default="size_attack")
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    tasks = [(a, n, s, k) for a in args.arms for n in args.sizes for s in range(args.seeds) for k in range(N_SHADOW)]
    tasks.sort(key=lambda t: -t[1] * (1 if t[0] in ("independent_marginals", "gaussian_copula_shrink010") else 20))
    print(f"{len(tasks)} shadow fits on {args.workers} workers", flush=True)
    done, start = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = [pool.submit(_shadow, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            done.append(fut.result())
            if i % 16 == 0 or i == len(tasks):
                print(f"[{i:3d}/{len(tasks)}] {time.time() - start:6.0f}s", flush=True)

    rows = []
    for (arm, size, seed), group in pd.DataFrame(done).groupby(["arm", "size", "seed"]):
        shadows = [r for r in done if (r["arm"], r["size"], r["seed"]) == (arm, size, seed)]
        rows.append({"arm": arm, "size_patients": size, "seed": seed, "train_rows": shadows[0]["rows"], **attack(shadows)})
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{args.out}.csv", index=False)
    summary = df.groupby(["arm", "size_patients"]).agg(train_rows=("train_rows", "mean"), auroc=("auroc", "mean"),
                                                        auroc_sd=("auroc", "std"), tpr1=("tpr1", "mean"),
                                                        tpr5=("tpr5", "mean"))
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n" + summary.to_string())
    summary.to_csv(OUT_DIR / f"{args.out}_summary.csv")
    print(f"\nwrote {OUT_DIR}/{args.out}.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
