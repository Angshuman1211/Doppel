"""Learned membership attacks on MIMIC-III, beside the benchmark's distance attack on the same shadows.

Post-course extension work, in answer to a review comment. The benchmark's attacks are a logistic regression
on one nearest-neighbour distance (three distances, worst case taken), checked against copy and
unseen-record releases. A reviewer may ask whether a stronger, learned attack finds more. This runs two:

  gbm    gradient boosting on 18 features per record: for each of the three distances, the 1st, 2nd and
         3rd nearest synthetic distances, the mean of the 5 and of the 10 nearest, and the record's distance
         to its nearest other real record (so the model can tell an unusual record from an exposed one)
  lira   a per-record likelihood-ratio attack after Carlini et al. (2022): for each record, Gaussians fitted
         to its log nearest-neighbour distance (three distances) over the shadows where it was in and where
         it was out, with a variance shared across records, and the log likelihood ratio as the score

and, on the same shadows, the benchmark's attack (logistic regression per distance, worst case) and a
logistic regression on the 18 features, so that the gain from features and from the model can be told
apart. Every attack is scored leave-one-shadow-out: fitted on 31 shadows and scored on the 32nd, in turn.

    python -m research.learned_attack --seeds 3 --workers 4
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

from generators import schema as S  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
FEATURE_DIR = OUT_DIR / "learned_attack_features"
ARMS = ["exact_copy", "disjoint_real", "independent_marginals", "gaussian_copula_shrink010", "tvae",
        "tabular_diffusion", "tabular_diffusion_reg"]
DISTANCES = ["numeric4", "gower", "codes"]
N_SHADOW = 32
FPR = 0.01


def _quiet():
    warnings.filterwarnings("ignore")


def _distance_matrices(train, synth, scaler):
    from scipy.spatial.distance import cdist

    from evaluation.membership_inference import NUMERIC_COLS, codes_distances, gower_distances

    return {"numeric4": cdist(scaler.transform(train[NUMERIC_COLS]), scaler.transform(synth[NUMERIC_COLS])),
            "gower": gower_distances(train, synth),
            "codes": codes_distances(train, synth)}


def shadow_features(arm, seed, n_shadow=N_SHADOW):
    """Per shadow and record: 18 features, the three log nearest-neighbour distances and membership."""
    from sklearn.preprocessing import StandardScaler

    from evaluation.eval_runner import real_generator_fn
    from evaluation.membership_inference import NUMERIC_COLS
    from evaluation.mia_calibration import build_generator

    real = S.load_real()
    train = real[real[S.SPLIT_COL] == S.TRAIN_SPLIT].reset_index(drop=True)
    holdout = real[real[S.SPLIT_COL] != S.TRAIN_SPLIT].reset_index(drop=True)
    generate = (build_generator(arm, holdout) if arm in ("exact_copy", "disjoint_real")
                else real_generator_fn(arm, disk_cache=False))
    scaler = StandardScaler().fit(train[NUMERIC_COLS])
    n = len(train)

    # each record's distance to its nearest other real record, per distance
    own = _distance_matrices(train, train, scaler)
    typical = []
    for name in DISTANCES:
        d = own[name].copy()
        np.fill_diagonal(d, np.inf)
        typical.append(d.min(axis=1))

    rng = np.random.default_rng(seed)
    feats, logd, member = [], [], []
    for k in range(n_shadow):
        idx = rng.choice(n, size=n // 2, replace=False)
        synth = generate(train.iloc[idx].reset_index(drop=True), NUMERIC_COLS, len(idx), seed * 1000 + k)
        cols = []
        for name, t in zip(DISTANCES, typical):
            d = np.sort(_distance_matrices(train, synth, scaler)[name], axis=1)
            cols += [d[:, 0], d[:, 1], d[:, 2], d[:, :5].mean(axis=1), d[:, :10].mean(axis=1), t]
        f = np.column_stack(cols)
        feats.append(f)
        logd.append(np.log(f[:, [0, 6, 12]] + 1e-6))
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        member.append(m)
    return np.stack(feats), np.stack(logd), np.stack(member)


def _score(labels, scores):
    from sklearn.metrics import roc_auc_score, roc_curve

    aucs = [roc_auc_score(y, s) for y, s in zip(labels, scores)]
    fpr, tpr, _ = roc_curve(np.concatenate(labels), np.concatenate(scores))
    return float(np.mean(aucs)), float(tpr[fpr <= FPR].max())


def leave_one_shadow_out(X, member, make_model):
    """X: (shadows, records, features). Fit on all shadows but one, score the held-out one."""
    K = X.shape[0]
    labels, scores = [], []
    for j in range(K):
        rest = [k for k in range(K) if k != j]
        model = make_model()
        model.fit(X[rest].reshape(-1, X.shape[2]), member[rest].ravel())
        labels.append(member[j])
        scores.append(model.predict_proba(X[j])[:, 1])
    return _score(labels, scores)


def lira(logd, member):
    """Per-record Gaussians on the log nearest-neighbour distances, fitted on the other shadows, with a
    variance per feature shared across records (Carlini et al.'s choice when shadows are few)."""
    K = logd.shape[0]
    labels, scores = [], []
    for j in range(K):
        rest = np.array([k for k in range(K) if k != j])
        x, m = logd[rest], member[rest][..., None]
        out = []
        for side in (m, ~m):
            cnt = side.sum(axis=0)
            glob = (x * side).sum(axis=(0, 1)) / side.sum(axis=(0, 1))
            mu = np.where(cnt > 0, (x * side).sum(axis=0) / np.maximum(cnt, 1), glob)
            var = (((x - mu) ** 2) * side).sum(axis=(0, 1)) / side.sum(axis=(0, 1)) + 1e-6
            out.append((mu, var))
        (mu_in, v_in), (mu_out, v_out) = out
        z = logd[j]
        ll = (-(z - mu_in) ** 2 / (2 * v_in) - 0.5 * np.log(v_in)) - (-(z - mu_out) ** 2 / (2 * v_out) - 0.5 * np.log(v_out))
        labels.append(member[j])
        scores.append(ll.sum(axis=1))
    return _score(labels, scores)


def _run(arm, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    start = time.time()
    X, logd, member = shadow_features(arm, seed)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_DIR / f"{arm}_{seed}.npz", X=X, logd=logd, member=member)
    row = {"arm": arm, "seed": seed, "n_shadow": X.shape[0]}
    for i, name in enumerate(DISTANCES):
        # the benchmark's attack: logistic regression on minus the nearest distance (sign is irrelevant to LR)
        row[f"dist_{name}"], row[f"dist_{name}_tpr1"] = leave_one_shadow_out(X[..., [6 * i]], member, LogisticRegression)
    best = max(DISTANCES, key=lambda n: row[f"dist_{n}"])
    row["dist_worst"], row["dist_worst_tpr1"], row["dist_worst_attack"] = row[f"dist_{best}"], row[f"dist_{best}_tpr1"], best
    row["feat_lr"], row["feat_lr_tpr1"] = leave_one_shadow_out(
        X, member, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
    row["gbm"], row["gbm_tpr1"] = leave_one_shadow_out(
        X, member, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                          early_stopping=False, random_state=0))
    row["lira"], row["lira_tpr1"] = lira(logd, member)
    row["seconds"] = time.time() - start
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="learned_attack")
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    tasks = [(a, 42 + s) for a in args.arms for s in range(args.seeds)]
    cost = {"tabular_diffusion": 10, "tabular_diffusion_reg": 4, "tvae": 2}
    tasks.sort(key=lambda t: -cost.get(t[0], 1))
    print(f"{len(tasks)} runs ({len(tasks) * N_SHADOW} shadow fits) on {args.workers} workers", flush=True)
    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = [pool.submit(_run, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            print(f"[{i:2d}/{len(tasks)}] {rows[-1]['arm']} {rows[-1]['seed']} {time.time() - start:6.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / f"{args.out}.csv", index=False)

    df = pd.DataFrame(rows)
    cols = ["dist_worst", "feat_lr", "gbm", "lira", "dist_worst_tpr1", "feat_lr_tpr1", "gbm_tpr1", "lira_tpr1"]
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print("\n" + df.groupby("arm")[cols].mean().reindex(args.arms).to_string())
    print(f"\nwrote {OUT_DIR}/{args.out}.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
