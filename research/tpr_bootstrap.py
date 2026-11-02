"""TPR at 1% FPR with an interval clustered by record, on the saved learned-attack features.

Post-course extension work, in answer to a review comment. The roughly 1,500 pooled non-member scores of a
32-shadow run come from 94 distinct records, so they are not independent. This refits the attack models only,
collects the leave-one-shadow-out scores, and bootstraps whole records: a record that is resampled brings all of
its member and non-member scores with it. Each replicate averages the TPR over the three seeds of a setting.

    python -m research.tpr_bootstrap --resamples 1000 --workers 4
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.learned_attack import FEATURE_DIR, OUT_DIR  # noqa: E402

FPR = 0.01
ARMS = ["independent_marginals", "gaussian_copula", "gaussian_copula_shrink025", "gaussian_copula_shrink010", "ctgan",
        "ctgan_small", "tvae", "tabular_diffusion", "tabular_diffusion_reg", "sequential_cart"]


def _gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, early_stopping=False, random_state=0)


def loso_scores(X, member, make_model):
    K, n = X.shape[:2]
    scores = np.empty((K, n))
    for j in range(K):
        rest = [k for k in range(K) if k != j]
        scores[j] = make_model().fit(X[rest].reshape(-1, X.shape[2]), member[rest].ravel()).predict_proba(X[j])[:, 1]
    return scores


def tpr_at(scores, member, weights):
    """Pooled TPR at FPR <= 1%, with records weighted by how often the bootstrap drew them."""
    w = np.broadcast_to(weights, scores.shape).ravel()
    s, m = scores.ravel(), member.ravel()
    order = np.argsort(-s, kind="stable")
    w, m = w[order], m[order]
    tp, fp = np.cumsum(w * m), np.cumsum(w * ~m)
    pos, neg = tp[-1], fp[-1]
    ok = fp <= FPR * neg
    return float(tp[ok].max() / pos) if ok.any() and pos > 0 else 0.0


def _run(arm, seed, resamples):
    from sklearn.linear_model import LogisticRegression

    warnings.filterwarnings("ignore")
    z = np.load(FEATURE_DIR / f"{arm}_{seed}.npz")
    X, member = z["X"], z["member"]
    attacks = {"gbm": loso_scores(X, member, _gbm), "dist_gower": loso_scores(X[..., [6]], member, LogisticRegression)}
    rng = np.random.default_rng(seed)
    n = X.shape[1]
    draws = [np.bincount(rng.integers(n, size=n), minlength=n).astype(float) for _ in range(resamples)]
    return arm, seed, {k: (tpr_at(s, member, np.ones(n)), np.array([tpr_at(s, member, d) for d in draws])) for k, s in attacks.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, a, s, args.resamples) for a in args.arms for s in (42, 43, 44)]
        for fut in as_completed(futures):
            arm, seed, r = fut.result()
            results.setdefault(arm, []).append(r)
            print(arm, seed, flush=True)
    rows = []
    for arm, runs in results.items():
        for attack in runs[0]:
            point = np.mean([r[attack][0] for r in runs])
            reps = np.mean([r[attack][1] for r in runs], axis=0)
            rows.append({"arm": arm, "attack": attack, "tpr1": point, "lo": float(np.percentile(reps, 2.5)), "hi": float(np.percentile(reps, 97.5))})
    df = pd.DataFrame(rows).sort_values(["attack", "arm"])
    df.to_csv(OUT_DIR / "tpr_bootstrap.csv", index=False)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
