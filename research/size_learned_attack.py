"""Learned membership attacks against cohort size on UCI Diabetes 130-US, with copy and unseen-record controls.

Post-course extension work, in answer to a review comment. `research/size_attack.py` showed with the Gower
distance attack that the diffusion model's exposure falls to near chance by 5,000 patients. On MIMIC-III a
learned attack calibrated per record found more than any distance attack (`research/learned_attack.py`), so
the cohort-size curve has to be retested with one.

Same cohorts as the size sweep and the size attack. For each cohort, 16 shadow generators are each fitted on a
random half of the cohort's rows and release as many rows as they were fitted on; the first 8 use exactly the
member draws and fit seeds of `size_attack.py`. Every cohort row is described in each shadow by its 1st, 2nd
and 3rd nearest Gower distances to the release, the mean of its 5 and 10 nearest, and its Gower distance to
the nearest other cohort row. On the same shadows: the distance attack (logistic regression on the nearest
distance), gradient boosting on the 6 features, and the likelihood-ratio attack on the log nearest distance,
all scored leave-one-shadow-out. Two control releases run through the same harness: the members themselves
(copy) and as many rows drawn from the fixed holdout (unseen records).

    python -m research.size_learned_attack --sizes 50 250 1000 5000 --seeds 2 --workers 3
"""

import argparse
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.codec import KIND_CONTINUOUS, KIND_INTEGER  # noqa: E402
from generators.generate import load_generator  # noqa: E402
from research.learned_attack import leave_one_shadow_out, lira  # noqa: E402
from research.size_attack import cohort  # noqa: E402
from research.size_sweep import ARMS  # noqa: E402
from research.size_sweep import load_context as _load_context  # noqa: E402

load_context = lru_cache(maxsize=1)(_load_context)

OUT_DIR = ROOT / "research" / "output"
FEATURE_DIR = OUT_DIR / "size_learned_features"
N_SHADOW = 16
K = 10
CONTROLS = ("exact_copy", "disjoint_real")
DEFAULT_ARMS = ["exact_copy", "disjoint_real", "independent_marginals", "gaussian_copula_shrink010", "tvae",
                "tabular_diffusion"]


def _quiet():
    warnings.filterwarnings("ignore")


def _encoded(codec, df, numeric, discrete, scale, levels):
    enc = codec.encode(df)
    num = enc[numeric].to_numpy(float) / scale
    disc = np.column_stack([pd.Categorical(enc[c].astype(str), categories=levels[c]).codes for c in discrete])
    return num, disc


def knn_gower(q, t, n_cols, k=K, exclude_self=False, chunk=128):
    """The k smallest Gower distances from each query row to the target rows, in chunks to bound memory."""
    (qn, qd), (tn, td) = q, t
    out = np.empty((len(qn), k), dtype=np.float32)
    for lo in range(0, len(qn), chunk):
        hi = min(lo + chunk, len(qn))
        d = np.abs(qn[lo:hi, None, :] - tn[None, :, :]).sum(-1)
        d += (qd[lo:hi, None, :] != td[None, :, :]).sum(-1)
        if exclude_self:
            d[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
        kk = min(k, d.shape[1] - 1)
        part = np.partition(d, kk, axis=1)[:, :k] if d.shape[1] > k else d
        out[lo:hi, :part.shape[1]] = np.sort(part, axis=1)[:, :k] / n_cols
    return out


@lru_cache(maxsize=4)
def _setup(size, seed):
    _, codec, holdout, pool = load_context()
    rows = cohort(pool, size, seed)
    numeric = [c for c, kd in codec.spec.items() if kd in (KIND_INTEGER, KIND_CONTINUOUS)]
    discrete = [c for c in codec.spec if c not in numeric]
    enc_rows = codec.encode(rows)
    scale = (enc_rows[numeric].max() - enc_rows[numeric].min()).replace(0, 1).to_numpy(float)
    levels = {c: sorted(set(enc_rows[c].astype(str)) | set(codec.encode(holdout)[c].astype(str))) for c in discrete}
    return codec, holdout, rows, numeric, discrete, scale, levels


def _typical(size, seed):
    codec, _, rows, numeric, discrete, scale, levels = _setup(size, seed)
    cand = _encoded(codec, rows, numeric, discrete, scale, levels)
    return ("typical", size, seed, knn_gower(cand, cand, len(numeric) + len(discrete), k=1, exclude_self=True)[:, 0])


def _shadow(arm, size, seed, shadow_id):
    codec, holdout, rows, numeric, discrete, scale, levels = _setup(size, seed)
    rng = np.random.default_rng([seed, shadow_id, size])  # as in size_attack.py
    members = np.zeros(len(rows), dtype=int)
    members[rng.choice(len(rows), size=len(rows) // 2, replace=False)] = 1
    member_df = rows[members == 1].reset_index(drop=True)
    start = time.time()
    if arm == "exact_copy":
        synth = member_df
    elif arm == "disjoint_real":
        synth = holdout.sample(n=len(member_df), replace=len(member_df) > len(holdout),
                               random_state=1000 * seed + shadow_id).reset_index(drop=True)
    else:
        generator, params = ARMS[arm]
        fit_rng, sample_rng = (np.random.default_rng(s) for s in np.random.SeedSequence([seed, shadow_id]).spawn(2))
        model = load_generator(generator)(**params).fit(codec.encode(member_df), codec.spec, fit_rng)
        synth = codec.decode(model.sample(len(member_df), sample_rng))
    cand = _encoded(codec, rows, numeric, discrete, scale, levels)
    syn = _encoded(codec, synth, numeric, discrete, scale, levels)
    knn = knn_gower(cand, syn, len(numeric) + len(discrete))
    return (arm, size, seed, shadow_id, knn, members, time.time() - start)


def evaluate(knn, typical, member):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([knn[..., :3], knn[..., :5].mean(-1, keepdims=True), knn.mean(-1, keepdims=True),
                        np.broadcast_to(typical[None, :, None], knn.shape[:2] + (1,))], axis=-1).astype(float)
    member = member.astype(bool)
    row = {}
    row["dist"], row["dist_tpr1"] = leave_one_shadow_out(X[..., [0]], member, LogisticRegression)
    row["dist_8"], row["dist_8_tpr1"] = leave_one_shadow_out(X[:8, :, [0]], member[:8], LogisticRegression)
    row["gbm"], row["gbm_tpr1"] = leave_one_shadow_out(
        X, member, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                          early_stopping=False, random_state=0))
    row["lira"], row["lira_tpr1"] = lira(np.log(X[..., [0]] + 1e-6), member)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 250, 1000, 5000])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default="size_learned_attack")
    args = parser.parse_args(argv)
    # ctgan's DataTransformer runs joblib Parallel(n_jobs=-1), which would start a process per core in every worker
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        os.environ[var] = "1"

    cost = {"tabular_diffusion": 20, "tvae": 20}
    tasks = [(a, n, s, k) for a in args.arms for n in args.sizes for s in range(args.seeds) for k in range(N_SHADOW)]
    tasks.sort(key=lambda t: -t[1] * cost.get(t[0], 1))
    print(f"{len(tasks)} shadow releases on {args.workers} workers", flush=True)
    shadows, typical, start = {}, {}, time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = [pool.submit(_typical, n, s) for n in args.sizes for s in range(args.seeds)]
        futures += [pool.submit(_shadow, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r[0] == "typical":
                typical[r[1:3]] = r[3]
            else:
                shadows.setdefault(r[:3], {})[r[3]] = (r[4], r[5])
            if i % 25 == 0 or i == len(futures):
                print(f"[{i:4d}/{len(futures)}] {time.time() - start:6.0f}s", flush=True)

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for (arm, size, seed), group in sorted(shadows.items()):
        knn = np.stack([group[k][0] for k in sorted(group)])
        member = np.stack([group[k][1] for k in sorted(group)])
        np.savez_compressed(FEATURE_DIR / f"{arm}_{size}_{seed}.npz", knn=knn, typical=typical[(size, seed)], member=member)
        rows.append({"arm": arm, "size_patients": size, "seed": seed, "train_rows": knn.shape[1],
                     **evaluate(knn, typical[(size, seed)], member)})
        print(arm, size, seed, flush=True)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{args.out}.csv", index=False)
    cols = ["dist_8", "dist", "gbm", "lira", "dist_tpr1", "gbm_tpr1", "lira_tpr1"]
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print("\n" + df.groupby(["arm", "size_patients"])[cols].mean().to_string())
    print(f"\nwrote {OUT_DIR}/{args.out}.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
