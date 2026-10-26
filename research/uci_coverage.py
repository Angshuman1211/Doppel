"""How synthetic rows sit on the training rows, by cohort size, on UCI Diabetes 130-US.

Post-course extension work, in answer to a review comment. The size attack (`research/size_attack.py`)
found that TVAE keeps a high near-copy rate at every cohort size while its membership exposure falls to the
floor from 362 training rows on. The paper explained that by mode collapse onto a few training rows, an
explanation measured on MIMIC-III (31 of 94 admissions reached) but not on UCI. This measures it on UCI.

For each cohort (the size sweep's draws), the generator is fitted on all the cohort's rows and releases as
many rows as it was fitted on. Every synthetic row is assigned its nearest training row by Gower distance on
the generic encoding. Reported per cohort:

- reached: the share of training rows that are the nearest neighbour of at least one synthetic row;
- max_on_one: the most synthetic rows that share one nearest training row;
- top10_share: the share of synthetic rows whose nearest training row is among the 10% most-hit ones.

    python -m research.uci_coverage --sizes 50 250 1000 5000 --seeds 2 --workers 3
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
from research.size_attack import cohort  # noqa: E402
from research.size_sweep import ARMS  # noqa: E402
from research.size_sweep import load_context as _load_context  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
load_context = lru_cache(maxsize=1)(_load_context)
CHUNK = 256


def nearest_index(query: pd.DataFrame, target: pd.DataFrame, numeric, discrete, scale) -> np.ndarray:
    """Index of the nearest target row for every query row (Gower), in chunks so memory stays bounded."""
    tn = target[numeric].to_numpy(float) / scale
    td = target[discrete].astype(str).to_numpy()
    qn_all = query[numeric].to_numpy(float) / scale
    qd_all = query[discrete].astype(str).to_numpy()
    out = np.empty(len(query), dtype=int)
    for start in range(0, len(query), CHUNK):
        qn, qd = qn_all[start:start + CHUNK], qd_all[start:start + CHUNK]
        d = np.abs(qn[:, None, :] - tn[None, :, :]).sum(-1)
        for j in range(td.shape[1]):
            d += qd[:, j][:, None] != td[:, j][None, :]
        out[start:start + CHUNK] = d.argmin(axis=1)
    return out


def _quiet():
    warnings.filterwarnings("ignore")


def _run(arm, size, seed):
    _, codec, _, pool = load_context()
    rows = cohort(pool, size, seed)
    generator, params = ARMS[arm]
    fit_rng, sample_rng = (np.random.default_rng(s) for s in np.random.SeedSequence([seed, 7]).spawn(2))
    start = time.time()
    model = load_generator(generator)(**params).fit(codec.encode(rows), codec.spec, fit_rng)
    synth = codec.decode(model.sample(len(rows), sample_rng))

    numeric = [c for c, k in codec.spec.items() if k in (KIND_INTEGER, KIND_CONTINUOUS)]
    discrete = [c for c in codec.spec if c not in numeric]
    train_f, syn_f = codec.encode(rows), codec.encode(synth)
    scale = (train_f[numeric].max() - train_f[numeric].min()).replace(0, 1).to_numpy(float)
    nn = nearest_index(syn_f, train_f, numeric, discrete, scale)
    hits = np.bincount(nn, minlength=len(rows))
    top = np.sort(hits)[::-1][:max(1, len(rows) // 10)]
    return {"arm": arm, "size_patients": size, "seed": seed, "train_rows": len(rows),
            "reached": float((hits > 0).mean()), "max_on_one": int(hits.max()),
            "top10_share": float(top.sum() / len(nn)), "seconds": time.time() - start}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 250, 1000, 5000])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--arms", nargs="+", default=["independent_marginals", "gaussian_copula_shrink010",
                                                      "tvae", "tabular_diffusion"])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    tasks = [(a, n, s) for a in args.arms for n in args.sizes for s in range(args.seeds)]
    tasks.sort(key=lambda t: -t[1])
    print(f"{len(tasks)} fits on {args.workers} workers", flush=True)
    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = [pool.submit(_run, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            print(f"[{i:2d}/{len(tasks)}] {time.time() - start:6.0f}s {rows[-1]['arm']:26s} n={rows[-1]['size_patients']}", flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / "uci_coverage.csv", index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby(["arm", "size_patients"]).agg(train_rows=("train_rows", "mean"), reached=("reached", "mean"),
                                                        max_on_one=("max_on_one", "mean"),
                                                        top10_share=("top10_share", "mean"))
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n" + summary.to_string())
    summary.to_csv(OUT_DIR / "uci_coverage_summary.csv")
    print(f"\nwrote {OUT_DIR}/uci_coverage.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
