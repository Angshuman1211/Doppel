"""The membership attack across redrawn patient splits on MIMIC-III.

Post-course extension work, in answer to a review comment. The privacy comparisons in the paper are Welch
tests over 20 generator seeds on one patient split. Seeds resample the generator and the shadow subsets, not
the patients, so those p-values describe generator variability only. This reruns the benchmark's attack on
redrawn patient-level splits, so the key privacy orderings can be tested over patients as the utility
orderings already are.

Each split is drawn as in `research/split_study.py` (80% of patients to training, admissions follow their
patient, split seeds 1000 + k). On each split and for each setting, the benchmark's shadow attack is run
unchanged on that split's training admissions: 8 shadow generators on random halves, the three scored
distances on the same shadows, and the worst case of the three. One generator seed per split.

    python -m research.split_attack --splits 10 --workers 4
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
from research.split_study import patient_split  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
ARMS = ["independent_marginals", "gaussian_copula_shrink010", "tvae", "tabular_diffusion", "tabular_diffusion_reg"]
N_SHADOW = 8
SEED = 42
COMPARISONS = [("tabular_diffusion", "tabular_diffusion_reg"), ("tabular_diffusion_reg", "tvae"),
               ("tvae", "gaussian_copula_shrink010"), ("gaussian_copula_shrink010", "independent_marginals")]


def _quiet():
    warnings.filterwarnings("ignore")


def _run(arm, split_seed):
    from evaluation.eval_runner import cached, real_generator_fn
    from evaluation.membership_inference import (NUMERIC_COLS, codes_distances, gower_distances,
                                                 run_membership_inference)

    train, _ = patient_split(S.load_real(), split_seed)
    start = time.time()
    generator_fn = cached(real_generator_fn(arm, disk_cache=False))
    attacks = {"numeric4": None, "gower": gower_distances, "codes": codes_distances}
    row = {"arm": arm, "split_seed": split_seed, "n_train": len(train)}
    for name, fn in attacks.items():
        r = run_membership_inference(train, generator_fn, NUMERIC_COLS, n_shadow=N_SHADOW, seed=SEED, distance_fn=fn)
        row[f"mia_{name}"] = r["mean_attack_auroc"]
        row[f"tpr1_{name}"] = r["tpr_at_fpr_1pct"]
    row["mia_worst"] = max(row[f"mia_{n}"] for n in attacks)
    row["seconds"] = time.time() - start
    return row


def holm(p):
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def tests(df):
    from scipy.stats import wilcoxon

    wide = df.pivot(index="split_seed", columns="arm", values="mia_worst")
    rows = []
    for a, b in COMPARISONS:
        d = (wide[a] - wide[b]).dropna().to_numpy()
        rows.append({"comparison": f"{a} - {b}", "n_splits": len(d), "diff": d.mean(),
                     "wins": int((d > 0).sum()), "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    tasks = [(a, 1000 + k) for a in args.arms for k in range(args.splits)]
    cost = {"tabular_diffusion": 10, "tabular_diffusion_reg": 4, "tvae": 2}
    tasks.sort(key=lambda t: -cost.get(t[0], 1))
    print(f"{len(tasks)} attack runs ({len(tasks) * N_SHADOW} shadow fits) on {args.workers} workers", flush=True)
    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = [pool.submit(_run, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            print(f"[{i:2d}/{len(tasks)}] {time.time() - start:6.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / "split_attack.csv", index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby("arm")[["mia_worst", "mia_gower", "mia_codes", "tpr1_gower"]].agg(["mean", "std"])
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n" + summary.to_string())
        t = tests(df)
        print("\npaired Wilcoxon over splits on worst-case AUROC, Holm-corrected\n" + t.to_string(index=False))
    t.to_csv(OUT_DIR / "split_attack_tests.csv", index=False)
    print(f"\nwrote {OUT_DIR}/split_attack.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
