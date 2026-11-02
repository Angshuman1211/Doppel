"""Cohort size at a fixed number of training passes per record, on UCI Diabetes 130-US.

Post-course extension work, in answer to a review comment. The diffusion model trains for a fixed 3,000 steps
at batch 64, so a smaller cohort also means more passes over each record, and the fall in attack score with
cohort size mixes the two. This holds passes per fitted record fixed and varies the cohort.

Passes are counted on the rows a shadow generator is fitted on (half the cohort), so 3,000 steps on 33 member
rows is 5,818 passes. Steps follow each draw's member count, so that cell is close to the size attack's
configuration and not identical to it.
Same cohorts, member draws and fit seeds as `research/size_attack.py`, width 256, 8 shadows, two cohort draws;
the distance attack and the two learned attacks of `research/size_learned_attack.py` score each cell.

Holding 5,818 passes at 5,000 patients would need about 325,000 steps per fit, so that cell is not run.

    python -m research.passes_study --workers 4
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

from generators.generate import load_generator  # noqa: E402
from research.size_learned_attack import _encoded, _setup, _typical, evaluate, knn_gower  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
CACHE = OUT_DIR / "passes_features"
N_SHADOW = 8
BATCH = 64
GRID = {100: [50, 250, 1000, 5000], 1000: [50, 250, 1000, 5000], 5818: [50, 250, 1000]}


def _quiet():
    warnings.filterwarnings("ignore")


def _shadow(passes, size, seed, shadow_id):
    path = CACHE / f"p{passes}_{size}_{seed}_{shadow_id}.npz"
    if path.exists():
        return passes, size, seed, shadow_id
    codec, _, rows, numeric, discrete, scale, levels = _setup(size, seed)
    rng = np.random.default_rng([seed, shadow_id, size])    # as in size_attack.py
    members = np.zeros(len(rows), dtype=int)
    members[rng.choice(len(rows), size=len(rows) // 2, replace=False)] = 1
    member_df = rows[members == 1].reset_index(drop=True)
    steps = max(1, round(passes * len(member_df) / BATCH))
    fit_rng, sample_rng = (np.random.default_rng(s) for s in np.random.SeedSequence([seed, shadow_id]).spawn(2))
    start = time.time()
    model = load_generator("tabular_diffusion")(steps=steps, hidden=256).fit(codec.encode(member_df), codec.spec, fit_rng)
    synth = codec.decode(model.sample(len(member_df), sample_rng))
    cand = _encoded(codec, rows, numeric, discrete, scale, levels)
    syn = _encoded(codec, synth, numeric, discrete, scale, levels)
    knn = knn_gower(cand, syn, len(numeric) + len(discrete))
    np.savez_compressed(path, knn=knn, member=members, steps=steps, seconds=time.time() - start)
    return passes, size, seed, shadow_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="passes_study")
    args = parser.parse_args(argv)
    # ctgan's DataTransformer is not used here, but keep every worker to one core
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        os.environ[var] = "1"
    CACHE.mkdir(parents=True, exist_ok=True)

    tasks = [(p, n, s, k) for p, sizes in GRID.items() for n in sizes for s in range(args.seeds) for k in range(N_SHADOW)]
    tasks.sort(key=lambda t: t[0] * t[1])       # cheapest first, so partial results exist early
    print(f"{len(tasks)} shadow fits on {args.workers} workers", flush=True)
    start = time.time()
    typical = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        tf = [pool.submit(_typical, n, s) for n in {n for v in GRID.values() for n in v} for s in range(args.seeds)]
        for fut in as_completed(tf):
            r = fut.result()
            typical[r[1:3]] = r[3]
        futures = [pool.submit(_shadow, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 8 == 0 or i == len(tasks):
                print(f"[{i:3d}/{len(tasks)}] {time.time() - start:6.0f}s", flush=True)

    rows = []
    for passes, sizes in GRID.items():
        for size in sizes:
            for seed in range(args.seeds):
                z = [np.load(CACHE / f"p{passes}_{size}_{seed}_{k}.npz") for k in range(N_SHADOW)]
                knn, member = np.stack([x["knn"] for x in z]), np.stack([x["member"] for x in z])
                rows.append({"passes": passes, "size_patients": size, "seed": seed, "train_rows": knn.shape[1],
                             "steps": int(np.mean([x["steps"] for x in z])),
                             "fit_seconds": float(np.mean([x["seconds"] for x in z])),
                             **evaluate(knn, typical[(size, seed)], member)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"{args.out}.csv", index=False)
    cols = ["steps", "dist", "gbm", "lira", "dist_tpr1", "gbm_tpr1", "lira_tpr1"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n" + df.groupby(["passes", "size_patients"])[cols].mean().to_string())
    print(f"\nwrote {OUT_DIR}/{args.out}.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
