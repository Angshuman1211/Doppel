"""Two AIM diagnostics that `dp_study.py` cannot produce, each backing a number in dp_aim_result.md.

Both exist because a number in a result document has to regenerate from the repository, and neither of
these came from the study runner: `dp_study` cannot vary a generator's budget shares, and the clique
degrees are only visible in AIM's stdout.

1. **Budget-share ablation.** AIM takes its budget from the histograms over the non-core columns, which
   confounds any fidelity comparison against `dp_marginals` until it is measured. This refits at two
   splits and reports mean JSD, which is the `0.250 vs 0.223` pair in the result document.
2. **Selected clique degrees.** AIM prints the clique it measures each round. This captures that output
   and counts one-way against two-way, which is the `260 / 122` pair.

    python -m research.aim_diagnostics --seeds 10
"""

import argparse
import contextlib
import io
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators import schema as S  # noqa: E402
from generators.generate import synthesize  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
SHARE_VARIANTS = {"default": {"aim": 0.40, "marginals": 0.30, "codes": 0.30},
                  "aim_starved": {"aim": 0.15, "marginals": 0.55, "codes": 0.30}}
CLIQUE = re.compile(r"Measuring Clique \((.*?)\)")


def train_split():
    real = S.load_real()
    return real[real[S.SPLIT_COL] == S.TRAIN_SPLIT].reset_index(drop=True)


def fit_capturing(train, seed, params):
    """Synthesize once, returning the synthetic frame and whatever AIM printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _, _, synth = synthesize(train, "dp_aim", seed, params, n_rows=len(train))
    return synth, buffer.getvalue()


def share_ablation(train, epsilon, seeds):
    from evaluation.fidelity_metrics import run_fidelity_report

    rows = []
    for label, shares in SHARE_VARIANTS.items():
        for seed in range(42, 42 + seeds):
            synth, _ = fit_capturing(train, seed, {"epsilon": epsilon, "shares": shares})
            rows.append({"variant": label, "aim_share": shares["aim"], "seed": seed,
                         "mean_js": run_fidelity_report(train, synth)["mean_js_divergence"]})
    return pd.DataFrame(rows)


def clique_degrees(train, epsilon, seeds):
    rows = []
    for seed in range(42, 42 + seeds):
        _, printed = fit_capturing(train, seed, {"epsilon": epsilon})
        degrees = Counter(len([p for p in match.split(",") if p.strip()])
                          for match in CLIQUE.findall(printed))
        rows.append({"epsilon": epsilon, "seed": seed,
                     "one_way": degrees.get(1, 0), "two_way": degrees.get(2, 0)})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1.0, 8.0])
    parser.add_argument("--ablation-epsilon", type=float, default=8.0)
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    train, start = train_split(), time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ablation = share_ablation(train, args.ablation_epsilon, args.seeds)
    ablation.to_csv(OUT_DIR / "aim_share_ablation.csv", index=False)
    print(f"budget-share ablation at epsilon = {args.ablation_epsilon}, {args.seeds} seeds")
    for label, g in ablation.groupby("variant", sort=False):
        print(f"  {label:12s} aim share {g.aim_share.iloc[0]:.2f}: "
              f"mean_js = {g.mean_js.mean():.4f} +/- {g.mean_js.std():.4f}")

    cliques = pd.concat([clique_degrees(train, e, args.seeds) for e in args.epsilons],
                        ignore_index=True)
    cliques.to_csv(OUT_DIR / "aim_clique_degrees.csv", index=False)
    print(f"\nclique degrees AIM selected, {args.seeds} seeds per budget")
    for epsilon, g in cliques.groupby("epsilon"):
        one, two = int(g.one_way.sum()), int(g.two_way.sum())
        print(f"  epsilon = {epsilon}: {one} one-way, {two} two-way "
              f"({two / max(one + two, 1):.0%} pairwise)")
    total = cliques[["one_way", "two_way"]].sum()
    print(f"  pooled: {int(total.one_way)} one-way, {int(total.two_way)} two-way")
    print(f"\nwrote {OUT_DIR}/aim_share_ablation.csv and aim_clique_degrees.csv "
          f"({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
