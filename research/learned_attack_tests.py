"""Paired tests over redrawn patient splits for the learned and distance attacks.

Reads `research/output/learned_attack_splits.csv` (from `research.learned_attack --splits 10`). For each attack
(worst-case distance, gradient boosting, likelihood ratio) and each of AUROC and TPR at 1% FPR, every pair of the
five settings is compared with a two-sided paired Wilcoxon signed-rank test over the splits, and the ten pairs
are Holm-corrected as one family per attack and metric.

    python -m research.learned_attack_tests
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.split_attack import holm  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
SETTINGS = ["tabular_diffusion", "tvae", "tabular_diffusion_reg", "gaussian_copula_shrink010", "independent_marginals"]
METRICS = ["dist_worst", "gbm", "lira", "dist_worst_tpr1", "gbm_tpr1", "lira_tpr1"]


def main():
    from scipy.stats import wilcoxon

    df = pd.read_csv(OUT_DIR / "learned_attack_splits.csv")
    rows = []
    for metric in METRICS:
        wide = df.pivot(index="split_seed", columns="arm", values=metric)
        block = []
        for a, b in itertools.combinations(SETTINGS, 2):
            d = (wide[a] - wide[b]).dropna().to_numpy()
            block.append({"metric": metric, "a": a, "b": b, "n_splits": len(d), "diff": d.mean(),
                          "a_wins": int((d > 0).sum()), "p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0})
        block = pd.DataFrame(block)
        block["p_holm"] = holm(block.p.to_numpy())
        rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_DIR / "learned_attack_split_tests.csv", index=False)

    summary = df.groupby("arm")[METRICS].agg(["mean", "std"])
    with pd.option_context("display.width", 240, "display.float_format", "{:.3f}".format):
        print(summary.to_string())
        print("\n" + out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
