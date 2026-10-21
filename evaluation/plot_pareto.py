"""
Doppel - Track 2: static fidelity/utility/privacy figure for the written report.

The web dashboard renders this interactively; a Word/PDF report needs a plain image instead. This reuses
evaluation.pareto's own per-seed loader, so the figure's numbers are exactly the ones in full_pareto_result.md
and full_evaluation_result.md, not retyped from the markdown tables.

    python -m evaluation.plot_pareto [--out docs/figures/pareto_frontier.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.pareto import METRICS, load

# Same palette, shapes and family grouping as the web dashboard (web/src/lib/marks.jsx, web/public/data/dashboard.json),
# so the static report figure and the interactive one read as one visual language.
ARMS = [
    # id, label, color, marker, direct label (dx, dy, ha[, leader line]) or None. Six arms sit within 0.03 of each other on the
    # privacy axis, so only the ones that stand clear are labeled on the plot; the legend names every arm.
    ("independent_marginals", "Independent marginals", "#898781", "s", (0.004, -0.045, "left")),
    ("gaussian_copula", "Gaussian copula", "#2a78d6", "o", None),
    ("gaussian_copula_shrink025", "Gaussian copula, shrink 0.25", "#2a78d6", "D", None),
    ("gaussian_copula_shrink010", "Gaussian copula, shrink 0.10", "#2a78d6", "h", (-0.075, 0.14, "center", True)),
    ("ctgan", "CTGAN", "#eb6834", "^", None),
    ("ctgan_small", "CTGAN, small networks", "#eb6834", "P", None),
    ("tvae", "TVAE", "#1baf7a", "v", (0.008, -0.03, "left")),
    ("tabular_diffusion", "Tabular diffusion", "#8e3185", "o", (0.0, 0.11, "center")),
    ("tabular_diffusion_reg", "Tabular diffusion, regularized", "#8e3185", "D", (0.0, 0.10, "center")),
]
FAIL_LINE = 0.65


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/pareto_frontier.png")
    args = ap.parse_args()

    by_arm = load()
    missing = [arm for arm, *_ in ARMS if arm not in by_arm]
    if missing:
        raise SystemExit(f"No results for {missing}. Run evaluation.eval_runner for every seed first.")

    i_util, i_mia = METRICS.index("utility_tstr"), METRICS.index("mia_worst")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=200)
    ax.set_facecolor("#fafaf8")
    fig.patch.set_facecolor("white")

    ax.axvline(FAIL_LINE, color="#c0392b", linestyle=":", linewidth=1, zorder=1)
    ax.text(FAIL_LINE + 0.002, 0.305, "fail line 0.65", color="#c0392b", fontsize=9, va="bottom")
    ax.axvline(0.5, color="#999", linestyle=":", linewidth=1, zorder=1)
    ax.text(0.5 + 0.002, 0.305, "guessing 0.5", color="#777", fontsize=9, va="bottom")

    for arm, label, color, marker, lab in ARMS:
        rows = by_arm[arm]
        util, mia = rows[:, i_util], rows[:, i_mia]
        mx, my = mia.mean(), util.mean()
        sx, sy = mia.std(ddof=1), util.std(ddof=1)
        ax.errorbar(mx, my, xerr=sx, yerr=sy, color=color, elinewidth=1.4, capsize=3, zorder=2, linestyle="none")
        ax.scatter([mx], [my], s=110, color=color, marker=marker, edgecolor="white", linewidth=1.3, zorder=3, label=label)
        if lab:
            dx, dy, ha = lab[:3]
            leader = dict(arrowstyle="-", color="#999", linewidth=0.8, shrinkB=6) if len(lab) > 3 else None
            ax.annotate(label, (mx, my), xytext=(mx + dx, my + dy), fontsize=9.5, color="#222", ha=ha, va="center",
                        arrowprops=leader)

    ax.set_xlabel("Privacy leakage: worst-case membership attack AUROC  (lower is better)")
    ax.set_ylabel("Utility: mean TSTR AUROC  (higher is better)")
    ax.set_title("Utility against privacy leakage, nine arms × 20 seeds\nwhiskers are ±1 sd over seeds", fontsize=12, loc="left")
    ax.set_xlim(0.46, 0.83)
    ax.set_ylim(0.28, 0.95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e3e1db", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, bbox_to_anchor=(0.99, 0.01), ncol=1)
    fig.text(0.01, 0.01, "Project Doppel · evaluation.pareto · n = 20 seeds per arm, mean ± 1 sd", fontsize=7.5, color="#888")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
