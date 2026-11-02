"""Corrected resampled tests over the redrawn patient splits (Nadeau and Bengio, 2003).

Post-course extension work, in answer to a review comment. The redrawn splits are redraws of the same 100
patients, so their training sets overlap and a paired test that treats them as independent is too liberal. The
corrected resampled t-test replaces the variance of the mean, s^2/J, with (1/J + n_test/n_train) s^2.

The correction was derived for test error, so it applies directly to the utility and, as a conservative
check, to fidelity. A membership attack has no separate test set: for the attack comparisons the same
inflation (n_test/n_train = 0.25, the split design) is used as a sensitivity check and not as a formal test.
Each family is Holm-corrected as in the Wilcoxon analyses. Also writes the 95% intervals with the same variance.

    python -m research.nb_correction
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.split_attack import holm  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
T, L, I, D, R_, V, C, G = ("gaussian_copula_shrink010", "gaussian_copula", "independent_marginals", "tabular_diffusion",
                           "tabular_diffusion_reg", "tvae", "ctgan_small", "ctgan")
UTILITY = [(D, T), (R_, T), (V, T), (T, I), (T, L), (C, G)]
FIDELITY = {"corr_diff": [(T, L), (T, I), (C, I), (V, I)], "mean_js": [(T, D), (D, V)]}
ATTACK_ARMS = [D, V, R_, T, I]


def corrected(v, ratio):
    v = v.dropna()
    J = len(v)
    se = np.sqrt((1 / J + ratio) * v.var(ddof=1))
    t = v.mean() / se if se > 0 else np.inf
    return J, v.mean(), 2 * stats.t.sf(abs(t), J - 1), se


def family(wide, pairs, ratio, name):
    rows = []
    for a, b in pairs:
        J, diff, p, se = corrected(wide[a] - wide[b], ratio)
        half = stats.t.ppf(0.975, J - 1) * se
        rows.append({"family": name, "a": a, "b": b, "n_splits": J, "diff": diff, "a_wins": int(((wide[a] - wide[b]) > 0).sum()),
                     "p_corrected": p, "lo": diff - half, "hi": diff + half})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_corrected.to_numpy())
    return out


def main():
    d = pd.read_csv(OUT_DIR / "split_study.csv")
    ratio = float((d.n_holdout / d.n_train).mean())
    d["tstr_mean"] = d[["tstr_lr", "tstr_rf"]].mean(axis=1)
    blocks = []
    for metric, label in [("tstr_mean", "utility_mean"), ("tstr_rf", "utility_rf")]:
        blocks.append(family(d.groupby(["split_seed", "arm"])[metric].mean().unstack(), UTILITY, ratio, label))
    for metric, pairs in FIDELITY.items():
        blocks.append(family(d.groupby(["split_seed", "arm"])[metric].mean().unstack(), pairs, ratio, metric))
    s = pd.read_csv(OUT_DIR / "learned_attack_splits.csv")
    for metric in ["dist_worst", "gbm", "lira", "dist_worst_tpr1", "gbm_tpr1", "lira_tpr1"]:
        blocks.append(family(s.pivot(index="split_seed", columns="arm", values=metric),
                             list(itertools.combinations(ATTACK_ARMS, 2)), 0.25, f"attack_{metric}"))
    g, w = s.pivot(index="split_seed", columns="arm", values="gbm"), s.pivot(index="split_seed", columns="arm", values="dist_worst")
    l_ = s.pivot(index="split_seed", columns="arm", values="lira")
    gains = pd.concat({f"{a}__gbm": g[a] - w[a] for a in ATTACK_ARMS} | {f"{a}__lira": l_[a] - w[a] for a in ATTACK_ARMS}, axis=1)
    rows = [{"family": "learned_minus_distance", "a": c, "b": "dist_worst", "n_splits": len(gains), "diff": gains[c].mean(),
             "a_wins": int((gains[c] > 0).sum()), "p_corrected": corrected(gains[c], 0.25)[2]} for c in gains]
    gain = pd.DataFrame(rows)
    gain["p_holm"] = holm(gain.p_corrected.to_numpy())
    blocks.append(gain)
    out = pd.concat(blocks, ignore_index=True)
    out.to_csv(OUT_DIR / "nb_correction.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.float_format", "{:.4f}".format):
        print(f"utility and fidelity n_test/n_train = {ratio:.3f}")
        print(out[["family", "a", "b", "diff", "a_wins", "p_corrected", "p_holm", "lo", "hi"]].to_string(index=False))
    # per-setting means with corrected intervals, for the headline numbers
    iv = []
    for (arm, metric), x in s.melt(id_vars="arm", value_vars=["dist_worst", "gbm", "lira", "dist_worst_tpr1", "gbm_tpr1", "lira_tpr1"], var_name="metric").groupby(["arm", "metric"])["value"]:
        J, mean, _, se = corrected(x, 0.25)
        half = stats.t.ppf(0.975, J - 1) * se
        iv.append({"arm": arm, "metric": metric, "mean": mean, "lo": mean - half, "hi": mean + half})
    pd.DataFrame(iv).to_csv(OUT_DIR / "nb_correction_intervals.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
