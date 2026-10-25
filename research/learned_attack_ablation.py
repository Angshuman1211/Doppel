"""Where the learned attack's gain comes from, on the features saved by `research/learned_attack.py`.

Refits only the attack models, never a generator. For each saved (setting, seed):
  gbm_no_typical   the 15 synthetic-distance features, without each record's distance to its nearest real record
  gbm_<distance>   the 6 features of one distance only
  gbm_8_shadows    the full model, trained and scored on the first 8 shadows only (the benchmark's count)
  dist_8_shadows   the benchmark's worst-case distance attack on those 8 shadows

    python -m research.learned_attack_ablation
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.learned_attack import ARMS, DISTANCES, FEATURE_DIR, OUT_DIR, leave_one_shadow_out  # noqa: E402


def gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          early_stopping=False, random_state=0)


def ablate(X, member):
    from sklearn.linear_model import LogisticRegression

    row = {}
    synth_cols = [c for c in range(18) if c % 6 != 5]
    row["gbm_no_typical"], row["gbm_no_typical_tpr1"] = leave_one_shadow_out(X[..., synth_cols], member, gbm)
    for i, name in enumerate(DISTANCES):
        row[f"gbm_{name}"], row[f"gbm_{name}_tpr1"] = leave_one_shadow_out(X[..., 6 * i:6 * i + 6], member, gbm)
    row["gbm_8_shadows"], row["gbm_8_shadows_tpr1"] = leave_one_shadow_out(X[:8], member[:8], gbm)
    row["dist_8_shadows"] = max(leave_one_shadow_out(X[:8, :, [6 * i]], member[:8], LogisticRegression)[0]
                                for i in range(3))
    return row


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for path in sorted(p for p in FEATURE_DIR.glob("*.npz") if not p.name.startswith("split")):
        arm, seed = path.stem.rsplit("_", 1)
        z = np.load(path)
        rows.append({"arm": arm, "seed": int(seed), **ablate(z["X"], z["member"])})
        print(arm, seed, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "learned_attack_ablation.csv", index=False)
    cols = ["gbm_no_typical", "gbm_numeric4", "gbm_gower", "gbm_codes", "gbm_8_shadows", "dist_8_shadows"]
    order = [a for a in ARMS if a in set(df.arm)]
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print("\n" + df.groupby("arm")[cols].mean().reindex(order).to_string())
        print("\n" + df.groupby("arm")[[c + "_tpr1" for c in cols[:5]]].mean().reindex(order).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
