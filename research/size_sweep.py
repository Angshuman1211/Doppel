"""Does memorization track cohort size? The size sweep (improvement plan item 1).

Post-course extension work. The project's central claim is that capacity relative to cohort size drives
memorization, inferred from a single cohort of 94 training rows. This measures it: the same generators,
the same metrics, over training sets from 50 to 5,000 patients drawn from UCI's Diabetes 130-US hospitals
data (101,766 encounters, 71,518 patients).

Three choices keep the comparison honest across sizes:

- the codec is fitted once on the full dataset, so the modeled schema is identical at every size. Fitting
  per subsample would shrink the frame at small n, which is the very thing being measured. The schema
  (levels, ranges) is therefore treated as public, the same assumption the DP study makes;
- the holdout is one fixed set of patients, disjoint from every training draw, so all sizes are scored
  against the same yardstick;
- every arm emits the same number of synthetic rows regardless of n, since the synthetic set size is a
  free parameter and letting it track n would confound the memorization measure.

    python -m research.size_sweep --sizes 50 100 250 500 1000 2500 5000 --seeds 2 --workers 6
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

from generators.codec import KIND_CONTINUOUS, KIND_INTEGER  # noqa: E402
from generators.diagnostics import _nearest_gower  # noqa: E402
from generators.generate import load_generator  # noqa: E402
from research.tabular_codec import TabularCodec  # noqa: E402

DATA = ROOT / "research" / "data" / "uci_diabetes.csv"
OUT_DIR = ROOT / "research" / "output"
ID_COLS = ["encounter_id", "patient_nbr"]
TARGET = "readmitted"
HOLDOUT_PATIENTS = 2000
N_SYNTH = 1500
MAX_LEVELS = 50

ARMS = {
    "independent_marginals": ("independent_marginals", {}),
    "gaussian_copula_shrink010": ("gaussian_copula", {"shrinkage": 0.10}),
    "ctgan_small": ("ctgan", {"epochs": 300, "embedding_dim": 32,
                              "generator_dim": [64, 64], "discriminator_dim": [64, 64]}),
    "tvae": ("tvae", {"epochs": 300}),
    "tabular_diffusion": ("tabular_diffusion", {}),
}

# Capacity grid for the journal version. The sweep above varies the cohort and holds the generator fixed,
# so it measures only half of "capacity relative to cohort size" — a reviewer's objection, and one
# size_sweep_result.md already concedes. `steps` is the training budget and `hidden` the network width,
# the two knobs that set how much the diffusion model can store. `diffusion_s3000_h256` repeats
# `tabular_diffusion`'s configuration, so the two must agree on a shared seed; that is the check that the
# grid is wired to the same generator.
CAPACITY_ARMS = {f"diffusion_s{steps}_h{hidden}": ("tabular_diffusion", {"steps": steps, "hidden": hidden})
                 for steps in (500, 1000, 3000) for hidden in (128, 256)}
# The five above stay the default arm list: the capacity grid is resolvable by name (size_attack.py
# looks arms up in ARMS too) but a plain `python -m research.size_sweep` must still fit the five the
# main sweep reports, not eleven.
SWEEP_ARMS = sorted(ARMS)
ARMS.update(CAPACITY_ARMS)


def load_context():
    df = pd.read_csv(DATA, low_memory=False)
    codec = TabularCodec(id_cols=ID_COLS, max_levels=MAX_LEVELS).fit(df)
    patients = np.sort(df["patient_nbr"].unique())
    rng = np.random.default_rng(0)
    holdout_patients = set(rng.choice(patients, size=HOLDOUT_PATIENTS, replace=False))
    is_holdout = df["patient_nbr"].isin(holdout_patients)
    return df, codec, df[is_holdout].reset_index(drop=True), df[~is_holdout].reset_index(drop=True)


def gower_memorization(codec, fit_df, ref_df, synth_df):
    """DCR ratio and near-copy rate, as in generators/diagnostics.py but on the generic frame."""
    numeric = [c for c, k in codec.spec.items() if k in (KIND_INTEGER, KIND_CONTINUOUS)]
    discrete = [c for c in codec.spec if c not in numeric]
    fit_f, ref_f, syn_f = (codec.encode(d) for d in (fit_df, ref_df, synth_df))
    scale = (fit_f[numeric].max() - fit_f[numeric].min()).replace(0, 1).to_numpy(float)
    synth_d = _nearest_gower(syn_f, fit_f, numeric, discrete, scale)
    ref_d = _nearest_gower(ref_f, fit_f, numeric, discrete, scale)
    return {"dcr_ratio": float(np.median(synth_d) / np.median(ref_d)),
            "near_copy_rate": float(np.mean(synth_d < np.percentile(ref_d, 5)))}


def utility(train_like, holdout_df, codec):
    """Train-on-synthetic, test-on-real AUROC for early readmission (<30 days)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    def xy(df):
        frame = codec.encode(df)
        y = (df[TARGET].astype(str) == "<30").astype(int).to_numpy()
        x = pd.get_dummies(frame.drop(columns=[TARGET], errors="ignore"), drop_first=False)
        return x, y

    x_tr, y_tr = xy(train_like)
    x_te, y_te = xy(holdout_df)
    x_te = x_te.reindex(columns=x_tr.columns, fill_value=0)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return np.nan
    model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1).fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, model.predict_proba(x_te)[:, 1]))


def _quiet():
    warnings.filterwarnings("ignore")


def _run(arm, size, seed):
    from evaluation.fidelity_metrics import js_divergence_categorical, js_divergence_continuous

    df, codec, holdout, pool = load_context()
    rng = np.random.default_rng(10_000 + seed)
    patients = np.sort(pool["patient_nbr"].unique())
    chosen = set(rng.choice(patients, size=min(size, len(patients)), replace=False))
    train = pool[pool["patient_nbr"].isin(chosen)].reset_index(drop=True)

    generator, params = ARMS[arm]
    frame = codec.encode(train)
    fit_rng, sample_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))

    start = time.time()
    model = load_generator(generator)(**params).fit(frame, codec.spec, fit_rng)
    synth = codec.decode(model.sample(N_SYNTH, sample_rng))
    seconds = time.time() - start

    numeric = [c for c, k in codec.spec.items() if k in (KIND_INTEGER, KIND_CONTINUOUS)]
    js = [js_divergence_continuous(train[c].dropna(), synth[c].dropna()) for c in numeric]
    js += [js_divergence_categorical(train[c], synth[c]) for c in codec.spec if c not in numeric]
    corr = (train[numeric].corr().to_numpy(), synth[numeric].corr().to_numpy())
    upper = np.triu(np.ones_like(corr[0], dtype=bool), k=1)

    row = {"arm": arm, "size_patients": size, "seed": seed, "train_rows": len(train),
           "mean_js": float(np.nanmean(js)),
           "corr_diff": float(np.mean(np.abs(np.nan_to_num(corr[0][upper]) - np.nan_to_num(corr[1][upper])))),
           "tstr_rf": utility(synth, holdout, codec), "seconds": seconds}
    row.update(gower_memorization(codec, train, holdout.sample(min(len(holdout), 1500), random_state=1), synth))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 100, 250, 500, 1000, 2500, 5000])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--arms", nargs="+", default=SWEEP_ARMS,
                        help=f"default is the main sweep's {len(SWEEP_ARMS)} arms; "
                             f"the capacity grid ({', '.join(sorted(CAPACITY_ARMS))}) is opt-in")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    # Named like size_attack.py's, because a fixed output path means a run with a different arm list
    # silently replaces the main sweep's results rather than sitting beside them.
    parser.add_argument("--out", default="size_sweep", help="stem for the CSVs under research/output")
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    tasks = [(a, n, s) for a in args.arms for n in args.sizes for s in range(args.seeds)]
    tasks.sort(key=lambda t: -(t[1] * (3 if t[0] == "ctgan_small" else 1)))
    print(f"{len(tasks)} fits on {args.workers} workers", flush=True)

    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = [pool.submit(_run, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            print(f"[{i:3d}/{len(tasks)}] {time.time() - start:6.0f}s  "
                  f"{rows[-1]['arm']:26s} n={rows[-1]['size_patients']:<5d} "
                  f"near_copy={rows[-1]['near_copy_rate']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / f"{args.out}.csv", index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby(["arm", "size_patients"]).agg(
        train_rows=("train_rows", "mean"), near_copy=("near_copy_rate", "mean"),
        dcr_ratio=("dcr_ratio", "mean"), mean_js=("mean_js", "mean"),
        corr_diff=("corr_diff", "mean"), tstr_rf=("tstr_rf", "mean"), seconds=("seconds", "mean"))
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print("\n" + summary.to_string())
    summary.to_csv(OUT_DIR / f"{args.out}_summary.csv")
    print(f"\nwrote {OUT_DIR}/{args.out}.csv and _summary.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
