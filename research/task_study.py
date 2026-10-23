"""Does the utility ranking depend on which outcome is predicted? (improvement plan item 6)

Post-course extension work. Every utility number in the project comes from one prediction task,
in-hospital mortality on MIMIC, so "generator A is more useful than generator B" is really "A beats B at
one task on one holdout". Item 6 asks for a second task. Two findings sit behind this script:

- **MIMIC cannot supply one.** The only other binary outcome in the schema is `readmit_30d`, and the
  35-row holdout contains 2 positives, both admissions of the same patient; the clean holdout (P-003)
  contains none. `--mimic` prints that census and the share of patient resamples in which the task has no
  positive case at all, which is the quantitative form of "not measurable".
- So the second task is run where it is measurable, on the UCI Diabetes 130-US data the size sweep
  already uses, with the same codec, the same fixed holdout patients and the same arms, so the numbers
  sit beside [`size_sweep_result.md`](size_sweep_result.md) rather than starting a new comparison.

Four tasks spanning base rates from 11% to 53%, each one a column the generator models and which is
removed from the features for its own task. A TRTR ceiling is fitted per task on the same real training
subsample, because a task nobody can learn from real data cannot rank generators either.

    python -m research.task_study --mimic
    python -m research.task_study --sizes 250 1000 --seeds 3 --workers 6
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
from research.size_sweep import ARMS, load_context  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
N_SYNTH = 1500
HOLDOUT_EVAL = 4000          # holdout rows scored per task, capped so 30 fits stay affordable

# (source column withheld from the features, label rule). Comparisons go through str so a decoded
# synthetic frame, where a categorical code is a string, matches the real frame's integer dtype.
TASKS = {
    "readmit_lt30": ("readmitted", lambda d: d["readmitted"].astype(str).str.strip() == "<30"),
    "long_stay": ("time_in_hospital", lambda d: pd.to_numeric(d["time_in_hospital"],
                                                              errors="coerce") >= 7),
    "high_meds": ("num_medications", lambda d: pd.to_numeric(d["num_medications"],
                                                             errors="coerce") >= 20),
    "emergency_admission": ("admission_type_id",
                            lambda d: d["admission_type_id"].astype(str).str.strip().str.replace(
                                r"\.0$", "", regex=True) == "1"),
}


def features(df: pd.DataFrame, codec, source: str) -> pd.DataFrame:
    """Modeled columns minus the task's own source column, one-hot encoded."""
    frame = codec.encode(df).drop(columns=[source, f"miss_{source}"], errors="ignore")
    return pd.get_dummies(frame, drop_first=False)


def auroc(train_like: pd.DataFrame, test_df: pd.DataFrame, codec, task: str) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    source, label = TASKS[task]
    x_tr, x_te = features(train_like, codec, source), features(test_df, codec, source)
    y_tr = label(train_like).astype(int).to_numpy()
    y_te = label(test_df).astype(int).to_numpy()
    x_te = x_te.reindex(columns=x_tr.columns, fill_value=0)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return np.nan
    model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1).fit(x_tr, y_tr)
    return float(roc_auc_score(y_te, model.predict_proba(x_te)[:, 1]))


def _quiet():
    warnings.filterwarnings("ignore")


def _run(arm, size, seed):
    df, codec, holdout, pool = load_context()
    rng = np.random.default_rng(10_000 + seed)       # the size sweep's draw, so the cohorts match
    patients = np.sort(pool["patient_nbr"].unique())
    chosen = set(rng.choice(patients, size=min(size, len(patients)), replace=False))
    train = pool[pool["patient_nbr"].isin(chosen)].reset_index(drop=True)
    test = holdout.sample(min(len(holdout), HOLDOUT_EVAL), random_state=1).reset_index(drop=True)

    generator, params = ARMS[arm]
    fit_rng, sample_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    start = time.time()
    model = load_generator(generator)(**params).fit(codec.encode(train), codec.spec, fit_rng)
    synth = codec.decode(model.sample(N_SYNTH, sample_rng))
    seconds = time.time() - start

    row = {"arm": arm, "size_patients": size, "seed": seed, "train_rows": len(train),
           "seconds": seconds}
    for task in TASKS:
        row[f"tstr_{task}"] = auroc(synth, test, codec, task)
        row[f"trtr_{task}"] = auroc(train, test, codec, task)
    return row


# ------------------------------------------------------------------------------- the MIMIC census


def mimic_census(n_boot=2000, seed=11):
    """Why the second task cannot be run on MIMIC: the holdout has no usable positives."""
    from generators import schema as S

    from research.stats_report import patient_resamples

    real = S.load_real()
    train = real[real[S.SPLIT_COL] == S.TRAIN_SPLIT]
    holdout = real[real[S.SPLIT_COL] != S.TRAIN_SPLIT].reset_index(drop=True)
    clean = S.clean_holdout(real)

    print("MIMIC-III demo, binary outcomes in the modeled schema")
    print(f"{'cohort':16s} {'rows':>5s} {'mortality':>10s} {'readmit_30d':>12s} {'patients w/ readmit':>20s}")
    for name, frame in (("train", train), ("holdout", holdout), ("clean holdout", clean)):
        carriers = int((frame.groupby("subject_id").readmit_30d.sum() > 0).sum())
        print(f"{name:16s} {len(frame):5d} {int(frame.hospital_expire_flag.sum()):10d} "
              f"{int(frame.readmit_30d.sum()):12d} {carriers:20d}")

    print("\nshare of patient resamples of the holdout with no positive case, so no AUROC:")
    for target in ("hospital_expire_flag", "readmit_30d"):
        y = holdout[target].to_numpy()
        empty = np.mean([len(np.unique(y[idx])) < 2
                         for idx in patient_resamples(holdout, n_boot, seed)])
        print(f"  {target:22s} {empty:.1%}")


# ------------------------------------------------------------------------------------- the sweep


def summarize(df: pd.DataFrame):
    from scipy.stats import spearmanr

    tasks = list(TASKS)
    for size, group in df.groupby("size_patients"):
        ranks = {}
        print(f"\nn = {size} patients ({int(group.train_rows.mean())} rows), "
              f"mean over {group.seed.nunique()} seeds")
        header = "".join(f"{t:>22s}" for t in tasks)
        print(f"{'arm':28s}{header}")
        for arm, rows in group.groupby("arm"):
            values = [rows[f"tstr_{task}"].mean() for task in tasks]
            ranks[arm] = values
            print(f"{arm:28s}" + "".join(f"{v:22.3f}" for v in values))
        print(f"{'TRTR ceiling (real data)':28s}"
              + "".join(f"{group[f'trtr_{t}'].mean():22.3f}" for t in tasks))

        table = pd.DataFrame(ranks, index=tasks).T.dropna()
        if len(table) > 2:
            print("  Spearman correlation between the arm orderings of two tasks:")
            for i, a in enumerate(tasks):
                for b in tasks[i + 1:]:
                    if a in table and b in table:
                        rho = spearmanr(table[a], table[b]).statistic
                        print(f"    {a:22s} vs {b:22s} rho = {rho:+.2f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mimic", action="store_true", help="print the MIMIC census and stop")
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 1000])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--arms", nargs="+", default=sorted(ARMS))
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    _quiet()

    if args.mimic:
        mimic_census()
        return 0

    tasks = [(a, n, s) for a in args.arms for n in args.sizes for s in range(args.seeds)]
    tasks.sort(key=lambda t: -(t[1] * (3 if t[0] == "ctgan_small" else 1)))
    print(f"{len(tasks)} fits on {args.workers} workers, {len(TASKS)} tasks each", flush=True)

    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = [pool.submit(_run, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            print(f"[{i:3d}/{len(tasks)}] {time.time() - start:6.0f}s  {rows[-1]['arm']:26s} "
                  f"n={rows[-1]['size_patients']:<5d}", flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / "task_study.csv", index=False)

    df = pd.DataFrame(rows).sort_values(["size_patients", "arm", "seed"], ignore_index=True)
    df.to_csv(OUT_DIR / "task_study.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        summarize(df)
    print(f"\nwrote {OUT_DIR}/task_study.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
