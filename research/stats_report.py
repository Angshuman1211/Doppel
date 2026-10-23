"""Confidence intervals over patients, and multiplicity-corrected comparisons.

Post-course extension work (improvement plan item 5; §VII of the paper, "seeds vary the generators, not
the patients, and p-values are uncorrected"). The reported benchmark gives mean ± sd over 20 generator
seeds on one fixed 80/20 patient-level split. That interval answers "how much does refitting move the
number", not "how much does the cohort move it", and the cohort is 80 patients for fidelity and 20 for
utility. With nine arms compared on several metrics it also runs a family of tests at an uncorrected
5%.

Two pieces here:

1. **Cluster bootstrap over patients.** Patients are resampled with replacement, admissions following
   their patient, so the resample respects the dependence the split procedure respects (ADR-001).
   Generators are held fixed: each arm and seed is fitted once on the real train split, and the bootstrap
   moves only the cohort the metric is measured against. Utility resamples the holdout the classifier is
   tested on, fidelity resamples the real train reference the synthetic table is compared with. The same
   resamples are reused for every arm and seed, so any arm-vs-arm difference is a paired bootstrap and
   gets its own interval rather than two overlapping marginal ones.

2. **Holm-Bonferroni** over a pre-specified family of comparisons, applied both to these paired bootstrap
   intervals and to the paired Wilcoxon tests across the 10 redrawn splits of `research/split_study.py`,
   recomputed here from its CSV rather than copied from its write-up.

The comparison family is fixed in `COMPARISONS` below and contains exactly the contrasts the report and
the split study already claim. It was not chosen after looking at these p-values, which is the only way a
Holm correction means anything.

    python -m research.stats_report --workers 6

Costs about 18 minutes on 6 workers: 180 generator fits, then 500 fidelity reports per fit.
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

from evaluation.arms import REPORTED_ARMS, resolve  # noqa: E402
from generators import schema as S  # noqa: E402
from generators.generate import synthesize  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
SEEDS = tuple(range(42, 62))          # the 20 seeds the reported benchmark uses
B_UTIL = 2000                         # holdout resamples: cheap, the classifier is not refitted
B_FID = 500                           # train-reference resamples: one fidelity report each, 23 ms
BOOT_SEED = 7
ALPHA = 0.05
# The report's "TSTR AUROC" column is the mean of these two, in this order.
CLASSIFIERS = ("logistic_regression", "random_forest")
METRICS = ("tstr", "corr_diff", "mean_js")
HIGHER_IS_BETTER = {"tstr": True, "corr_diff": False, "mean_js": False}

# Pre-specified family: every contrast the report or the split study already makes a claim about.
COMPARISONS = [
    ("tstr", "tabular_diffusion", "gaussian_copula_shrink010"),
    ("tstr", "tabular_diffusion_reg", "gaussian_copula_shrink010"),
    ("tstr", "tvae", "gaussian_copula_shrink010"),
    ("tstr", "gaussian_copula_shrink010", "gaussian_copula"),
    ("tstr", "gaussian_copula_shrink010", "independent_marginals"),
    ("tstr", "ctgan_small", "ctgan"),
    ("corr_diff", "gaussian_copula_shrink010", "gaussian_copula"),
    ("corr_diff", "gaussian_copula_shrink010", "independent_marginals"),
    ("corr_diff", "ctgan_small", "independent_marginals"),
    ("corr_diff", "tvae", "independent_marginals"),
    ("mean_js", "gaussian_copula_shrink010", "tabular_diffusion"),
    ("mean_js", "tabular_diffusion", "tvae"),
]


# --------------------------------------------------------------------------------------- bootstrap


def patient_resamples(df: pd.DataFrame, n_boot: int, seed: int) -> list[np.ndarray]:
    """Row indices for `n_boot` cluster bootstraps: patients drawn with replacement, admissions follow.

    Deterministic in `seed`, so every worker builds the identical set and comparisons across arms are
    paired on the same resample. Row counts vary between resamples, as they must when the sampling unit
    is a patient with between 1 and 15 admissions.
    """
    ids = df[S.ID_COLS[0]].to_numpy()
    subjects = np.sort(np.unique(ids))
    rows = [np.flatnonzero(ids == s) for s in subjects]
    picks = np.random.default_rng(seed).integers(0, len(subjects), size=(n_boot, len(subjects)))
    return [np.concatenate([rows[j] for j in pick]) for pick in picks]


def percentile_ci(values: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    lo, hi = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def reflect(values: np.ndarray, estimate: float) -> np.ndarray:
    """2*estimate - bootstrap values, which turns a percentile interval into a basic (pivotal) one.

    Needed because the fidelity statistics are two-sample divergences: the synthetic table was fitted on
    the original cohort, so its divergence from a *redrawn* cohort is systematically larger and the
    percentile interval can sit entirely above the reported number. The basic interval keeps the
    bootstrap's spread while staying centred on the statistic actually reported. For AUROC the two
    intervals agree, so this is applied everywhere rather than per metric.
    """
    return 2 * estimate - values


def bootstrap_p(values: np.ndarray) -> float:
    """Smallest level at which the interval of `values` excludes zero.

    The usual achieved-significance-level reading of a bootstrap interval; pass reflected values to get
    the level for the basic interval. Resolution is 2/B, so a p below that floor is reported at the
    floor rather than as zero.
    """
    v = values[~np.isnan(values)]
    tail = min((v <= 0).mean(), (v >= 0).mean())
    return float(min(1.0, max(2 * tail, 2 / len(v))))


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, float)
    n = len(p)
    adjusted, running = np.empty(n), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (n - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def holm_by(df: pd.DataFrame, family: str, column: str = "p") -> pd.Series:
    """Holm within each value of `family`, assigned by row label.

    By label and not by position: `groupby` hands back the groups as contiguous blocks, so writing a
    concatenation of per-group results straight into the frame silently permutes the p-values whenever
    the families are interleaved.
    """
    out = pd.Series(np.nan, index=df.index)
    for _, group in df.groupby(family, sort=False):
        out.loc[group.index] = holm(group[column].to_numpy())
    return out


# --------------------------------------------------------------------------------------- one fit


def _tstr_predictions(synth: pd.DataFrame, holdout: pd.DataFrame):
    """`tstr_eval`'s pipeline, stopping at the predictions so the bootstrap never refits a classifier."""
    from evaluation.utility_eval import CLASSIFIERS as MAKE, build_features

    X_train, y_train, encoder, imputer, scaler = build_features(synth, fit=True)
    X_test, y_test, *_ = build_features(holdout, encoder=encoder, num_imputer=imputer,
                                        num_scaler=scaler, fit=False)
    preds = [MAKE[name]().fit(X_train, y_train).predict_proba(X_test)[:, 1] for name in CLASSIFIERS]
    return np.column_stack(preds), y_test


def _quiet():
    warnings.filterwarnings("ignore")


def _run(arm, seed):
    from sklearn.metrics import roc_auc_score

    from evaluation.fidelity_metrics import run_fidelity_report

    real = S.load_real()
    train = real[real[S.SPLIT_COL] == S.TRAIN_SPLIT].reset_index(drop=True)
    holdout = real[real[S.SPLIT_COL] != S.TRAIN_SPLIT].reset_index(drop=True)

    generator, params = resolve(arm)
    start = time.time()
    _, _, synth = synthesize(train, generator, seed, params)
    # eval_runner round-trips generator output through a CSV read by S.load_real, which pins
    # icd9_primary as a string (P-005). Match it so these numbers sit on the benchmark's footing.
    tmp = OUT_DIR / "tmp" / f"stats_{arm}_{seed}.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    synth.to_csv(tmp, index=False)
    synth = S.load_real(tmp)
    tmp.unlink()

    preds, labels = _tstr_predictions(synth, holdout)
    full = run_fidelity_report(train, synth)
    point = {"tstr_" + name: float(roc_auc_score(labels, preds[:, j]))
             for j, name in enumerate(CLASSIFIERS)}
    point["tstr"] = float(np.mean([point["tstr_" + n] for n in CLASSIFIERS]))
    point["mean_js"] = full["mean_js_divergence"]
    point["corr_diff"] = full["correlation_diff"]

    boot_tstr = np.full((B_UTIL, len(CLASSIFIERS)), np.nan)
    for b, idx in enumerate(patient_resamples(holdout, B_UTIL, BOOT_SEED)):
        y = labels[idx]
        if y.min() == y.max():      # a resample that drew no deaths: AUROC undefined, not zero
            continue
        boot_tstr[b] = [roc_auc_score(y, preds[idx, j]) for j in range(len(CLASSIFIERS))]

    boot_fid = np.empty((B_FID, 2))
    for b, idx in enumerate(patient_resamples(train, B_FID, BOOT_SEED + 1)):
        report = run_fidelity_report(train.iloc[idx].reset_index(drop=True), synth)
        boot_fid[b] = [report["mean_js_divergence"], report["correlation_diff"]]

    return {"arm": arm, "seed": seed, "point": point, "seconds": time.time() - start,
            "boot_tstr": boot_tstr, "boot_fid": boot_fid}


# --------------------------------------------------------------------------------------- assembly


def assemble(results, arms):
    """Per-arm bootstrap distributions of the seed-averaged metric, shape (B,) for each metric."""
    curves, points = {}, {}
    for arm in arms:
        got = [r for r in results if r["arm"] == arm]
        tstr = np.stack([r["boot_tstr"] for r in got])          # (seeds, B, classifiers)
        fid = np.stack([r["boot_fid"] for r in got])            # (seeds, B, 2)
        with warnings.catch_warnings():                          # all-NaN slice: a resample with no deaths
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_tstr = np.nanmean(tstr, axis=(0, 2))
        curves[arm] = {"tstr": mean_tstr,
                       "mean_js": fid[:, :, 0].mean(axis=0),
                       "corr_diff": fid[:, :, 1].mean(axis=0)}
        points[arm] = pd.DataFrame([r["point"] for r in got])
    return curves, points


def point_table(curves, points, arms):
    rows = []
    for arm in arms:
        for metric in METRICS:
            values = points[arm][metric]
            estimate, boot = values.mean(), curves[arm][metric]
            basic_lo, basic_hi = percentile_ci(reflect(boot, estimate))
            cohort_lo, cohort_hi = percentile_ci(boot)
            rows.append({"arm": arm, "metric": metric, "estimate": estimate,
                         "seed_sd": values.std(ddof=1),
                         "seed_se": values.std(ddof=1) / np.sqrt(len(values)),
                         "patient_sd": float(np.nanstd(boot, ddof=1)),
                         # the reported statistic's own interval
                         "basic_lo": basic_lo, "basic_hi": basic_hi, "basic_width": basic_hi - basic_lo,
                         # the same metric measured against a redrawn cohort, to read against the floor
                         "cohort_mean": float(np.nanmean(boot)),
                         "cohort_lo": cohort_lo, "cohort_hi": cohort_hi})
    return pd.DataFrame(rows)


def comparison_table(curves, points):
    rows = []
    for metric, arm_a, arm_b in COMPARISONS:
        observed = points[arm_a][metric].mean() - points[arm_b][metric].mean()
        # Paired on the same resample, so the out-of-sample shift that moves each arm's divergence
        # affects both and largely cancels in the difference. `shift` records how much did not.
        diffs = reflect(curves[arm_a][metric] - curves[arm_b][metric], observed)
        lo, hi = percentile_ci(diffs)
        better = arm_a if (observed > 0) == HIGHER_IS_BETTER[metric] else arm_b
        rows.append({"metric": metric, "arm_a": arm_a, "arm_b": arm_b, "diff": observed,
                     "ci_lo": lo, "ci_hi": hi, "p": bootstrap_p(diffs),
                     "shift": observed - float(np.nanmean(diffs)),
                     "favours": better if lo * hi > 0 else "—"})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm_by(df, "metric")
    df["p_holm_pooled"] = holm(df["p"].to_numpy())
    df["survives"] = df.p_holm < ALPHA
    return df


def reference_noise(train: pd.DataFrame):
    """Divergence that patient resampling alone creates: a resampled cohort scored against the real one.

    The floor for reading the fidelity bootstrap. Any gap between an arm's in-sample divergence and its
    bootstrap mean has to be compared against this, because two draws of 80 patients differ from each
    other even with no generator involved.
    """
    from evaluation.fidelity_metrics import run_fidelity_report

    rows = [run_fidelity_report(train.iloc[idx].reset_index(drop=True), train)
            for idx in patient_resamples(train, B_FID, BOOT_SEED + 1)]
    return {"mean_js": np.array([r["mean_js_divergence"] for r in rows]),
            "corr_diff": np.array([r["correlation_diff"] for r in rows])}


def floor_table(points, noise, arms):
    """Each arm's fidelity against the floor a redrawn cohort sets, which is the real measurement limit.

    `p_floor` is the share of redrawn cohorts whose own divergence from the real one is at least as large
    as the arm's. A large value means the synthetic table is no further from the real cohort than another
    draw of the same patients would have been, so the metric cannot tell them apart at this cohort size.
    """
    rows = []
    for metric, values in noise.items():
        for arm in arms:
            estimate = points[arm][metric].mean()
            rows.append({"metric": metric, "arm": arm, "estimate": estimate,
                         "floor_mean": float(values.mean()),
                         "floor_lo": percentile_ci(values)[0], "floor_hi": percentile_ci(values)[1],
                         "p_floor": float((values >= estimate).mean()),
                         "at_floor": bool((values >= estimate).mean() > ALPHA / 2)})
    return pd.DataFrame(rows)


def save_cache(results, path: Path):
    columns = list(results[0]["point"])
    np.savez_compressed(
        path, arms=np.array([r["arm"] for r in results]),
        seeds=np.array([r["seed"] for r in results]), point_cols=np.array(columns),
        points=np.array([[r["point"][c] for c in columns] for r in results]),
        boot_tstr=np.stack([r["boot_tstr"] for r in results]),
        boot_fid=np.stack([r["boot_fid"] for r in results]))


def load_cache(path: Path):
    z = np.load(path, allow_pickle=False)
    columns = [str(c) for c in z["point_cols"]]
    return [{"arm": str(arm), "seed": int(seed), "boot_tstr": tstr, "boot_fid": fid,
             "point": dict(zip(columns, point))}
            for arm, seed, tstr, fid, point
            in zip(z["arms"], z["seeds"], z["boot_tstr"], z["boot_fid"], z["points"])]


def split_tests(path: Path):
    """Paired Wilcoxon across the 10 redrawn splits, with Holm over the same family."""
    from scipy.stats import wilcoxon

    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["tstr"] = df[["tstr_rf", "tstr_lr"]].mean(axis=1)
    per_split = df.groupby(["arm", "split_seed"])[["tstr", "tstr_rf", "mean_js", "corr_diff"]].mean()

    rows = []
    for metric, arm_a, arm_b in COMPARISONS:
        a, b = per_split.loc[arm_a], per_split.loc[arm_b]
        shared = a.index.intersection(b.index)
        for column, label in ([(metric, metric)] if metric != "tstr"
                              else [("tstr", "tstr"), ("tstr_rf", "tstr_rf")]):
            diff = (a.loc[shared, column] - b.loc[shared, column]).to_numpy()
            diff = diff[~np.isnan(diff)]          # a split where one arm emitted a one-class label
            stat = wilcoxon(diff) if len(diff) and np.any(diff != 0) else None
            rows.append({"metric": label, "arm_a": arm_a, "arm_b": arm_b, "n_splits": len(diff),
                         "diff": diff.mean(), "wins": int((diff > 0).sum()),
                         "p": 1.0 if stat is None else float(stat.pvalue)})
    out = pd.DataFrame(rows)
    # Same families as the bootstrap table: Holm within a metric. tstr_rf is the split study's own
    # utility column, kept so its published p-values stay checkable, and corrected separately.
    out["p_holm"] = holm_by(out, "metric")
    primary = out.index[out.metric != "tstr_rf"]
    out.loc[primary, "p_holm_pooled"] = holm(out.loc[primary, "p"].to_numpy())
    out["survives"] = out.p_holm < ALPHA
    return out


# --------------------------------------------------------------------------------------- driver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=REPORTED_ARMS)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--from-cache", action="store_true",
                        help="re-aggregate the last run's bootstrap arrays instead of refitting")
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    _quiet()      # the noise floor runs here, and ks_2samp warns on every tied resample

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / "stats_report_cache.npz"
    start = time.time()
    if args.from_cache:
        results = load_cache(cache)
        print(f"re-aggregating {len(results)} cached fits from {cache.name}", flush=True)
    else:
        tasks = [(arm, seed) for arm in args.arms for seed in SEEDS[:args.seeds]]
        cost = {"tabular_diffusion": 46, "tabular_diffusion_reg": 20,
                "ctgan": 27, "ctgan_small": 20, "tvae": 8}
        tasks.sort(key=lambda t: -cost.get(t[0], 1))
        print(f"{len(tasks)} fits on {args.workers} workers, "
              f"{B_UTIL} holdout and {B_FID} train resamples each", flush=True)

        results = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
            futures = [pool.submit(_run, *t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if i % 10 == 0 or i == len(tasks):
                    print(f"[{i:3d}/{len(tasks)}] {time.time() - start:6.0f}s", flush=True)
        save_cache(results, cache)

    curves, points = assemble(results, args.arms)
    degenerate = np.mean(np.isnan(results[0]["boot_tstr"][:, 0]))
    print(f"\n{degenerate:.2%} of holdout resamples drew no deaths and were dropped "
          f"(6 of the 20 holdout patients carry the only deaths)")

    real = S.load_real()
    noise = reference_noise(real[real[S.SPLIT_COL] == S.TRAIN_SPLIT].reset_index(drop=True))
    print("resampling-only floor, a redrawn cohort scored against the real one:")
    for metric, values in noise.items():
        lo, hi = percentile_ci(values)
        print(f"  {metric:10s} {values.mean():.4f}  95% [{lo:.4f}, {hi:.4f}]")
    pd.DataFrame(noise).to_csv(OUT_DIR / "stats_report_noise_floor.csv", index=False)

    table = point_table(curves, points, args.arms)
    comparisons = comparison_table(curves, points)
    floors = floor_table(points, noise, args.arms)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        for metric in METRICS:
            print(f"\n{metric}: estimate, sd over seeds, and 95% patient bootstrap interval")
            print(table[table.metric == metric].drop(columns="metric").to_string(index=False))
        print("\nfidelity against the floor a redrawn cohort sets")
        print(floors.to_string(index=False))
        print("\npaired patient bootstrap, Holm-corrected within each metric family")
        print(comparisons.to_string(index=False))

        splits = split_tests(OUT_DIR / "split_study.csv")
        if splits is not None:
            print(f"\npaired Wilcoxon across {splits.n_splits.max()} redrawn splits, Holm-corrected")
            print(splits.to_string(index=False))
            splits.to_csv(OUT_DIR / "stats_report_split_tests.csv", index=False)
        else:
            print("\nskipped the split tests: run research.split_study first")

    table.to_csv(OUT_DIR / "stats_report_points.csv", index=False)
    comparisons.to_csv(OUT_DIR / "stats_report_comparisons.csv", index=False)
    floors.to_csv(OUT_DIR / "stats_report_floors.csv", index=False)
    # Utility and fidelity have different resample counts, so the raw draws go to one file each.
    for name, metrics in (("utility", ("tstr",)), ("fidelity", ("corr_diff", "mean_js"))):
        pd.DataFrame({f"{arm}_{m}": curves[arm][m] for arm in args.arms for m in metrics}).to_csv(
            OUT_DIR / f"stats_report_bootstrap_{name}.csv", index=False)
    print(f"\nwrote {OUT_DIR}/stats_report_*.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
