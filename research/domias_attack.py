"""A density-based membership attack (DOMIAS) against the released synthetic table.

Post-course extension work (improvement plan item 3, and limitation 3 of the membership survey). Every
attack in the benchmark scores a record by its distance to the nearest synthetic row. DOMIAS
(van Breugel et al., 2023) instead scores the density ratio

    score(x) = p_synth(x) / p_ref(x)

so a record only looks like a member if the release puts more mass on it than the population does. That
corrects for records sitting in dense regions, which is the same confound this project hit with the
holdout, and it is the attack the paper's limitations name but do not run.

Threat model, deliberately weaker than the benchmark's shadow attack:

- the attacker sees the released synthetic table and a reference sample from the population, nothing else.
  No shadow generators, no retraining, no access to the generator;
- the generator is fitted on 47 of the 94 training admissions, so the other 47 are non-members drawn from
  the same distribution. That removes the holdout distribution shift that inflates a direct attack;
- the reference sample is the clean holdout (subject 41976 removed), which is disjoint from both;
- the column schema is public, as in the size-sweep and DP studies.

The distance attack is run on the same split, so the two are directly comparable.

    python -m research.domias_attack --seeds 10 --workers 6
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

from evaluation.arms import resolve  # noqa: E402
from generators import schema as S  # noqa: E402
from generators.codec import KIND_CONTINUOUS, KIND_INTEGER, FrameCodec  # noqa: E402
from generators.generate import synthesize  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
ARMS = ["independent_marginals", "gaussian_copula_shrink010", "ctgan_small",
        "tvae", "tabular_diffusion", "tabular_diffusion_reg"]
N_SYNTH = 500
N_COMPONENTS = 5
# With 47 non-members the smallest non-zero false-positive rate is 1/47, about 2%, so 1% is not
# measurable here and 5% is the lowest honest threshold (membership survey, limitation 1).
FPR_LEVELS = (0.05, 0.10)


def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(scores, labels, fpr_level):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(fpr_level, fpr, tpr))


def density_scores(candidates, synth, reference, codec, seed):
    """log p_synth(x) - log p_ref(x) on a PCA projection fitted only on attacker-visible data."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity
    from sklearn.preprocessing import StandardScaler

    numeric = [c for c, k in codec.spec.items() if k in (KIND_INTEGER, KIND_CONTINUOUS)]
    frames = [codec.encode(d)[numeric].to_numpy(float) for d in (candidates, synth, reference)]
    cand_x, synth_x, ref_x = frames

    visible = np.vstack([synth_x, ref_x])          # the attacker never sees the candidates' labels
    scaler = StandardScaler().fit(visible)
    pca = PCA(n_components=min(N_COMPONENTS, visible.shape[1]), random_state=seed).fit(scaler.transform(visible))

    def project(x):
        return pca.transform(scaler.transform(x))

    def fit_kde(x):
        # Scott's rule, the usual default when no bandwidth is tuned.
        n, d = x.shape
        bandwidth = max(n ** (-1.0 / (d + 4)) * x.std(axis=0).mean(), 1e-3)
        return KernelDensity(bandwidth=bandwidth).fit(x)

    cand_p = project(cand_x)
    return fit_kde(project(synth_x)).score_samples(cand_p) - fit_kde(project(ref_x)).score_samples(cand_p)


def distance_scores(candidates, synth, codec):
    """Baseline: minus the Gower distance to the nearest synthetic row, on the same split."""
    from generators.diagnostics import _nearest_gower

    numeric = [c for c, k in codec.spec.items() if k in (KIND_INTEGER, KIND_CONTINUOUS)]
    discrete = [c for c in codec.spec if c not in numeric]
    cand_f, synth_f = codec.encode(candidates), codec.encode(synth)
    scale = (cand_f[numeric].max() - cand_f[numeric].min()).replace(0, 1).to_numpy(float)
    return -_nearest_gower(cand_f, synth_f, numeric, discrete, scale)


def _quiet():
    warnings.filterwarnings("ignore")


def _run(arm, seed):
    real = S.load_real()
    train = real[real[S.SPLIT_COL] == S.TRAIN_SPLIT].reset_index(drop=True)
    reference = S.clean_holdout(real)

    rng = np.random.default_rng(seed)
    members_idx = rng.choice(len(train), size=len(train) // 2, replace=False)
    is_member = np.zeros(len(train), dtype=int)
    is_member[members_idx] = 1
    members = train.iloc[members_idx].reset_index(drop=True)

    generator, params = resolve(arm)
    start = time.time()
    _, _, synth = synthesize(members, generator, seed, params, n_rows=N_SYNTH)
    codec = FrameCodec(min_count=5).fit(train)

    row = {"arm": arm, "seed": seed, "n_members": len(members), "n_nonmembers": int((is_member == 0).sum()),
           "seconds": time.time() - start}
    for name, scores in [("domias", density_scores(train, synth, reference, codec, seed)),
                         ("distance", distance_scores(train, synth, codec))]:
        row[f"{name}_auroc"] = auroc(scores, is_member)
        for level in FPR_LEVELS:
            row[f"{name}_tpr{int(level * 100)}"] = tpr_at_fpr(scores, is_member, level)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    tasks = [(a, 42 + s) for a in args.arms for s in range(args.seeds)]
    print(f"{len(tasks)} runs on {args.workers} workers", flush=True)
    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        futures = [pool.submit(_run, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 10 == 0 or i == len(tasks):
                print(f"[{i:3d}/{len(tasks)}] {time.time() - start:5.0f}s", flush=True)

    df = pd.DataFrame(rows).sort_values(["arm", "seed"], ignore_index=True)
    df.to_csv(OUT_DIR / "domias_attack.csv", index=False)
    summary = df.groupby("arm").agg(
        domias_auroc=("domias_auroc", "mean"), domias_sd=("domias_auroc", "std"),
        domias_tpr5=("domias_tpr5", "mean"),
        distance_auroc=("distance_auroc", "mean"), distance_sd=("distance_auroc", "std"),
        distance_tpr5=("distance_tpr5", "mean"), n=("seed", "count"))
    summary["domias_minus_distance"] = summary.domias_auroc - summary.distance_auroc
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print("\n" + summary.to_string())
    summary.to_csv(OUT_DIR / "domias_attack_summary.csv")
    print(f"\nwrote {OUT_DIR}/domias_attack.csv and _summary.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
