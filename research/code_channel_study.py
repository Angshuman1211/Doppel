"""How much of the membership signal is the shared decoder's rare-code channel, on MIMIC-III.

Post-course extension work, in answer to a review comment. `generators/codec.py` draws secondary codes seen in
fewer than five training admissions from a pool of the training rows' own codes, and the primary code is a
categorical column over training levels, so a member's rare code can reappear in a release and a non-member's
cannot. This refits the non-private settings with one change at decoding: rare secondary and rare primary
codes are drawn uniformly from the public ICD-9 dictionary, as the DP baselines do. Codes seen in at least five
training admissions stay, since they are modelled columns.

Same seeds, member draws and generator seeds as `research/learned_attack.py`, so its results are the paired
baseline. Also records the mean divergence of a full fit, which is what the public vocabulary costs.

    python -m research.code_channel_study --workers 4
"""

import argparse
import json
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

from generators import schema as S  # noqa: E402
from generators.codec import FrameCodec  # noqa: E402
from generators.generate import load_generator  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
ARMS = ["independent_marginals", "gaussian_copula_shrink010", "tvae", "tabular_diffusion", "tabular_diffusion_reg"]


class PublicRareCodec(FrameCodec):
    """FrameCodec with rare codes decoded from the public dictionary instead of the training rows."""

    def fit(self, train):
        from generators.dp_copula import _public_dictionary

        super().fit(train)
        primary = train[S.PRIMARY_COL].astype(str).value_counts()
        self.rare_primary = set(primary[primary < self.min_count].index)
        public = [str(c) for c in _public_dictionary()]
        self.public_codes = np.array(public, dtype=object)
        keep = set(self.vocab)
        self.tail_codes = np.array([c for c in public if c not in keep], dtype=object)
        self.tail_weights = np.ones(len(self.tail_codes), dtype=float)
        return self

    def decode(self, frame, rng):
        out = super().decode(frame, rng)
        for r in np.flatnonzero(out[S.PRIMARY_COL].astype(str).isin(self.rare_primary).to_numpy()):
            new = str(rng.choice(self.public_codes))
            codes = json.loads(out[S.CODES_COL].iloc[r])
            codes[0] = new
            out.iloc[r, out.columns.get_loc(S.CODES_COL)] = json.dumps(codes)
            out.iloc[r, out.columns.get_loc(S.PRIMARY_COL)] = new
        return out


def public_synthesize(train, arm, seed, n_rows):
    from evaluation.arms import resolve

    generator, params = resolve(arm)
    fit_rng, sample_rng, decode_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    codec = PublicRareCodec(min_count=5).fit(train)
    model = load_generator(generator)(**(params or {})).fit(codec.encode(train), codec.spec, fit_rng)
    return codec.decode(model.sample(n_rows, sample_rng), decode_rng)


def _quiet():
    warnings.filterwarnings("ignore")


def _run(arm, seed):
    from evaluation.fidelity_metrics import run_fidelity_report
    from research.learned_attack import attack_row, shadow_features

    start = time.time()

    def generate(member_df, numeric_cols, n_samples, gen_seed):
        return public_synthesize(member_df.reset_index(drop=True), arm, gen_seed, n_samples)

    X, logd, member = shadow_features(arm, seed, generate=generate)
    real = S.load_real()
    train = real[real[S.SPLIT_COL] == S.TRAIN_SPLIT].reset_index(drop=True)
    tmp = OUT_DIR / "tmp" / f"code_channel_{arm}_{seed}.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    public_synthesize(train, arm, seed, len(train)).to_csv(tmp, index=False)
    fidelity = run_fidelity_report(train, S.load_real(tmp))
    tmp.unlink()
    return {"arm": arm, "seed": seed, "mean_js": fidelity["mean_js_divergence"], "corr_diff": fidelity["correlation_diff"],
            **attack_row(X, logd, member), "seconds": time.time() - start}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="code_channel_study")
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        os.environ[var] = "1"

    tasks = [(a, 42 + s) for a in args.arms for s in range(args.seeds)]
    print(f"{len(tasks)} runs on {args.workers} workers", flush=True)
    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_quiet) as pool:
        for i, fut in enumerate(as_completed([pool.submit(_run, *t) for t in tasks]), 1):
            rows.append(fut.result())
            print(f"[{i:2d}/{len(tasks)}] {rows[-1]['arm']} {rows[-1]['seed']} {time.time() - start:6.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / f"{args.out}.csv", index=False)
    df = pd.DataFrame(rows)
    cols = ["mean_js", "dist_codes", "dist_gower", "dist_worst", "gbm", "lira", "dist_worst_tpr1", "gbm_tpr1", "lira_tpr1"]
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print("\n" + df.groupby("arm")[cols].mean().to_string())
    print(f"\nwrote {OUT_DIR}/{args.out}.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
