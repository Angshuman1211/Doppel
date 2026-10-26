"""Does our diffusion reimplementation behave like the official TabDDPM? (journal version, task 3)

Post-course extension work. `generators/diffusion.py` is our own implementation of a TabDDPM-style
model, and the paper's conclusions about the diffusion family all rest on it. A reviewer is entitled to
ask whether it behaves like the published model or like something else wearing its name. This fits both
on the same cohort and scores them with the same metrics.

The official model is used directly, not reimplemented again: `tab_ddpm.GaussianMultinomialDiffusion`
and `tab_ddpm.modules.MLPDiffusion` from yandex-research/tab-ddpm, pinned to one commit. Only their model
package is imported, which depends on nothing but numpy and torch; their training pipeline, their data
loaders and their CatBoost evaluation are not involved, because what is being compared is the generative
model and not their experimental harness.

What is matched, so the comparison is about the implementation and not the configuration: cosine beta
schedule, 1,000 diffusion timesteps, noise (epsilon) parametrization with MSE loss on the numeric block,
x0 parametrization on the categorical block, a 2-layer MLP of width 256, Adam at lr 1e-3 with weight
decay 1e-5, batch size 64, and 3,000 training steps.

What cannot be matched, and is the point of the exercise:

- **the multinomial loss.** Ours uses cross-entropy to the clean class; the official default is the full
  stochastic variational bound (`multinomial_loss_type='vb_stochastic'`). Our docstring claims this is
  what most TabDDPM implementations do in place of the KL. This measures what it costs.
- **the numeric transform.** Ours maps each column through its own fitted marginal to a normal; theirs
  uses a quantile transform, which is what their pipeline applies.
- **where the time embedding enters.** Ours concatenates it to the input; theirs projects the input to
  `dim_t` and adds the embedding there.

    python -m research.tabddpm_check --size 1000 --seeds 3
"""

import argparse
import contextlib
import io
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.codec import KIND_CONTINUOUS, KIND_INTEGER  # noqa: E402
from generators.generate import load_generator  # noqa: E402
from research.size_sweep import gower_memorization, load_context, utility  # noqa: E402

OUT_DIR = ROOT / "research" / "output"
TABDDPM_DIR = ROOT / "research" / "third_party" / "tab-ddpm"
TABDDPM_REPO = "https://github.com/yandex-research/tab-ddpm.git"
TABDDPM_COMMIT = "b476257dd460b778ba09eb97f7a51d6490fa17f8"      # MIT, Kotelnikov et al. 2023

N_SYNTH = 1500          # the size sweep's release size, so near-copy rates are comparable to it
# Matched across both implementations. `dim_t` is the official default; ours uses its own 64, noted above.
MATCHED = {"steps": 3000, "timesteps": 1000, "hidden": 256, "layers": 2,
           "lr": 1e-3, "weight_decay": 1e-5, "batch_size": 64, "dropout": 0.0}
# The matched setting is our generator's own default, and its batch of 64 is far smaller than anything in
# the authors' configs (1024 to 4096) with far fewer steps. If the official model is unstable there, that
# is a statement about the setting and not about the model, so this second preset moves toward the
# authors' own regime and is the control for exactly that.
AUTHORS = MATCHED | {"steps": 20000, "batch_size": 256}
CONFIGS = {"matched": MATCHED, "authors": AUTHORS}
DIM_T = 128


def ensure_tabddpm(path: Path = TABDDPM_DIR) -> Path:
    """Clone the official repo at the pinned commit if it is not already there."""
    if not (path / "tab_ddpm").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"cloning {TABDDPM_REPO} at {TABDDPM_COMMIT[:12]} into {path}", flush=True)
        subprocess.run(["git", "clone", "--quiet", TABDDPM_REPO, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--quiet", TABDDPM_COMMIT], check=True)
    return path


def official_modules(path: Path = TABDDPM_DIR):
    sys.path.insert(0, str(ensure_tabddpm(path)))
    from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
    from tab_ddpm.modules import MLPDiffusion
    from tab_ddpm.utils import FoundNANsError
    return GaussianMultinomialDiffusion, MLPDiffusion, FoundNANsError


def _columns(spec):
    numeric = [c for c, k in spec.items() if k in (KIND_INTEGER, KIND_CONTINUOUS)]
    return numeric, [c for c in spec if c not in numeric]


def official_synthesize(frame: pd.DataFrame, spec: dict, n_rows: int, seed: int, config: dict) -> pd.DataFrame:
    """Fit the official TabDDPM on an encoded frame and return a synthetic frame in the same layout."""
    import torch
    from sklearn.preprocessing import QuantileTransformer

    GaussianMultinomialDiffusion, MLPDiffusion, FoundNANsError = official_modules()
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    numeric, discrete = _columns(spec)
    # Their model takes numerics in a transformed space and categoricals as ordinal codes over levels
    # taken from the training frame, which is the same domain our codec already fixed.
    transformer = QuantileTransformer(output_distribution="normal", random_state=seed,
                                      n_quantiles=max(10, min(1000, len(frame) // 2)))
    x_num = transformer.fit_transform(frame[numeric].to_numpy(float)) if numeric else np.zeros((len(frame), 0))
    levels = {c: sorted(frame[c].astype(str).unique()) for c in discrete}
    codes = {c: {v: i for i, v in enumerate(levels[c])} for c in discrete}
    x_cat = np.column_stack([frame[c].astype(str).map(codes[c]).to_numpy() for c in discrete]) \
        if discrete else np.zeros((len(frame), 0))
    num_classes = np.array([len(levels[c]) for c in discrete], dtype=int)

    x = torch.tensor(np.hstack([x_num, x_cat]), dtype=torch.float32)
    d_in = len(numeric) + int(num_classes.sum())
    denoise_fn = MLPDiffusion(d_in=d_in, num_classes=0, is_y_cond=False, dim_t=DIM_T,
                              rtdl_params={"d_layers": [config["hidden"]] * config["layers"],
                                           "dropout": config["dropout"]})
    model = GaussianMultinomialDiffusion(
        num_classes=num_classes, num_numerical_features=len(numeric), denoise_fn=denoise_fn,
        num_timesteps=config["timesteps"], gaussian_loss_type="mse", gaussian_parametrization="eps",
        parametrization="x0", scheduler="cosine", device=torch.device("cpu"))
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    dummy_y = torch.zeros(config["batch_size"], dtype=torch.long)
    for _ in range(config["steps"]):
        idx = torch.randint(0, len(x), (config["batch_size"],), generator=generator)
        optimizer.zero_grad()
        loss_multi, loss_gauss = model.mixed_loss(x[idx], {"y": dummy_y})
        (loss_multi + loss_gauss).backward()
        optimizer.step()

    model.eval()
    with contextlib.redirect_stdout(io.StringIO()):     # their sampler prints a timestep counter
        sample, _ = model.sample_all(n_rows, n_rows, y_dist=torch.ones(1), ddim=False)
    sample = sample.numpy()

    out = {}
    if numeric:
        back = transformer.inverse_transform(sample[:, :len(numeric)])
        out.update({c: back[:, j] for j, c in enumerate(numeric)})
    for j, c in enumerate(discrete):
        k = np.clip(np.rint(sample[:, len(numeric) + j]).astype(int), 0, len(levels[c]) - 1)
        out[c] = np.asarray(levels[c], dtype=object)[k]
    return pd.DataFrame(out)[list(spec)]


def score(codec, train, holdout, synth) -> dict:
    """The size sweep's metrics, so these numbers sit beside size_sweep_result.md."""
    from evaluation.fidelity_metrics import js_divergence_categorical, js_divergence_continuous

    numeric, discrete = _columns(codec.spec)
    js = [js_divergence_continuous(train[c].dropna(), synth[c].dropna()) for c in numeric]
    js += [js_divergence_categorical(train[c], synth[c]) for c in discrete]
    corr = (train[numeric].corr().to_numpy(), synth[numeric].corr().to_numpy())
    upper = np.triu(np.ones_like(corr[0], dtype=bool), k=1)
    row = {"mean_js": float(np.nanmean(js)),
           "corr_diff": float(np.mean(np.abs(np.nan_to_num(corr[0][upper]) - np.nan_to_num(corr[1][upper])))),
           "tstr_rf": utility(synth, holdout, codec)}
    row.update(gower_memorization(codec, train, holdout.sample(min(len(holdout), 1500), random_state=1), synth))
    return row


def run_one(implementation: str, size: int, seed: int, config_name: str = "matched") -> dict:
    config = CONFIGS[config_name]
    df, codec, holdout, pool = load_context()
    rng = np.random.default_rng(10_000 + seed)      # the size sweep's cohort draw
    patients = np.sort(pool["patient_nbr"].unique())
    chosen = set(rng.choice(patients, size=min(size, len(patients)), replace=False))
    train = pool[pool["patient_nbr"].isin(chosen)].reset_index(drop=True)
    frame = codec.encode(train)

    start = time.time()
    failure = ""
    try:
        if implementation == "official":
            synth_frame = official_synthesize(frame, codec.spec, N_SYNTH, seed, config)
        else:
            fit_rng, sample_rng = (np.random.default_rng(s)
                                   for s in np.random.SeedSequence(seed).spawn(2))
            model = load_generator("tabular_diffusion")(**config).fit(frame, codec.spec, fit_rng)
            synth_frame = model.sample(N_SYNTH, sample_rng)
        synth = codec.decode(synth_frame)
    except BaseException as exc:        # their FoundNANsError derives from BaseException, not Exception
        return {"implementation": implementation, "config": config_name, "size_patients": size,
                "seed": seed, "train_rows": len(train), "seconds": time.time() - start,
                "failure": f"{type(exc).__name__}: {exc}"[:200]}

    row = {"implementation": implementation, "config": config_name, "size_patients": size,
           "seed": seed, "train_rows": len(train), "seconds": time.time() - start, "failure": failure}
    row.update(score(codec, train, holdout, synth))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=1000, help="cohort size in patients")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--implementations", nargs="+", default=["ours", "official"])
    parser.add_argument("--config", default="matched", choices=sorted(CONFIGS))
    parser.add_argument("--out", default="tabddpm_check")
    args = parser.parse_args(argv)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    warnings.filterwarnings("ignore")

    ensure_tabddpm()
    rows, start = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Serial: each fit already pins itself to one thread and the official sampler holds the whole
    # release in memory for 1,000 timesteps.
    for implementation in args.implementations:
        for seed in range(args.seeds):
            rows.append(run_one(implementation, args.size, seed, args.config))
            last = rows[-1]
            note = last["failure"] or f"near_copy={last['near_copy_rate']:.3f}"
            print(f"[{len(rows):2d}] {implementation:9s} seed {seed} {last['seconds']:6.0f}s  {note}",
                  flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / f"{args.out}.csv", index=False)

    df = pd.DataFrame(rows)
    summary = df.groupby(["implementation", "config"]).agg(
        n=("seed", "count"), failures=("failure", lambda s: int((s.astype(str) != "").sum())),
        near_copy=("near_copy_rate", "mean"), dcr_ratio=("dcr_ratio", "mean"),
        mean_js=("mean_js", "mean"), corr_diff=("corr_diff", "mean"),
        tstr_rf=("tstr_rf", "mean"), seconds=("seconds", "mean"))
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print("\n" + summary.to_string())
    summary.to_csv(OUT_DIR / f"{args.out}_summary.csv")
    print(f"\nwrote {OUT_DIR}/{args.out}.csv ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
