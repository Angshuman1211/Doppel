"""Differentially private baseline: noisy marginals, optionally joined by a noisy Gaussian copula.

Post-course extension work (improvement plan item 4): the benchmark has no arm with a formal guarantee,
so every privacy statement in it is attack-relative. This adds one, built on the same shared codec as the
other generators so the comparison stays like for like.

Mechanism, under bounded differential privacy (one row replaced, so a count moves by at most 2):

- Each column's histogram gets Laplace noise, then is clipped at zero and renormalized. Categorical and
  binary columns use their levels; numeric columns use fixed-width bins over the schema's declared range.
- `DPGaussianCopula` spends part of the budget on the latent correlation matrix: the data is mapped to
  latent normals, clipped to +-CLIP, and the covariance gets Laplace noise with sensitivity
  2 * CLIP^2 / n per entry, after which the matrix is projected back to the nearest positive-definite
  correlation matrix.
- Budget composes sequentially: the per-column budget is `epsilon * marginal_share / n_columns`, and the
  correlation step takes the rest. With 110 modeled columns that per-column budget is small by
  construction, which is the point: it shows what a formal guarantee costs at n = 94.

The column *ranges and levels* are treated as public, taken from the schema and feature dictionary rather
than measured under the budget. That is the usual practical assumption for tabular DP and it is stated
here because it is an assumption, not a theorem.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from .base import Generator
from .codec import KIND_BINARY, KIND_CATEGORICAL, KIND_CONTINUOUS, KIND_INTEGER
from .copula import EPS, ledoit_wolf_shrinkage

CLIP = 3.0          # latent normals are clipped here before the covariance is measured
N_BINS = 20         # bins per numeric column
SENSITIVITY = 2.0   # bounded DP: replacing one row moves a histogram by at most 2 in L1


class _DPDiscrete:
    """Noisy level frequencies for a categorical or binary column."""

    def __init__(self, values, levels, epsilon, rng):
        counts = pd.Series(values).value_counts().reindex(levels, fill_value=0).to_numpy(float)
        noisy = counts + rng.laplace(0.0, SENSITIVITY / epsilon, size=len(counts))
        noisy = np.clip(noisy, 0.0, None)
        if noisy.sum() <= 0:
            noisy = np.ones_like(noisy)
        self.levels = np.asarray(levels, dtype=object)
        self.p = noisy / noisy.sum()
        self.upper = np.cumsum(self.p)
        self.lower = self.upper - self.p

    def to_normal(self, values, rng):
        index = {v: k for k, v in enumerate(self.levels)}
        k = np.array([index[v] for v in values])
        return norm.ppf(np.clip(rng.uniform(self.lower[k], self.upper[k]), EPS, 1 - EPS))

    def from_normal(self, z):
        k = np.searchsorted(self.upper, norm.cdf(z), side="right")
        return self.levels[np.minimum(k, len(self.levels) - 1)]

    def sample(self, n, rng):
        return self.levels[rng.choice(len(self.levels), size=n, p=self.p)]


class _DPNumeric:
    """Noisy histogram over a declared range; samples land uniformly inside the chosen bin."""

    def __init__(self, values, lo, hi, epsilon, rng, integer=False):
        self.edges = np.linspace(lo, hi if hi > lo else lo + 1.0, N_BINS + 1)
        counts, _ = np.histogram(np.clip(values, self.edges[0], self.edges[-1]), bins=self.edges)
        noisy = np.clip(counts + rng.laplace(0.0, SENSITIVITY / epsilon, size=N_BINS), 0.0, None)
        if noisy.sum() <= 0:
            noisy = np.ones_like(noisy)
        self.p = noisy / noisy.sum()
        self.upper = np.cumsum(self.p)
        self.lower = self.upper - self.p
        self.integer = integer

    def _from_u(self, u, rng):
        k = np.minimum(np.searchsorted(self.upper, u, side="right"), N_BINS - 1)
        within = (u - self.lower[k]) / np.maximum(self.p[k], EPS)
        x = self.edges[k] + within * (self.edges[k + 1] - self.edges[k])
        return np.rint(x) if self.integer else x

    def to_normal(self, values, rng):
        x = np.clip(values, self.edges[0], self.edges[-1])
        k = np.minimum(np.searchsorted(self.edges, x, side="right") - 1, N_BINS - 1)
        k = np.maximum(k, 0)
        u = rng.uniform(self.lower[k], self.upper[k])
        return norm.ppf(np.clip(u, EPS, 1 - EPS))

    def from_normal(self, z, rng):
        return self._from_u(np.clip(norm.cdf(z), EPS, 1 - EPS), rng)

    def sample(self, n, rng):
        return self._from_u(rng.uniform(size=n), rng)


def _nearest_correlation(cov: np.ndarray) -> np.ndarray:
    """Project a noisy symmetric matrix to the nearest positive-definite correlation matrix."""
    sym = (cov + cov.T) / 2
    vals, vecs = np.linalg.eigh(sym)
    psd = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    corr = psd / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


class DPMarginals(Generator):
    """Every column sampled from its own noisy histogram, no dependence between columns."""

    name = "dp_marginals"
    marginal_share = 1.0

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = float(epsilon)

    def _fit_marginals(self, frame, spec, rng):
        self.columns = list(spec)
        per_column = self.epsilon * self.marginal_share / len(self.columns)
        self.per_column_epsilon = per_column
        self.marginals = {}
        for c in self.columns:
            values = frame[c].to_numpy()
            if spec[c] in (KIND_CATEGORICAL, KIND_BINARY):
                levels = sorted(pd.unique(values).tolist())
                self.marginals[c] = _DPDiscrete(values, levels, per_column, rng)
            else:
                lo, hi = float(np.min(values)), float(np.max(values))
                integer = spec[c] == KIND_INTEGER
                self.marginals[c] = _DPNumeric(values, lo, hi, per_column, rng, integer=integer)
        return self

    def fit(self, frame, spec, rng):
        return self._fit_marginals(frame, spec, rng)

    def sample(self, n, rng):
        return pd.DataFrame({c: self.marginals[c].sample(n, rng) for c in self.columns})

    def hyperparams(self):
        return {"epsilon": self.epsilon, "mechanism": "laplace_histograms",
                "per_column_epsilon": round(self.per_column_epsilon, 6), "bins": N_BINS}


class DPGaussianCopula(DPMarginals):
    """Noisy marginals joined by a correlation matrix measured under the remaining budget."""

    name = "dp_copula"
    marginal_share = 0.7

    def fit(self, frame, spec, rng):
        self._fit_marginals(frame, spec, rng)
        n = len(frame)

        z = np.column_stack([
            self.marginals[c].to_normal(frame[c].to_numpy(), rng) for c in self.columns
        ])
        z = np.clip(z, -CLIP, CLIP)
        z -= z.mean(axis=0)
        sd = z.std(axis=0)
        z /= np.where(sd > 0, sd, 1.0)

        corr_epsilon = self.epsilon * (1 - self.marginal_share)
        cov = z.T @ z / n
        scale = (2 * CLIP ** 2 / n) / corr_epsilon
        noise = rng.laplace(0.0, scale, size=cov.shape)
        noisy = cov + (noise + noise.T) / 2

        lam = ledoit_wolf_shrinkage(z)
        self.corr = _nearest_correlation((1 - lam) * noisy + lam * np.eye(len(self.columns)))
        self.fitted_shrinkage = lam
        self.corr_epsilon = corr_epsilon
        return self

    def sample(self, n, rng):
        z = rng.multivariate_normal(np.zeros(len(self.columns)), self.corr, size=n, method="cholesky")
        out = {}
        for j, c in enumerate(self.columns):
            m = self.marginals[c]
            out[c] = m.from_normal(z[:, j], rng) if isinstance(m, _DPNumeric) else m.from_normal(z[:, j])
        return pd.DataFrame(out)

    def hyperparams(self):
        return {**super().hyperparams(), "mechanism": "laplace_histograms+noisy_correlation",
                "marginal_share": self.marginal_share, "correlation_epsilon": round(self.corr_epsilon, 6),
                "clip": CLIP, "fitted_shrinkage": round(float(self.fitted_shrinkage), 4)}
