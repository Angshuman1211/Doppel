"""AIM: a graphical-model DP baseline, for the journal version (Track 1 task 1).

The three DP arms in `dp_copula.py` spend their budget on one-way histograms, or on one-way histograms
plus a small correlation matrix, or on a fixed degree-1 Bayesian network. All three fix what to measure
before seeing the data. AIM (McKenna et al., 2022) instead *chooses* which marginals to measure, one at a
time, spending a slice of the budget each round on the marginal its current model fits worst. It is the
strongest general-purpose mechanism in this family and the obvious thing a reviewer will ask for.

This wraps the authors' own implementation (ryan112358/private-pgm, Apache 2.0, pinned commit) rather
than reimplementing it. Only the mechanism is theirs; the domain, the ICD-9 code handling and the
decoding are this project's, so the arm is comparable with the others.

What it inherits from `DPMarginals`, and why that matters: the **public domain** (`LEVELS`,
`PUBLIC_BOUNDS`, the public ICD-9 dictionary), the discretisation into public bins, the ICD-9 code
mechanism, and the decoding back to the real schema. Nothing is read from the training rows except
through a noisy measurement. AIM replaces only the one-way histograms.

Two accounting points, both stated because they are easy to get wrong:

- **Bounded vs unbounded.** The other arms are bounded (replace-one) DP. AIM is written for unbounded
  (add/remove-one) DP, and an epsilon-unbounded mechanism is only 2*epsilon-bounded, so AIM is run at
  **half** the arm's epsilon. Without that halving the arm would claim a guarantee twice as strong as it
  has.
- **Composition.** AIM does its own zCDP accounting internally from the (epsilon, delta) it is given.
  Rather than mix that with this module's zCDP bookkeeping, the two measurements are composed in
  (epsilon, delta) by **basic composition**: AIM gets (share * epsilon, delta/2) and the code counts get
  the rest. Slightly looser than a single zCDP account, and unambiguous.

ICD-9 codes stay outside the graphical model on purpose: the public dictionary has 14,567 codes, and a
variable of that cardinality would dominate the model and exhaust the budget on a single marginal. The
AIM authors make the same recommendation for high-cardinality attributes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .dp_copula import (COPULA_COLUMNS, DELTA, DPMarginals, _Histogram, gaussian_sigma,
                        rho_from_epsilon)

# AIM gets the core columns and their dependence; the lab columns keep independent noisy histograms,
# and the ICD-9 dictionary keeps its own measurement.
SHARES_AIM = {"aim": 0.40, "marginals": 0.30, "codes": 0.30}

AIM_REPO = Path(__file__).resolve().parents[1] / "research" / "third_party" / "private-pgm"
AIM_COMMIT = "9429881f0f21ef413424f6011b9d6ae20728176d"      # Apache 2.0, McKenna et al. 2022


def _load_aim():
    """Import the authors' AIM. Their module puts its own directory on sys.path for its siblings."""
    mechanisms = AIM_REPO / "mechanisms"
    if not (mechanisms / "aim.py").is_file():
        raise ModuleNotFoundError(
            f"AIM not found at {mechanisms}. Clone it with:\n"
            f"  git clone https://github.com/ryan112358/private-pgm.git {AIM_REPO}\n"
            f"  git -C {AIM_REPO} checkout {AIM_COMMIT}\n"
            f"  pip install {AIM_REPO}")
    if str(mechanisms) not in sys.path:
        sys.path.insert(0, str(mechanisms))
    # mbi warns about both of these on import, and the first is not optional here: in float32 its
    # marginal estimation diverges on this domain, the model's marginals come back NaN, and AIM's
    # exponential mechanism then fails with "probabilities contain NaN". Must precede any jax use.
    import jax                                           # noqa: E402
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    import aim                                           # noqa: E402
    from mbi import Dataset, Domain                      # noqa: E402
    return aim.AIM, Dataset, Domain


class DPAIM(DPMarginals):
    """One-way histograms replaced by AIM's adaptively chosen graphical model."""

    name = "dp_aim"
    table_level = True
    shares = SHARES_AIM

    def __init__(self, epsilon: float = 1.0, delta: float = DELTA, degree: int = 2,
                 max_cells: int = 2000, rounds: int | None = 50, max_model_size: float = 10.0,
                 max_iters: int = 100, core_columns: list[str] | None = None,
                 shares: dict | None = None):
        super().__init__(epsilon=epsilon, delta=delta)
        # Overridable so the split between AIM and the non-core histograms can be ablated: AIM's share
        # is taken from the budget those histograms would otherwise get, which confounds any fidelity
        # comparison against `dp_marginals` until it is measured.
        if shares is not None:
            self.shares = dict(shares)
        self.degree = int(degree)
        self.max_cells = int(max_cells)
        self.rounds = rounds
        self.max_model_size = float(max_model_size)
        self.max_iters = int(max_iters)
        # AIM models these jointly; everything else keeps an independent noisy histogram. The full
        # 50-column domain is not affordable: its 2-way workload is over a thousand candidate cliques,
        # each of which AIM re-projects the model onto every round, and a single fit did not finish in
        # ten minutes. This is the same restriction `DPGaussianCopula` already makes, and for a related
        # reason — a mechanism that spreads itself over every pair measures none of them.
        self.core_columns = list(COPULA_COLUMNS if core_columns is None else core_columns)

    # ------------------------------------------------------------------ fit

    def fit_table(self, train: pd.DataFrame, rng: np.random.Generator) -> "DPAIM":
        import itertools

        AIM, Dataset, Domain = _load_aim()
        self.n = len(train)
        self.domain = self._domain(train)
        cells = {c: self._cells(train, c, *self.domain[c]) for c in self.domain}
        sizes = {c: self._n_cells(*self.domain[c]) for c in self.domain}

        columns = [c for c in self.core_columns if c in self.domain]
        rest = [c for c in self.domain if c not in columns]

        epsilon_aim = self.epsilon * self.shares["aim"]
        epsilon_rest = self.epsilon * self.shares["marginals"]
        epsilon_codes = self.epsilon - epsilon_aim - epsilon_rest
        # AIM keeps its own zCDP account internally; the other two keep this module's. The three are
        # composed in (epsilon, delta) by basic composition, with delta split three ways, so the arm is
        # (epsilon, delta)-DP overall. Looser than a single zCDP account and unambiguous.
        self._measure_codes(train, rho_from_epsilon(epsilon_codes, self.delta / 3), rng)

        # The non-core columns, measured exactly as DPMarginals does but over this subset only, so the
        # L2 sensitivity covers the histograms actually released here.
        self.cells = {c: self._cells(train, c, *self.domain[c]) for c in self.domain}
        rho_rest = rho_from_epsilon(epsilon_rest, self.delta / 3)
        sigma_rest = gaussian_sigma(np.sqrt(2.0 * len(rest)), rho_rest) if rest else 0.0
        histograms = {}
        for c in rest:
            kind, bins = self.domain[c]
            counts = np.bincount(self.cells[c], minlength=self._n_cells(kind, bins)).astype(float)
            histograms[c] = _Histogram(kind, bins, counts + rng.normal(0.0, sigma_rest, len(counts)))
        self.sigma_marginals = sigma_rest
        # AIM seeds itself with every one-way marginal at sigma^2 = rounds / (2 * 0.9 * rho), which costs
        # 0.9 * columns * rho / rounds before its loop begins. With too few rounds that already exceeds
        # the budget, `remaining` goes negative, sigma becomes sqrt of a negative number and the run dies
        # inside the exponential mechanism with "probabilities contain NaN" rather than with a message
        # about rounds. Fail here instead, where the cause is visible.
        floor = int(np.ceil(0.9 * len(columns))) + 1
        if self.rounds is not None and self.rounds < floor:
            raise ValueError(f"rounds={self.rounds} is too few for {len(columns)} columns: AIM's initial "
                             f"one-way measurements would overspend the budget. Use at least {floor}, "
                             f"or rounds=None for the authors' default of 16 x columns.")
        domain = Domain(tuple(columns), tuple(sizes[c] for c in columns))
        data = Dataset({c: np.asarray(cells[c]) for c in columns}, domain)
        workload = [cl for cl in itertools.combinations(columns, self.degree)
                    if domain.size(cl) <= self.max_cells]
        workload = [(cl, 1.0) for cl in workload]

        # AIM's own code expects the legacy RandomState API for its exponential mechanism.
        seed = int(rng.integers(0, 2**31 - 1))
        mechanism = AIM(epsilon=epsilon_aim / 2.0, delta=self.delta / 3.0,   # unbounded -> bounded
                        prng=np.random.RandomState(seed), rounds=self.rounds,
                        max_model_size=self.max_model_size, max_iters=self.max_iters)
        _model, synthetic = mechanism.run(data, workload, num_synth_rows=max(self.n, 1000))
        frame = pd.DataFrame(synthetic.to_dict())      # mbi 2.x Dataset: columns of cell indices
        self.aim_cells = frame[columns].astype(int).reset_index(drop=True)
        self.aim_columns, self.workload_size = columns, len(workload)
        # Post-processing of AIM's output: per-column counts, used only to place a value inside its bin.
        histograms |= {c: _Histogram(self.domain[c][0], self.domain[c][1],
                                     np.bincount(self.aim_cells[c].to_numpy(), minlength=sizes[c]).astype(float))
                       for c in columns}
        self.histograms = {c: histograms[c] for c in self.domain}
        self.epsilon_aim, self.epsilon_rest, self.epsilon_codes = epsilon_aim, epsilon_rest, epsilon_codes
        return self

    # ------------------------------------------------------------------ sample

    def _sample_values(self, n: int, rng: np.random.Generator) -> dict:
        """Core columns from AIM's joint sample, the rest independently from their own histograms.

        Drawing whole rows of the core keeps the dependence AIM paid for; drawing the rest per column is
        what their independent histograms support, exactly as in `DPMarginals`.
        """
        idx = rng.integers(0, len(self.aim_cells), size=n)
        values = {}
        for c, histogram in self.histograms.items():
            if c in self.aim_columns:
                k = self.aim_cells[c].to_numpy()[idx]
                u = histogram.lower[k] + histogram.p[k] * rng.uniform(size=n)
            else:
                u = rng.uniform(size=n)
                k = histogram.cell(u)
            values[c] = histogram.value(k, u, rng)
        return values

    def hyperparams(self) -> dict:
        out = {"epsilon": self.epsilon, "delta": self.delta, "unit": "admission",
               "mechanism": "aim_private_pgm", "aim_commit": AIM_COMMIT[:12],
               "degree": self.degree, "max_cells": self.max_cells, "rounds": self.rounds,
               "max_model_size": self.max_model_size, "max_iters": self.max_iters,
               "composition": "basic over (AIM, marginals, codes)", "aim_neighbour_relation": "unbounded, run at eps/2"}
        if hasattr(self, "epsilon_aim"):
            out |= {"epsilon_aim": round(self.epsilon_aim, 4), "epsilon_rest": round(self.epsilon_rest, 4),
                    "epsilon_codes": round(self.epsilon_codes, 4), "aim_columns": list(self.aim_columns),
                    "workload_marginals": self.workload_size, "sigma_codes": round(self.sigma_codes, 3)}
        return out
