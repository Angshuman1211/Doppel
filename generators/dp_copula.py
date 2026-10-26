"""Differentially private baselines: noisy one-way marginals, optionally joined by a noisy Gaussian copula.

Post-course extension work (improvement plan item 4): the benchmark has no arm with a formal guarantee,
so every privacy statement in it is attack-relative. These two arms carry one.

Guarantee: (epsilon, delta)-differential privacy at the level of one admission (one row), bounded
(replace-one) neighbours, with the dataset size n public. Patient-level privacy follows by group privacy
over a patient's admissions (up to 15 in the MIMIC demo), so it is much weaker and is not claimed.

What makes it DP end to end, and what the first version (2026-10-22) got wrong:

1. **Nothing is read from the training data except through noise.** These generators do not use the
   shared `FrameCodec`. That codec is fitted on the training rows: its ICD-9 vocabulary, its rare-code
   pool and the counts used to draw rare codes, its lab fill values and its clipping bounds are all exact
   functions of the private data, so any output decoded through it leaks them. Here the generator works on
   the raw table against a **public domain**: the published schema's categorical levels, the public ICD-9
   dictionary (`output/icd9_lookup.csv`, 14,567 codes) and fixed clinical bounds for every numeric column
   (`PUBLIC_BOUNDS`), set from clinical knowledge rather than measured on the data.
2. **Noise is calibrated to everything that is released.** All one-way histograms are measured in one
   Gaussian mechanism whose L2 sensitivity covers every column at once (a replaced row moves each
   histogram by sqrt(2)). The ICD-9 code counts are a second measurement over the whole dictionary, with
   each admission's contribution capped at `MAX_SECONDARY` codes. The copula's correlation matrix is a
   third, with each record's latent vector clipped in L2 norm and the noise calibrated to the whole matrix
   (the first version calibrated it per entry, which over a 110 x 110 matrix overspent the budget by a
   factor of thousands).
3. **Every later step is post-processing of those noisy measurements**: clipping, renormalising,
   thresholding small code counts away, projecting to a correlation matrix, shrinking it by an amount set
   only from the public noise level. None of them looks at the data again.

Accounting uses zero-concentrated DP: a Gaussian measurement with L2 sensitivity D and noise sigma costs
rho = D^2 / (2 sigma^2), costs add, and rho converts to (epsilon, delta)-DP by
epsilon = rho + 2 sqrt(rho ln(1/delta)) (Bun and Steinke, 2016). The budget is split across the
measurements by `SHARES`.

Public assumptions, stated because they are assumptions: the column list, the categorical levels of the
published schema, the 20 lab items of the schema, the ICD-9 dictionary, the clinical bounds and n.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import schema as S
from .base import Generator

ICD9_DICTIONARY = Path("output/icd9_lookup.csv")
MAX_SECONDARY = 38          # contribution cap: secondary codes per admission (MIMIC-III records up to 39 in total)
N_BINS = 10                 # bins per bounded numeric column
DELTA = 1e-5                # well below 1/n for any cohort in this project
CODE_THRESHOLD_SD = 3.0     # post-processing: drop dictionary codes whose noisy count is below this many sigma
COPULA_PRIOR_SD = 0.2       # public prior scale for a true correlation, used only to set the shrinkage

# Public domain. Categorical levels are those of the published schema (docs/schema_and_feature_dictionary.md).
LEVELS = {
    "admission_type": ["ELECTIVE", "EMERGENCY", "NEWBORN", "URGENT"],
    "ethnicity": ["ASIAN", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO - PUERTO RICAN", "OTHER",
                  "UNKNOWN/NOT SPECIFIED", "WHITE"],
    "first_careunit": ["CCU", "CSRU", "MICU", "NICU", "SICU", "TSICU"],
    "gender": ["F", "M"],
}
BINARY = ["hospital_expire_flag", "readmit_30d"]

# Clinical bounds in MIMIC-III units, set from reference knowledge, not from the data. Values outside are
# clipped before measurement, which is part of the mechanism and costs nothing in privacy.
PUBLIC_BOUNDS = {
    "age": (0, 89), "los_hospital_days": (0.0, 60.0), "los_icu_days": (0.0, 40.0),
    "50882": (0.0, 50.0),     # bicarbonate, mEq/L
    "51301": (0.0, 60.0),     # white blood cells, K/uL
    "50902": (70.0, 140.0),   # chloride, mEq/L
    "51265": (0.0, 1000.0),   # platelet count, K/uL
    "50912": (0.0, 15.0),     # creatinine, mg/dL
    "51279": (1.0, 7.0),      # red blood cells, m/uL
    "51277": (10.0, 30.0),    # RDW, %
    "50868": (0.0, 40.0),     # anion gap, mEq/L
    "51250": (60.0, 130.0),   # MCV, fL
    "50971": (2.0, 8.0),      # potassium, mEq/L
    "51006": (0.0, 200.0),    # urea nitrogen, mg/dL
    "50983": (115.0, 165.0),  # sodium, mEq/L
    "51222": (4.0, 20.0),     # hemoglobin, g/dL
    "51221": (10.0, 60.0),    # hematocrit, %
    "51248": (20.0, 40.0),    # MCH, pg
    "50931": (40.0, 500.0),   # glucose, mg/dL
    "51249": (28.0, 38.0),    # MCHC, %
    "50960": (0.5, 5.0),      # magnesium, mg/dL
    "50970": (0.5, 12.0),     # phosphate, mg/dL
    "50893": (5.0, 13.0),     # calcium, total, mg/dL
}
FRACTION_BOUNDS = (0.0, 1.0)

# Columns joined by the copula: a small public set, because the noise on a correlation matrix grows with
# its size and at n near 100 a larger matrix would be pure noise.
COPULA_COLUMNS = ["age", "los_hospital_days", "los_icu_days", "n_secondary", "hospital_expire_flag",
                  "gender", "admission_type", "first_careunit"]

SHARES = {"marginals": 0.65, "codes": 0.35}
SHARES_COPULA = {"marginals": 0.55, "codes": 0.30, "copula": 0.15}


def rho_from_epsilon(epsilon: float, delta: float = DELTA) -> float:
    """Largest zCDP rho whose (epsilon, delta) conversion stays within epsilon."""
    log_term = np.log(1.0 / delta)
    return float((np.sqrt(log_term + epsilon) - np.sqrt(log_term)) ** 2)


def gaussian_sigma(l2_sensitivity: float, rho: float) -> float:
    return float(l2_sensitivity / np.sqrt(2.0 * rho))


def _public_dictionary() -> np.ndarray:
    return pd.read_csv(ICD9_DICTIONARY, dtype=str)["icd9_code"].str.strip().to_numpy(dtype=object)


class _Histogram:
    """A noisy histogram over public bins; `index` maps a record to its bin, `draw` samples values."""

    def __init__(self, kind, bins, noisy_counts):
        self.kind, self.bins = kind, bins
        p = np.clip(noisy_counts, 0.0, None)
        self.p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1.0 / len(p))
        self.upper = np.cumsum(self.p)
        self.lower = self.upper - self.p

    def cell(self, u):
        return np.minimum(np.searchsorted(self.upper, u, side="right"), len(self.p) - 1)

    def value(self, k, u, rng):
        """Value for bin k; `u` positions the draw inside the bin so a copula can steer it."""
        if self.kind == "level":
            return np.asarray(self.bins, dtype=object)[k]
        lo, hi, integer, extra = self.bins
        k = np.asarray(k)
        is_extra = (k == len(self.p) - 1) if extra is not None else np.zeros(len(k), dtype=bool)
        kk = np.minimum(k, N_BINS - 1)
        width = (hi - lo) / N_BINS
        within = np.clip((u - self.lower[k]) / np.maximum(self.p[k], 1e-12), 0.0, 1.0)
        x = lo + (kk + within) * width
        if integer:
            x = np.clip(np.floor(x), lo, hi)
        return np.where(is_extra, np.nan if extra is None or extra == "missing" else extra, x)


def _numeric_bin(values, lo, hi, extra):
    """Bin index over the public range; `extra` adds one cell for missing values or for a point value."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    k = np.clip(np.floor((np.clip(np.nan_to_num(v, nan=lo), lo, hi) - lo) / ((hi - lo) / N_BINS)), 0, N_BINS - 1)
    if extra == "missing":
        k = np.where(np.isnan(v), N_BINS, k)
    elif extra is not None:
        k = np.where(v >= extra, N_BINS, k)
    return k.astype(int)


class DPMarginals(Generator):
    """Each column sampled from its own noisy histogram; ICD-9 codes from a noisy dictionary count."""

    name = "dp_marginals"
    table_level = True          # works on the raw table against a public domain, never through FrameCodec
    shares = SHARES

    def __init__(self, epsilon: float = 1.0, delta: float = DELTA):
        self.epsilon, self.delta = float(epsilon), float(delta)

    # ------------------------------------------------------------------ the public domain

    def _domain(self, train: pd.DataFrame):
        """Per column: (kind, public bins). Uses only column names and the public tables above."""
        lab_ids = S.lab_ids(train)
        self.columns = list(train.columns)
        self.lab_ids = lab_ids
        dom = {c: ("level", LEVELS[c]) for c in LEVELS}
        dom |= {c: ("level", [0, 1]) for c in BINARY}
        # Age 89 is MIMIC's mask for every age above 89 (age_89_plus), so it gets its own cell.
        dom["age"] = ("numeric", (PUBLIC_BOUNDS["age"][0], S.AGE_CAP - 1, True, S.AGE_CAP))
        dom["los_hospital_days"] = ("numeric", (*PUBLIC_BOUNDS["los_hospital_days"], False, None))
        # Every admission in the schema has an ICU stay (the cohort is built from ICUSTAYS), so no missing cell.
        dom["los_icu_days"] = ("numeric", (*PUBLIC_BOUNDS["los_icu_days"], False, None))
        dom["n_secondary"] = ("level", list(range(MAX_SECONDARY + 1)))
        for i in lab_ids:
            mean_col, frac_col = S.lab_pair(i)
            dom[mean_col] = ("numeric", (*PUBLIC_BOUNDS[i], False, "missing"))
            dom[frac_col] = ("numeric", (*FRACTION_BOUNDS, False, "missing"))
        return dom

    @staticmethod
    def _cells(train, column, kind, bins):
        if column == "n_secondary":
            secondary = train[S.CODES_COL].map(lambda raw: len(set(json.loads(raw)[1:])))
            return np.minimum(secondary.to_numpy(), MAX_SECONDARY)
        if kind == "level":
            position = {v: k for k, v in enumerate(bins)}
            values = train[column].astype(str) if column in LEVELS else train[column].astype(int)
            # A value outside the public domain lands in the first level: clipping, still DP.
            return np.array([position.get(v, 0) for v in values])
        lo, hi, _, extra = bins
        return _numeric_bin(train[column], lo, hi, extra)

    @staticmethod
    def _n_cells(kind, bins):
        return len(bins) if kind == "level" else N_BINS + (1 if bins[3] is not None else 0)

    # ------------------------------------------------------------------ measurements

    def _measure_marginals(self, train, rho, rng):
        """All one-way histograms in one Gaussian mechanism. Replacing a row moves every histogram by
        one count down and one up, so the L2 sensitivity of the concatenation is sqrt(2 * columns)."""
        self.cells = {c: self._cells(train, c, *self.domain[c]) for c in self.domain}
        sigma = gaussian_sigma(np.sqrt(2.0 * len(self.domain)), rho)
        self.histograms = {}
        for c, (kind, bins) in self.domain.items():
            counts = np.bincount(self.cells[c], minlength=self._n_cells(kind, bins)).astype(float)
            self.histograms[c] = _Histogram(kind, bins, counts + rng.normal(0.0, sigma, len(counts)))
        self.sigma_marginals = sigma

    def _measure_codes(self, train, rho, rng):
        """Primary and secondary ICD-9 codes as two counts over the public dictionary. A row contributes
        one primary code and at most MAX_SECONDARY distinct secondary codes, so replacing it changes the
        primary count by sqrt(2) and the secondary count by at most sqrt(2 * MAX_SECONDARY) in L2."""
        dictionary = _public_dictionary()
        position = {c: k for k, c in enumerate(dictionary)}
        codes = train[S.CODES_COL].map(json.loads)
        primary = np.zeros(len(dictionary))
        secondary = np.zeros(len(dictionary))
        for row in codes:
            if row and row[0] in position:
                primary[position[row[0]]] += 1
            for c in sorted(set(row[1:]))[:MAX_SECONDARY]:
                if c in position:
                    secondary[position[c]] += 1
        sigma = gaussian_sigma(np.sqrt(2.0 + 2.0 * MAX_SECONDARY), rho)
        noisy_primary = primary + rng.normal(0.0, sigma, len(dictionary))
        noisy_secondary = secondary + rng.normal(0.0, sigma, len(dictionary))
        self.sigma_codes = sigma
        self.dictionary = dictionary
        self.primary_p = self._thresholded(noisy_primary, sigma)
        self.secondary_p = self._thresholded(noisy_secondary, sigma)

    @staticmethod
    def _thresholded(noisy, sigma):
        """Post-processing: keep codes whose noisy count clears the threshold, else spread uniformly."""
        kept = np.where(noisy > CODE_THRESHOLD_SD * sigma, noisy, 0.0)
        if kept.sum() <= 0:
            return np.full(len(noisy), 1.0 / len(noisy))
        return kept / kept.sum()

    # ------------------------------------------------------------------ fit and sample

    def fit_table(self, train: pd.DataFrame, rng: np.random.Generator) -> "DPMarginals":
        self.n = len(train)
        self.rho = rho_from_epsilon(self.epsilon, self.delta)
        self.domain = self._domain(train)
        self._measure_marginals(train, self.rho * self.shares["marginals"], rng)
        self._measure_codes(train, self.rho * self.shares["codes"], rng)
        return self

    def _uniforms(self, n, rng):
        """Independent uniforms per column; the copula subclass replaces some with correlated ones."""
        return {c: rng.uniform(size=n) for c in self.domain}

    def _sample_values(self, n, rng):
        """Column values on the public domain; subclasses change how columns depend on each other."""
        u = self._uniforms(n, rng)
        return {c: h.value(h.cell(u[c]), u[c], rng) for c, h in self.histograms.items()}

    def sample_table(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        values = self._sample_values(n, rng)

        out = {S.ID_COLS[0]: S.SYNTH_SUBJECT_BASE + np.arange(n), S.ID_COLS[1]: S.SYNTH_HADM_BASE + np.arange(n),
               S.SPLIT_COL: np.full(n, S.SYNTH_SPLIT, dtype=object)}
        for c in LEVELS:
            out[c] = values[c].astype(str)
        for c in BINARY:
            out[c] = values[c].astype(int)
        out["age"] = values["age"].astype(int)
        out[S.AGE_FLAG_COL] = out["age"] >= S.AGE_CAP
        out["los_hospital_days"] = values["los_hospital_days"]
        out["los_icu_days"] = values["los_icu_days"]
        for i in self.lab_ids:
            mean_col, frac_col = S.lab_pair(i)
            mean, frac = values[mean_col], values[frac_col]
            # A lab pair is measured together: no mean means no fraction, and a mean needs a fraction.
            fallback = self.histograms[frac_col]
            redraw = np.isnan(frac) & ~np.isnan(mean)
            if redraw.any():
                p = fallback.p[:N_BINS] / max(fallback.p[:N_BINS].sum(), 1e-12)
                k = rng.choice(N_BINS, size=int(redraw.sum()), p=p)
                frac[redraw] = fallback.value(k, fallback.lower[k] + rng.uniform(size=len(k)) * fallback.p[k], rng)
            frac[np.isnan(mean)] = np.nan
            out[mean_col], out[frac_col] = mean, frac

        primary = rng.choice(self.dictionary, size=n, p=self.primary_p)
        codes = []
        for r in range(n):
            k = int(values["n_secondary"][r])
            p = self.secondary_p.copy()
            p[self.dictionary == primary[r]] = 0.0
            available = int((p > 0).sum())
            k = min(k, available)
            drawn = rng.choice(self.dictionary, size=k, replace=False, p=p / p.sum()) if k else []
            codes.append([str(primary[r])] + [str(c) for c in drawn])
        out[S.PRIMARY_COL] = primary.astype(str)
        out[S.CODES_COL] = [json.dumps(c) for c in codes]
        out[S.N_DIAG_COL] = np.array([len(c) for c in codes])
        return pd.DataFrame(out)[self.columns]

    # The frame-level interface is not used: these arms bypass the codec on purpose.
    def fit(self, frame, spec, rng):
        raise TypeError(f"{self.name} is table-level; call synthesize(), which uses fit_table()")

    def sample(self, n, rng):
        raise TypeError(f"{self.name} is table-level; call synthesize(), which uses sample_table()")

    def hyperparams(self):
        out = {"epsilon": self.epsilon, "delta": self.delta, "unit": "admission",
               "mechanism": "gaussian_histograms_zcdp", "bins": N_BINS, "max_secondary_codes": MAX_SECONDARY,
               "budget_shares": dict(self.shares)}
        if hasattr(self, "rho"):
            out |= {"rho": round(self.rho, 6), "sigma_marginals": round(self.sigma_marginals, 3),
                    "sigma_codes": round(self.sigma_codes, 3)}
        return out


class DPGaussianCopula(DPMarginals):
    """The same noisy marginals, with the core columns joined by a noisy latent correlation matrix."""

    name = "dp_copula"
    shares = SHARES_COPULA

    def fit_table(self, train, rng):
        super().fit_table(train, rng)
        rho_copula = self.rho * self.shares["copula"]

        # Latent normal per record, through the *noisy* marginals (post-processing of DP output) with
        # jitter inside the record's bin, then each record's vector is clipped to L2 norm sqrt(d).
        d = len(COPULA_COLUMNS)
        z = np.column_stack([self._latent(c, rng) for c in COPULA_COLUMNS])
        clip_norm = np.sqrt(d)
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        z = z * np.minimum(1.0, clip_norm / np.maximum(norms, 1e-12))

        # Replacing one record changes sum(z z^T) by at most 2 * clip_norm^2 in Frobenius norm, and the
        # upper triangle's L2 norm is no larger, so that is the sensitivity of the released entries.
        sigma = gaussian_sigma(2.0 * clip_norm ** 2 / self.n, rho_copula)
        cov = z.T @ z / self.n
        noise = np.triu(rng.normal(0.0, sigma, (d, d)))
        noisy = cov + noise + np.triu(noise, 1).T

        # Post-processing: to a correlation matrix, then shrink toward independence by an amount set from
        # the public noise level alone (sigma against a public prior scale for a true correlation).
        diag = np.sqrt(np.clip(np.diag(noisy), 1e-3, None))
        corr = _nearest_correlation(noisy / np.outer(diag, diag))
        shrink = sigma ** 2 / (sigma ** 2 + COPULA_PRIOR_SD ** 2)
        self.corr = (1 - shrink) * corr + shrink * np.eye(d)
        self.sigma_copula, self.copula_shrinkage = sigma, shrink
        return self

    def _latent(self, column, rng):
        h, k = self.histograms[column], self.cells[column]
        u = rng.uniform(h.lower[k], h.upper[k])
        return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))

    def _uniforms(self, n, rng):
        u = super()._uniforms(n, rng)
        z = rng.multivariate_normal(np.zeros(len(COPULA_COLUMNS)), self.corr, size=n, method="cholesky")
        for j, c in enumerate(COPULA_COLUMNS):
            u[c] = norm.cdf(z[:, j])
        return u

    def hyperparams(self):
        out = super().hyperparams() | {"mechanism": "gaussian_histograms+noisy_correlation_zcdp",
                                       "copula_columns": COPULA_COLUMNS}
        if hasattr(self, "sigma_copula"):
            out |= {"sigma_copula": round(self.sigma_copula, 3),
                    "copula_shrinkage": round(float(self.copula_shrinkage), 4)}
        return out


class DPBayesNet(DPMarginals):
    """A differentially private Bayesian network with a public structure: in-hospital death is the root
    and every other column is its child, the degree-1 network of PrivBayes (Zhang et al., 2017) with the
    structure fixed from the prediction task instead of selected from the data.

    Measured: the death histogram and, for every other column, its joint histogram with death. Each record
    adds one count to each of these tables, so the L2 sensitivity of the whole set is sqrt(2 * columns),
    the same as for the one-way marginals: conditioning on death costs no extra budget. Each column's
    marginal is recovered by summing its noisy joint over death, which is post-processing.
    """

    name = "dp_bayesnet"
    root = "hospital_expire_flag"

    def _measure_marginals(self, train, rho, rng):
        self.cells = {c: self._cells(train, c, *self.domain[c]) for c in self.domain}
        sigma = gaussian_sigma(np.sqrt(2.0 * len(self.domain)), rho)
        y = self.cells[self.root]
        n_classes = self._n_cells(*self.domain[self.root])
        self.histograms, self.conditionals = {}, {}
        for c, (kind, bins) in self.domain.items():
            m = self._n_cells(kind, bins)
            if c == self.root:
                counts = np.bincount(y, minlength=m).astype(float)
                self.histograms[c] = _Histogram(kind, bins, counts + rng.normal(0.0, sigma, m))
                continue
            joint = np.zeros((n_classes, m))
            np.add.at(joint, (y, self.cells[c]), 1.0)
            noisy = joint + rng.normal(0.0, sigma, joint.shape)
            self.conditionals[c] = [_Histogram(kind, bins, noisy[k]) for k in range(n_classes)]
            self.histograms[c] = _Histogram(kind, bins, noisy.sum(axis=0))
        self.sigma_marginals = sigma

    def _sample_values(self, n, rng):
        root = self.histograms[self.root]
        u_root = rng.uniform(size=n)
        y = root.cell(u_root)
        values = {self.root: root.value(y, u_root, rng)}
        for c, per_class in self.conditionals.items():
            out = np.empty(n, dtype=object if per_class[0].kind == "level" else float)
            for k, h in enumerate(per_class):
                rows = np.flatnonzero(y == k)
                if len(rows):
                    u = rng.uniform(size=len(rows))
                    out[rows] = h.value(h.cell(u), u, rng)
            values[c] = out
        return values

    def hyperparams(self):
        return super().hyperparams() | {"mechanism": "gaussian_bayes_net_public_structure_zcdp",
                                        "network": f"{self.root} -> every other column"}


def _nearest_correlation(matrix: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive-definite correlation matrix (eigenvalue clip)."""
    sym = (matrix + matrix.T) / 2
    vals, vecs = np.linalg.eigh(sym)
    psd = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    corr = psd / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr
