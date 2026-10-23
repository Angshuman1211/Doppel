"""A schema-agnostic codec, so the generators can be run on a dataset other than MIMIC-III demo.

Post-course extension work (improvement plan item 1). The generators themselves are already generic:
they take a modeling frame and a spec of column kinds. Only `generators/codec.py` and the Stage 2
contract are MIMIC-specific, with ICD-9 multi-hot columns, `lab_<itemid>` pairs and `age_89_plus` rules
baked in. This builds the same frame/spec interface from any tabular CSV, which is what the size sweep
needs, and it deliberately lives in `research/` so the graded pipeline is untouched.

Differences from the MIMIC codec, all stated rather than hidden:

- no list-valued column, so there is no multi-hot vocabulary and no rare-code tail pool;
- high-cardinality categoricals are capped at `max_levels` by frequency, with the remainder mapped to
  OTHER, the same move Track 3 made for ethnicity;
- numeric columns that contain nulls get a `miss_<col>` indicator, as in the MIMIC codec;
- constant columns are dropped, because a generator cannot learn anything from them and CTGAN's
  transformer rejects them.
"""

import numpy as np
import pandas as pd

from generators.codec import KIND_BINARY, KIND_CATEGORICAL, KIND_CONTINUOUS, KIND_INTEGER

OTHER = "__OTHER__"
MISSING = "__MISSING__"


def miss_col(column: str) -> str:
    return f"miss_{column}"


class TabularCodec:
    """Encodes an arbitrary table into the (frame, spec) pair the generators expect, and back.

    `id_cols` are carried for splitting but never modeled. A column is categorical if it is non-numeric
    or has at most `max_categorical_ints` distinct values, which catches integer category codes such as
    `admission_type_id`.
    """

    def __init__(self, id_cols=(), max_levels: int = 25, max_categorical_ints: int = 12):
        self.id_cols = list(id_cols)
        self.max_levels = max_levels
        self.max_categorical_ints = max_categorical_ints

    def fit(self, df: pd.DataFrame) -> "TabularCodec":
        self.columns = [c for c in df.columns if c not in self.id_cols]
        self.dropped = [c for c in self.columns if df[c].nunique(dropna=False) <= 1]
        self.columns = [c for c in self.columns if c not in self.dropped]

        self.levels, self.bounds, self.nullable = {}, {}, []
        spec = {}
        for c in self.columns:
            s = df[c]
            numeric = pd.api.types.is_numeric_dtype(s)
            if not numeric or s.nunique(dropna=True) <= self.max_categorical_ints:
                counts = s.astype(object).where(s.notna(), MISSING).astype(str).value_counts()
                keep = list(counts.index[: self.max_levels])
                if len(counts) > self.max_levels:
                    keep.append(OTHER)
                self.levels[c] = keep
                spec[c] = KIND_BINARY if len(keep) == 2 else KIND_CATEGORICAL
            else:
                if s.isna().any():
                    self.nullable.append(c)
                    spec[miss_col(c)] = KIND_BINARY
                values = s.dropna()
                self.bounds[c] = (float(values.min()), float(values.max()))
                integer = pd.api.types.is_integer_dtype(s) or (values % 1 == 0).all()
                spec[c] = KIND_INTEGER if integer else KIND_CONTINUOUS
        # Indicators come first so the frame's column order is stable and readable.
        self.spec = {c: spec[c] for c in sorted(spec, key=lambda c: (not c.startswith("miss_"), c))}
        self.fill = {c: float(df[c].median()) for c in self.nullable}
        return self

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.columns:
            s = df[c]
            if c in self.levels:
                text = s.astype(object).where(s.notna(), MISSING).astype(str)
                allowed = set(self.levels[c])
                out[c] = text.where(text.isin(allowed), OTHER) if OTHER in allowed else \
                    text.where(text.isin(allowed), self.levels[c][0])
            else:
                if c in self.nullable:
                    out[miss_col(c)] = s.isna().astype(int)
                out[c] = s.fillna(self.fill.get(c, 0.0)).astype(float)
        return pd.DataFrame(out, index=df.index)[list(self.spec)]

    def decode(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.reset_index(drop=True)
        out = {}
        for c in self.columns:
            if c in self.levels:
                out[c] = frame[c].astype(str).to_numpy()
            else:
                lo, hi = self.bounds[c]
                v = np.clip(frame[c].astype(float).to_numpy(), lo, hi)
                if self.spec[c] == KIND_INTEGER:
                    v = np.rint(v)
                if c in self.nullable:
                    v = np.where(frame[miss_col(c)].astype(float).to_numpy() >= 0.5, np.nan, v)
                out[c] = v
        return pd.DataFrame(out)[self.columns]

    def describe(self) -> dict:
        return {"modeled_columns": len(self.spec), "categorical": sum(
                    1 for k in self.spec.values() if k == KIND_CATEGORICAL),
                "binary": sum(1 for k in self.spec.values() if k == KIND_BINARY),
                "numeric": sum(1 for k in self.spec.values() if k in (KIND_INTEGER, KIND_CONTINUOUS)),
                "dropped_constant": self.dropped, "nullable_numeric": self.nullable,
                "max_levels": self.max_levels}
