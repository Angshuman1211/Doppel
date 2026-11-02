"""Sequential CART in the style of synthpop (Nowok et al., 2016), the standard baseline in health statistics.

Each column is modelled by a tree on the columns before it, and a value is drawn from the training values
in the leaf a synthetic row falls into. The first column is drawn from its empirical distribution. This is
the synthpop `cart` method without smoothing, with its default minimum leaf size of 5, reimplemented on the
shared modelling frame; it is not the R package.

Drawing leaf values means every generated value is a training value, so this generator is expected to be
exposed, and that is the reason for including it.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import Generator
from .codec import KIND_CONTINUOUS, KIND_INTEGER


class SequentialCART(Generator):
    name = "sequential_cart"

    def __init__(self, min_samples_leaf: int = 5, ccp_alpha: float = 1e-8):
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha

    def _design(self, frame: pd.DataFrame, upto: int) -> np.ndarray:
        parts = []
        for c in self.columns[:upto]:
            if c in self.encoders:
                parts.append(self.encoders[c].transform(frame[[c]].astype(str)))
            else:
                parts.append(frame[[c]].to_numpy(float))
        return np.hstack(parts)

    def fit(self, frame: pd.DataFrame, spec: dict[str, str], rng: np.random.Generator) -> "SequentialCART":
        self.columns = list(spec)
        self.numeric = {c for c, k in spec.items() if k in (KIND_INTEGER, KIND_CONTINUOUS)}
        self.encoders = {c: OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(frame[[c]].astype(str))
                         for c in self.columns if c not in self.numeric}
        self.first = frame[self.columns[0]].to_numpy()
        self.models = {}
        for j, c in enumerate(self.columns[1:], 1):
            X = self._design(frame, j)
            values = frame[c].to_numpy()
            kwargs = dict(min_samples_leaf=self.min_samples_leaf, ccp_alpha=self.ccp_alpha,
                          random_state=int(rng.integers(2**31)))
            if c in self.numeric:
                tree = DecisionTreeRegressor(**kwargs).fit(X, values.astype(float))
            else:
                tree = DecisionTreeClassifier(**kwargs).fit(X, values.astype(str))
            leaves = tree.apply(X)
            self.models[c] = (tree, {leaf: values[leaves == leaf] for leaf in np.unique(leaves)})
        return self

    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        out = pd.DataFrame({self.columns[0]: rng.choice(self.first, size=n)})
        for j, c in enumerate(self.columns[1:], 1):
            tree, donors = self.models[c]
            leaves = tree.apply(self._design(out, j))
            drawn = np.empty(n, dtype=object)
            for leaf in np.unique(leaves):
                where = leaves == leaf
                drawn[where] = rng.choice(donors[leaf], size=int(where.sum()))
            out[c] = pd.Series(drawn).infer_objects() if c not in self.numeric else drawn.astype(float)
        return out[self.columns]

    def hyperparams(self) -> dict:
        return {"min_samples_leaf": self.min_samples_leaf, "ccp_alpha": self.ccp_alpha}
