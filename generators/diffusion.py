"""Diffusion arm: a TabDDPM-style denoising diffusion model (Kotelnikov et al., 2023) on the shared modeling frame.

Two processes share one denoiser, as in TabDDPM:

- numeric columns go through the same rank-to-normal transform the copula uses, then standard Gaussian
  diffusion (Ho et al., 2020) with the model predicting the noise;
- each categorical and binary column gets multinomial diffusion (Hoogeboom et al., 2021), where the forward
  process mixes the one-hot toward uniform and the model predicts the clean class.

On 94 rows the network is the main overfitting risk, so it stays small and every size is a hyperparameter
the sweep can move. Trained with the simplified objective: MSE on the Gaussian part, cross-entropy on the
clean class for the multinomial part, which is what most TabDDPM implementations use in place of the full KL.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Generator
from .codec import KIND_CONTINUOUS, KIND_INTEGER
from .copula import _Continuous
from .ctgan_tvae import TORCH_THREADS


def cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal (2021) cosine schedule, returned as alpha_bar for t = 0 .. T."""
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    f = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    return (f / f[0]).clamp(1e-6, 1.0).float()


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _Denoiser(nn.Module):
    def __init__(self, n_numeric: int, cardinalities: list[int], hidden: int, layers: int, time_dim: int, dropout: float):
        super().__init__()
        self.n_numeric = n_numeric
        self.cardinalities = cardinalities
        width = n_numeric + sum(cardinalities)

        self.time = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.time_dim = time_dim

        blocks: list[nn.Module] = []
        in_dim = width + time_dim
        for _ in range(layers):
            blocks += [nn.Linear(in_dim, hidden), nn.SiLU()]
            if dropout:
                blocks.append(nn.Dropout(dropout))
            in_dim = hidden
        blocks.append(nn.Linear(in_dim, width))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.net(torch.cat([x, self.time(timestep_embedding(t, self.time_dim))], dim=-1))
        eps = h[:, : self.n_numeric]
        logits, at = [], self.n_numeric
        for k in self.cardinalities:
            logits.append(h[:, at: at + k])
            at += k
        return eps, logits


class TabularDiffusion(Generator):
    name = "tabular_diffusion"

    def __init__(self, steps: int = 3000, timesteps: int = 1000, hidden: int = 256, layers: int = 2,
                 time_dim: int = 64, lr: float = 1e-3, weight_decay: float = 1e-5, batch_size: int = 64,
                 dropout: float = 0.0):
        self.params = dict(steps=steps, timesteps=timesteps, hidden=hidden, layers=layers, time_dim=time_dim,
                           lr=lr, weight_decay=weight_decay, batch_size=batch_size, dropout=dropout)

    # --- encoding -------------------------------------------------------------------------------

    def _encode(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        numeric = np.column_stack([self.marginals[c].to_normal(frame[c].to_numpy(), None) for c in self.numeric_cols]) \
            if self.numeric_cols else np.zeros((len(frame), 0))
        codes = np.column_stack([[self.levels[c].index(v) for v in frame[c].to_numpy()] for c in self.discrete_cols]) \
            if self.discrete_cols else np.zeros((len(frame), 0), dtype=int)
        return torch.tensor(numeric, dtype=torch.float32), torch.tensor(codes, dtype=torch.long)

    def _one_hot(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.cat([F.one_hot(codes[:, j], k).float() for j, k in enumerate(self.cardinalities)], dim=-1) \
            if self.discrete_cols else torch.zeros((len(codes), 0))

    # --- diffusion processes --------------------------------------------------------------------

    def _q_sample_numeric(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t][:, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def _q_sample_discrete(self, codes: torch.Tensor, t: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Each class keeps probability alpha_bar, the rest is spread uniformly over the column's classes."""
        out = torch.empty_like(codes)
        ab = self.alpha_bar[t]
        for j, k in enumerate(self.cardinalities):
            probs = F.one_hot(codes[:, j], k).float() * ab[:, None] + (1 - ab)[:, None] / k
            out[:, j] = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        return out

    def _posterior_sample(self, j: int, k: int, x_t: torch.Tensor, logits: torch.Tensor, t: int,
                          gen: torch.Generator) -> torch.Tensor:
        """Sample x_{t-1} from q(x_{t-1} | x_t, x0_hat) for one categorical column."""
        alpha, ab_prev = self.alphas[t], self.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)
        x0 = logits.softmax(dim=-1)
        from_t = F.one_hot(x_t, k).float() * alpha + (1 - alpha) / k
        from_0 = x0 * ab_prev + (1 - ab_prev) / k
        probs = from_t * from_0
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.multinomial(probs, 1, generator=gen).squeeze(1)

    # --- Generator interface --------------------------------------------------------------------

    def fit(self, frame: pd.DataFrame, spec: dict[str, str], rng: np.random.Generator) -> "TabularDiffusion":
        torch.set_num_threads(TORCH_THREADS)
        seed = int(rng.integers(2**31))
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 1)

        self.columns = list(spec)
        self.numeric_cols = [c for c, kind in spec.items() if kind in (KIND_INTEGER, KIND_CONTINUOUS)]
        self.discrete_cols = [c for c in self.columns if c not in self.numeric_cols]
        self.marginals = {c: _Continuous(frame[c].to_numpy()) for c in self.numeric_cols}
        self.levels = {c: sorted(frame[c].unique().tolist()) for c in self.discrete_cols}
        self.cardinalities = [len(self.levels[c]) for c in self.discrete_cols]

        alpha_bar = cosine_alpha_bar(self.params["timesteps"])
        self.alpha_bar = alpha_bar[1:]
        self.alphas = (alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-6, 1.0)
        self.betas = 1 - self.alphas

        x_num, codes = self._encode(frame)
        self.model = _Denoiser(len(self.numeric_cols), self.cardinalities, self.params["hidden"],
                               self.params["layers"], self.params["time_dim"], self.params["dropout"])
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"])

        n, batch = len(frame), min(self.params["batch_size"], len(frame))
        self.model.train()
        for _ in range(self.params["steps"]):
            idx = torch.randint(0, n, (batch,), generator=gen)
            t = torch.randint(0, self.params["timesteps"], (batch,), generator=gen)
            noise = torch.randn(len(idx), len(self.numeric_cols), generator=gen)

            x_t = self._q_sample_numeric(x_num[idx], t, noise)
            c_t = self._q_sample_discrete(codes[idx], t, gen)
            eps_hat, logits = self.model(torch.cat([x_t, self._one_hot(c_t)], dim=-1), t)

            loss = F.mse_loss(eps_hat, noise) if self.numeric_cols else torch.zeros(())
            if self.discrete_cols:
                ce = [F.cross_entropy(logit, codes[idx][:, j]) for j, logit in enumerate(logits)]
                loss = loss + torch.stack(ce).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        self.final_loss = float(loss.detach())
        return self

    @torch.no_grad()
    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        torch.set_num_threads(TORCH_THREADS)
        seed = int(rng.integers(2**31))
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 1)
        self.model.eval()

        x = torch.randn(n, len(self.numeric_cols), generator=gen)
        codes = torch.stack([torch.randint(0, k, (n,), generator=gen) for k in self.cardinalities], dim=-1) \
            if self.discrete_cols else torch.zeros((n, 0), dtype=torch.long)

        for t in range(self.params["timesteps"] - 1, -1, -1):
            t_batch = torch.full((n,), t, dtype=torch.long)
            eps_hat, logits = self.model(torch.cat([x, self._one_hot(codes)], dim=-1), t_batch)

            alpha, ab = self.alphas[t], self.alpha_bar[t]
            mean = (x - (1 - alpha) / (1 - ab).sqrt() * eps_hat) / alpha.sqrt()
            if t > 0:
                var = self.betas[t] * (1 - self.alpha_bar[t - 1]) / (1 - ab)
                x = mean + var.sqrt() * torch.randn(x.shape, generator=gen)
            else:
                x = mean
            codes = torch.stack([self._posterior_sample(j, k, codes[:, j], logits[j], t, gen)
                                 for j, k in enumerate(self.cardinalities)], dim=-1) \
                if self.discrete_cols else codes

        out = {c: self.marginals[c].from_normal(x[:, j].numpy()) for j, c in enumerate(self.numeric_cols)}
        out |= {c: np.array([self.levels[c][i] for i in codes[:, j].tolist()])
                for j, c in enumerate(self.discrete_cols)}
        return pd.DataFrame(out)[self.columns]

    def hyperparams(self) -> dict:
        return dict(self.params)
