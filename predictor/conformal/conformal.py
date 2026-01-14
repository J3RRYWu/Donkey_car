"""
Conformal Prediction utilities for trajectory prediction in latent space.

We implement Split Conformal Prediction with *per-horizon* quantiles:
  s_{i,t} = || z_pred_{i,t} - z_true_{i,t} ||  (norm over latent dims)
  q_t = quantile_{ceil((n+1)(1-alpha))/n}({s_{i,t}}_{i=1..n})

At inference, the prediction set for horizon t is:
  C_t(x) = { z : ||z - z_pred_t(x)|| <= q_t }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import numpy as np
import torch


@dataclass
class CPQuantiles:
    alpha: float
    norm: str  # "l2" or "linf"
    q: np.ndarray  # shape (H,)

    def to_dict(self) -> Dict:
        return {"alpha": float(self.alpha), "norm": str(self.norm), "q": self.q.tolist()}

    @staticmethod
    def from_dict(d: Dict) -> "CPQuantiles":
        return CPQuantiles(alpha=float(d["alpha"]), norm=str(d["norm"]), q=np.asarray(d["q"], dtype=np.float64))


def _flatten_latent(z: torch.Tensor) -> torch.Tensor:
    """Return (B, D) by flattening all non-batch dims."""
    return z.reshape(z.shape[0], -1)


def nonconformity_scores(z_pred: torch.Tensor, z_true: torch.Tensor, norm: str = "l2") -> torch.Tensor:
    """Compute per-sample nonconformity scores for a single horizon.
    z_pred, z_true: (B, ...) same shape
    returns: (B,) tensor
    """
    zp = _flatten_latent(z_pred)
    zt = _flatten_latent(z_true)
    diff = zp - zt
    n = str(norm).lower()
    if n == "l2":
        return torch.linalg.vector_norm(diff, ord=2, dim=1)
    if n == "linf":
        return torch.linalg.vector_norm(diff, ord=float("inf"), dim=1)
    raise ValueError(f"Unsupported norm: {norm} (expected 'l2' or 'linf')")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split conformal quantile: k = ceil((n+1)(1-alpha)), return k-th smallest (1-indexed).
    scores: shape (n,)
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = int(s.shape[0])
    if n <= 0:
        return float("nan")
    k = int(math.ceil((n + 1) * (1.0 - float(alpha))))
    k = max(1, min(k, n))
    return float(np.partition(s, k - 1)[k - 1])


def quantiles_per_horizon(scores_by_h: np.ndarray, alpha: float) -> np.ndarray:
    """scores_by_h: shape (n, H). Returns q: shape (H,)."""
    s = np.asarray(scores_by_h, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"scores_by_h must be 2D (n,H), got shape {s.shape}")
    n, H = s.shape
    q = np.full((H,), np.nan, dtype=np.float64)
    for t in range(H):
        q[t] = conformal_quantile(s[:, t], alpha=alpha)
    return q


def coverage_per_horizon(scores_by_h: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Coverage fraction per horizon: P(score <= q_t).
    scores_by_h: (n,H), q: (H,)
    returns: (H,)
    """
    s = np.asarray(scores_by_h, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    H = s.shape[1]
    out = np.full((H,), np.nan, dtype=np.float64)
    for t in range(H):
        st = s[:, t]
        st = st[np.isfinite(st)]
        if st.size == 0 or not np.isfinite(q[t]):
            out[t] = float("nan")
        else:
            out[t] = float(np.mean(st <= q[t]))
    return out


def set_size_summary(q: np.ndarray) -> Dict[str, float]:
    """Simple summary stats of per-horizon radii."""
    q = np.asarray(q, dtype=np.float64)
    qf = q[np.isfinite(q)]
    if qf.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {"mean": float(np.mean(qf)), "median": float(np.median(qf)), "p90": float(np.quantile(qf, 0.9))}


