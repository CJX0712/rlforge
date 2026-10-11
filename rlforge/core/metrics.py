"""Scalar metrics shared across algorithms and gates."""
from __future__ import annotations
import numpy as np


def discounted_return(rewards, gamma: float = 1.0) -> float:
    r = 0.0
    g = 1.0
    for rew in rewards:
        r += g * rew
        g *= gamma
    return float(r)


def mean_episode_return(returns: list[float]) -> float:
    return float(np.mean(returns)) if returns else 0.0


def improvement(after: float, before: float) -> float:
    """Relative improvement of `after` over `before` (baseline)."""
    if abs(before) < 1e-9:
        return float(after - before)
    return float((after - before) / (abs(before) + 1e-9))
