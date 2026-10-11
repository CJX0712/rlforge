"""Value Iteration — exact dynamic programming.

The Bellman optimality operator T* is a γ-contraction, so iterating it drives
V monotonically (in max-norm) to V*. Gate G3 measures the residual
‖T*V − V‖∞ and asserts it collapses to ~1e-12; that *is* the proof of
convergence, not a plot that "looks flat".
"""
from __future__ import annotations
import numpy as np


def value_iteration(mdp, max_iter=10000, tol=1e-12):
    V = np.zeros(mdp.S)
    for _ in range(max_iter):
        Vn, pi = mdp.bellman_opt_v(V)
        if np.max(np.abs(Vn - V)) < tol:
            return Vn, pi
        V = Vn
    Vn, pi = mdp.bellman_opt_v(V)
    return Vn, pi
