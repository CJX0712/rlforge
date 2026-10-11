"""Tabular MDP + Bellman operators.

This is the mathematical core of the dynamic-programming and tabular-TD
algorithms. Every operator here is a *contraction mapping* under the max
norm (Banach fixed-point theorem) — that is the invariant that proves
value iteration converges, and it is what gate G3 checks numerically.
"""
from __future__ import annotations
import numpy as np


class TabularMDP:
    """A finite MDP with transition tensor ``P[s,a,s']`` and reward ``R[s,a]``."""

    def __init__(self, P, R, gamma: float = 0.99, terminals=None):
        self.P = np.asarray(P, dtype=float)
        assert self.P.ndim == 3, "P must be (S,A,S)"
        self.S, self.A = self.P.shape[:2]
        # normalise transition rows (defensive)
        self.P = self.P / self.P.sum(axis=2, keepdims=True)
        R = np.asarray(R, dtype=float)
        self.R = np.sum(self.P * R, axis=2) if R.ndim == 3 else R.copy()
        self.gamma = float(gamma)
        self.terminals = set(terminals or [])
        # Sanitise terminals: absorbing, zero reward.
        for s in self.terminals:
            self.P[s] = 0.0
            self.P[s, :, s] = 1.0
            self.R[s] = 0.0

    # ---- Bellman operators -------------------------------------------------
    def bellman_opt_v(self, V):
        """V' = max_a [ R(s,a) + γ Σ P(s,a,·) V ] ; returns (V', π*)."""
        Q = self.R + self.gamma * np.einsum("ijk,k->ij", self.P, V)
        return np.max(Q, axis=1), np.argmax(Q, axis=1)

    def bellman_opt_q(self, Q):
        """Q' = R + γ Σ P(s,a,·) max_a' Q(·,a')."""
        Vmax = np.max(Q, axis=1)
        return self.R + self.gamma * np.einsum("ijk,k->ij", self.P, Vmax)

    def bellman_policy_v(self, V, pi):
        """Given a deterministic policy pi, return T^π V."""
        Rpi = self.R[np.arange(self.S), pi]
        Ppi = self.P[np.arange(self.S), pi]
        return Rpi + self.gamma * (Ppi @ V)

    def eval_policy(self, pi, tol=1e-12, max_iter=100000):
        V = np.zeros(self.S)
        for _ in range(max_iter):
            Vn = self.bellman_policy_v(V, pi)
            if np.max(np.abs(Vn - V)) < tol:
                return Vn
            V = Vn
        return V

    def optimal_v(self, tol=1e-12, max_iter=100000):
        V = np.zeros(self.S)
        for _ in range(max_iter):
            Vn, _ = self.bellman_opt_v(V)
            if np.max(np.abs(Vn - V)) < tol:
                return Vn
            V = Vn
        return V

    def bellman_residual(self, V):
        """‖T* V − V‖∞ — the contraction gap; → 0 iff V = V*."""
        Vn, _ = self.bellman_opt_v(V)
        return float(np.max(np.abs(Vn - V)))
