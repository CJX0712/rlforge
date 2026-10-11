"""Linear-Quadratic Regulator — the one RL problem with a *closed-form* optimum.

A(s, a) = A x + B u,  cost = Σ (xᵀQx + uᵀRu).  The optimal gain K solves the
discrete Riccati equation; gate G2 checks the learned controller's cost
matches x₀ᵀ P x₀ to high precision — a math-proof, not a leaderboard proxy.
"""
from __future__ import annotations
import numpy as np
from ..core.env import Env, Step
from ..core.rng import RNG


def solve_lqr(A, B, Q, R, max_iter=1000, tol=1e-12):
    """Iterate the discrete Riccati recursion to its fixed point.

    P ← Q + AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA.  Returns (P, K). This *is* value
    iteration restricted to linear-quadratic control, so its convergence is
    guaranteed for stabilizable (A,B) with the chosen weights.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    Q = np.asarray(Q, float)
    R = np.asarray(R, float)
    n = A.shape[0]
    P = Q.copy()
    for _ in range(max_iter):
        BP = B.T @ P
        inner = R + BP @ B
        K = np.linalg.solve(inner, BP @ A)
        Pn = Q + A.T @ P @ A - (BP @ A).T @ K
        if np.max(np.abs(Pn - P)) < tol:
            P = Pn
            break
        P = Pn
    BP = B.T @ P
    K = np.linalg.solve(R + BP @ B, BP @ A)
    return P, K


def is_stable(A_cl, tol=1.0 - 1e-9):
    """True if every closed-loop eigenvalue lies strictly inside the unit disk."""
    return np.all(np.abs(np.linalg.eigvals(A_cl)) < tol)


class LQR(Env):
    def __init__(self, A=None, B=None, Q=None, R=None, horizon=40, gamma=1.0):
        if A is None:
            A = np.array([[1.0, 0.1], [0.0, 1.0]])
        if B is None:
            B = np.array([[0.0], [0.1]])
        if Q is None:
            Q = np.eye(A.shape[0])
        if R is None:
            R = np.eye(B.shape[1])
        self.A, self.B, self.Q, self.R = map(np.asarray, (A, B, Q, R))
        self.horizon = horizon
        self.gamma = gamma
        self.n, self.m = self.A.shape[0], self.B.shape[1]
        self._rng = RNG(0)
        self._x = np.zeros(self.n)
        self._t = 0

    @property
    def n_actions(self):
        return self.m

    @property
    def n_states(self):
        return None  # continuous

    def optimal(self):
        return solve_lqr(self.A, self.B, self.Q, self.R)

    def reset(self, seed: int | None = None):
        self._rng = RNG(seed if seed is not None else 0)
        self._x = self._rng.standard_normal(self.n) * 0.5
        self._t = 0
        return self._x.copy()

    def step(self, action):
        u = np.asarray(action, float).reshape(self.m)
        reward = float(-(self._x @ self.Q @ self._x + u @ self.R @ u))
        self._x = self.A @ self._x + self.B @ u
        self._t += 1
        done = self._t >= self.horizon
        return Step(obs=self._x.copy(), reward=reward, done=done, info={})

    def render(self):
        return f"x={np.round(self._x, 3)} t={self._t}/{self.horizon}"

    def cost_of_policy(self, K, x0=None, seed=1, episodes=32):
        """Average finite-horizon cost of u=Kx from x0 (or a seed-fixed batch)."""
        rng = RNG(seed)
        total = 0.0
        for _ in range(episodes):
            x = rng.standard_normal(self.n) * 0.5 if x0 is None else np.asarray(x0, float)
            c = 0.0
            for _ in range(self.horizon):
                u = K @ x
                c += float(x @ self.Q @ x + u @ self.R @ u)
                x = self.A @ x + self.B @ u
            total += c
        return total / episodes
