"""CliffWalking — the canonical benchmark with a *known* analytic optimum.

Standard 4x12 grid. Start bottom-left, goal bottom-right, the bottom-row
interior cells are a cliff (episode ends, reward -100). With γ=1 the optimal
deterministic policy has value V(start) = -13 exactly. Gate G2 asserts this
to 4 decimals — a published, independent ground truth.
"""
from __future__ import annotations
import numpy as np
from ..core.env import Env, Step
from ..core.mdp import TabularMDP
from ..core.rng import RNG

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ROWS, COLS = 4, 12


class CliffWalking(Env):
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
        self.S = ROWS * COLS
        self.A = 4
        self._start = (ROWS - 1, 0)
        self._goal = (ROWS - 1, COLS - 1)
        self._rng = None

    @property
    def n_actions(self):
        return 4

    @property
    def n_states(self):
        return self.S

    def _idx(self, r, c):
        return r * COLS + c

    def _rc(self, s):
        return divmod(s, COLS)

    def _is_cliff(self, r, c):
        return r == ROWS - 1 and 1 <= c <= COLS - 2

    def reset(self, seed: int | None = None):
        self._rng = RNG(seed if seed is not None else 0)
        self._pos = self._start
        return self._idx(*self._pos)

    def step(self, action: int):
        r, c = self._pos
        dr, dc = _ACTIONS[action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            nr, nc = r, c
        self._pos = (nr, nc)
        s = self._idx(nr, nc)
        if self._is_cliff(nr, nc):
            return Step(obs=self._idx(*self._start), reward=-100.0, done=True,
                        info={"cliff": True})
        if (nr, nc) == self._goal:
            return Step(obs=s, reward=0.0, done=True, info={})
        return Step(obs=s, reward=-1.0, done=False, info={})

    def render(self):
        r, c = self._pos
        lines = []
        for rr in range(ROWS):
            row = []
            for cc in range(COLS):
                if (rr, cc) == (r, c):
                    row.append("@")
                elif (rr, cc) == self._goal:
                    row.append("G")
                elif self._is_cliff(rr, cc):
                    row.append("X")
                elif (rr, cc) == self._start:
                    row.append("S")
                else:
                    row.append(".")
            lines.append("".join(row))
        return "\n".join(lines)

    def tabular(self):
        P = np.zeros((self.S, self.A, self.S))
        R = np.zeros((self.S, self.A))
        terminals = [self._idx(*self._goal)]
        for s in range(self.S):
            r, c = self._rc(s)
            for a in range(self.A):
                dr, dc = _ACTIONS[a]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ROWS and 0 <= nc < COLS):
                    nr, nc = r, c
                ns = self._idx(nr, nc)
                if self._is_cliff(nr, nc):
                    P[s, a, self._idx(*self._start)] = 1.0
                    R[s, a] = -100.0
                else:
                    P[s, a, ns] = 1.0
                    R[s, a] = 0.0 if (nr, nc) == self._goal else -1.0
        return TabularMDP(P, R, gamma=self.gamma, terminals=terminals)
