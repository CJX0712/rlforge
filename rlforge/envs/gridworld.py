"""GridWorld — a deterministic maze with a *provable* optimal policy.

Why this env matters: its optimal policy can be computed independently by
BFS (shortest path). Gate G2 checks that value iteration's greedy policy
matches BFS *exactly* — a cross-validation of the Bellman solver against
brute force, not against itself.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from ..core.env import Env, Step
from ..core.mdp import TabularMDP
from ..core.rng import RNG

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GridWorld(Env):
    def __init__(self, layout=None, gamma: float = 0.99):
        if layout is None:
            layout = [
                "S...#",
                ".##.#",
                ".#...",
                "...G.",
            ]
        self.grid = [list(r) for r in layout]
        self.H = len(self.grid)
        self.W = len(self.grid[0])
        self.gamma = gamma
        self.S = self.H * self.W
        self.A = 4
        self._start = self._find("S")
        self._goal = self._find("G")
        self._walls = {self._idx(r, c) for r in range(self.H) for c in range(self.W)
                       if self.grid[r][c] == "#"}
        self._rng = None

    def _find(self, ch):
        for r in range(self.H):
            for c in range(self.W):
                if self.grid[r][c] == ch:
                    return (r, c)
        return (0, 0)

    def _idx(self, r, c):
        return r * self.W + c

    def _rc(self, s):
        return divmod(s, self.W)

    @property
    def n_actions(self):
        return 4

    @property
    def n_states(self):
        return self.S

    def reset(self, seed: int | None = None):
        self._rng = RNG(seed if seed is not None else 0)
        self._pos = self._start
        return self._idx(*self._pos)

    def step(self, action: int):
        r, c = self._pos
        dr, dc = _ACTIONS[action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr][nc] != "#":
            r, c = nr, nc
        self._pos = (r, c)
        s = self._idx(r, c)
        reward = 0.0 if (r, c) == self._goal else -1.0
        done = (r, c) == self._goal
        return Step(obs=s, reward=reward, done=done, info={"pos": (r, c)})

    def render(self):
        r, c = self._pos
        g = [row[:] for row in self.grid]
        if (r, c) != self._goal:
            g[r][c] = "@"
        return "\n".join("".join(row) for row in g)

    # ---- provable optimal policy via BFS (independent of Bellman) ----------
    def bfs_optimal_policy(self):
        """Return π[s] = action that steps toward the goal along a shortest path.

        Computed independently of the Bellman solver: BFS gives the
        distance-to-goal of every cell, then each cell picks the neighbour with
        the smallest distance. Following that policy can never loop, so it is a
        ground-truth optimum used to cross-check value iteration.
        """
        goal = self._idx(*self._goal)
        dist = {goal: 0}
        q = deque([goal])
        while q:
            s = q.popleft()
            r, c = self._rc(s)
            for dr, dc in _ACTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr][nc] != "#":
                    ns = self._idx(nr, nc)
                    if ns not in dist:
                        dist[ns] = dist[s] + 1
                        q.append(ns)
        pi = np.zeros(self.S, dtype=int)
        for s in range(self.S):
            if s == goal:
                continue
            best_a, best_d = 0, None
            r, c = self._rc(s)
            for a, (dr, dc) in enumerate(_ACTIONS):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr][nc] != "#":
                    ns = self._idx(nr, nc)
                    d = dist.get(ns)
                    if d is not None and (best_d is None or d < best_d):
                        best_d, best_a = d, a
            if best_d is not None:
                pi[s] = best_a
        return pi

    def tabular(self):
        """Build the exact tabular MDP from the deterministic dynamics."""
        P = np.zeros((self.S, self.A, self.S))
        R = np.zeros((self.S, self.A))
        for s in range(self.S):
            r, c = self._rc(s)
            for a in range(self.A):
                dr, dc = _ACTIONS[a]
                nr, nc = r + dr, c + dc
                ns = s
                rew = -1.0
                if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr][nc] != "#":
                    ns = self._idx(nr, nc)
                    rew = 0.0 if (nr, nc) == self._goal else -1.0
                P[s, a, ns] = 1.0
                R[s, a] = rew
        return TabularMDP(P, R, gamma=self.gamma, terminals=[self._idx(*self._goal)])
