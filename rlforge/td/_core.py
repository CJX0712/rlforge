"""Tabular temporal-difference control — shared engine.

Q-learning (off-policy), SARSA (on-policy), and Expected SARSA all share the
same loop; they differ only in how the bootstrap target at the next state is
formed. Reusing one engine is the "集大成" discipline — one proven core, three
well-known special cases, no duplicated bugs.
"""
from __future__ import annotations
import numpy as np
from ..core.rng import RNG


def tabular_td(mdp, method="q", *, episodes=3000, alpha=0.1, eps0=0.3,
               eps_min=0.02, gamma=None, seed=0):
    gamma = mdp.gamma if gamma is None else gamma
    Q = np.zeros((mdp.S, mdp.A))
    rng = RNG(seed)
    term = mdp.terminals
    for ep in range(episodes):
        eps = max(eps_min, eps0 * (1.0 - ep / episodes))
        s = rng.integers(mdp.S)
        if s in term:
            continue
        while s not in term:
            a = rng.integers(mdp.A) if rng.random() < eps else int(np.argmax(Q[s]))
            s2 = int(rng.choice(mdp.S, p=mdp.P[s, a]))
            r = mdp.R[s, a]
            if s2 in term:
                target = r
            elif method == "q":
                target = r + gamma * np.max(Q[s2])
            elif method == "sarsa":
                a2 = int(np.argmax(Q[s2])) if rng.random() >= eps else rng.integers(mdp.A)
                target = r + gamma * Q[s2, a2]
            elif method == "esarsa":
                probs = np.full(mdp.A, eps / mdp.A)
                probs[int(np.argmax(Q[s2]))] += 1.0 - eps
                target = r + gamma * float(probs @ Q[s2])
            else:
                raise ValueError(method)
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
    return Q
