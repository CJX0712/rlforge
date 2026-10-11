"""Q-learning (Watkins 1989) — off-policy TD control.

Learns Q* directly from any exploratory behaviour policy. On a deterministic
MDP, gate G4 asserts the greedy policy recovered from Q matches the optimal
policy and its value matches value iteration to 1e-3.
"""
from __future__ import annotations
from ._core import tabular_td


def q_learning(mdp, **kw):
    return tabular_td(mdp, method="q", **kw)
