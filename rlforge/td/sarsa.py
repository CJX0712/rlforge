"""SARSA (Rummery & Niranjan 1994) — on-policy TD control.

Uses the *actual* next action (sampled under the same ε-greedy policy) in the
bootstrap, so it learns the value of the exploring policy — safer near cliffs.
"""
from __future__ import annotations
from ._core import tabular_td


def sarsa(mdp, **kw):
    return tabular_td(mdp, method="sarsa", **kw)
