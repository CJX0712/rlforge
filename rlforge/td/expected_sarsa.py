"""Expected SARSA (van Seijen et al. 2009) — on-policy, variance-reduced.

Replaces the single sampled next action with its *expectation* under the
behaviour policy, cutting bootstrap variance without changing the fixed point.
"""
from __future__ import annotations
from ._core import tabular_td


def expected_sarsa(mdp, **kw):
    return tabular_td(mdp, method="esarsa", **kw)
