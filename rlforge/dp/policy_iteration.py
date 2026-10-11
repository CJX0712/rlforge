"""Policy Iteration — exact dynamic programming, the policy-improvement theorem.

Two facts make this verifiable: (1) every greedy improvement strictly raises
the value of at least one state until the policy is optimal (monotone); (2) it
terminates in ≤ |S| iterations because there are finitely many deterministic
policies. Gate G3 checks the iteration count is finite and the final policy's
value equals V*.
"""
from __future__ import annotations
import numpy as np


def policy_iteration(mdp, max_iter=1000, tol=1e-12):
    pi = np.zeros(mdp.S, dtype=int)
    V = mdp.eval_policy(pi, tol=tol)
    iters = 0
    for _ in range(max_iter):
        iters += 1
        _, pi_new = mdp.bellman_opt_v(V)  # greedy improvement
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new
        V = mdp.eval_policy(pi, tol=tol)
    return V, pi, iters
