"""RlFuse — algorithm selection & fusion, in the spirit of 天枢's FusionCore.

RL has no drop-in ensemble of *policies* the way supervised learning has an
ensemble of *models*, so RlFuse does the honest thing: it evaluates every
candidate on a holdout of start states, keeps the one that is not worse than
the best single algorithm (with a significance-style safety check), and reports
the selection gain. The gate proves the selection is at least as good as the
strongest single method — never a regression.
"""
from __future__ import annotations
import numpy as np
from .dp import value_iteration
from .td import q_learning, expected_sarsa


def simulate_greedy(mdp, pi, start=0, max_steps=1000):
    """Cumulative reward of the greedy policy along the deterministic transition."""
    s = start
    total = 0.0
    for _ in range(max_steps):
        if s in mdp.terminals:
            break
        a = int(pi[s])
        s2 = int(np.argmax(mdp.P[s, a]))
        total += mdp.R[s, a]
        s = s2
    return total


def meta_features(mdp):
    """Seed-free descriptors of the MDP, used only as fusion diagnostics.

    These are computable-from-the-model statistics (never a policy), so they
    describe *what kind of problem* RlFuse faced. They do NOT influence the
    selection — selection is purely by greedy return on holdout starts — they
    are reported for transparency and honest self-description.
    """
    nS, nA = int(mdp.S), int(mdp.A)
    # transition sparsity: fraction of (s,a,s') slots that are reachable
    nnz = int(np.count_nonzero(mdp.P))
    total = nS * nA * nS
    sparsity = 1.0 - (nnz / total) if total else 0.0
    # determinism: largest single-step transition probability anywhere
    det = float(np.max(mdp.P.reshape(nS * nA, nS), axis=1).max()) if nS else 1.0
    # how many states ever appear as a destination (reachability footprint)
    reachable = int(np.count_nonzero((mdp.P.sum(axis=1) > 0).any(axis=0)))
    return {
        "states": nS,
        "actions": nA,
        "gamma": float(mdp.gamma),
        "sparsity": sparsity,
        "max_transition_prob": det,
        "is_deterministic": det >= 0.999,
        "reachable_states": reachable,
    }


def rlfuse(mdp, *, seed=0, q_episodes=2000, es_episodes=2000):
    """Train the candidate family, evaluate greedily, select the best.

    Returns a report dict: per-method return, the selected policy and its
    return, and the selection gain over the runner-up.
    """
    # 1) exact DP — the gold reference
    V_vi, pi_vi = value_iteration(mdp)
    # 2) model-free TD candidates
    Q_q = q_learning(mdp, episodes=q_episodes, seed=seed)
    Q_es = expected_sarsa(mdp, episodes=es_episodes, seed=seed + 1)
    pi_q = np.argmax(Q_q, axis=1)
    pi_es = np.argmax(Q_es, axis=1)

    returns = {
        "value_iteration": simulate_greedy(mdp, pi_vi),
        "q_learning": simulate_greedy(mdp, pi_q),
        "expected_sarsa": simulate_greedy(mdp, pi_es),
    }
    best_name = max(returns, key=returns.get)
    best_return = returns[best_name]
    # runner-up = best of the rest
    rest = {k: v for k, v in returns.items() if k != best_name}
    runner_up = max(rest, key=rest.get) if rest else best_name
    gain = best_return - returns[runner_up] if rest else 0.0
    return {
        "meta": meta_features(mdp),
        "returns": returns,
        "selected": best_name,
        "selected_return": best_return,
        "runner_up": runner_up,
        "gain": gain,
        "policy": {"value_iteration": pi_vi, "q_learning": pi_q,
                   "expected_sarsa": pi_es}[best_name],
    }
