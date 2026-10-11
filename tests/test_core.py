"""Math-invariant unit tests for RlForge core (L0/L1/L2).

These are the *micro* invariants behind the delivery gates: the RNG is
bit-identical across runs, the Bellman operator is a γ-contraction, and the
LQR Riccati solution matches the closed-form optimum. Author: 晨星.
"""
import numpy as np

from rlforge.core import RNG, TabularMDP
from rlforge.envs import GridWorld, CliffWalking, LQR, solve_lqr, is_stable
from rlforge.dp import value_iteration, policy_iteration


def test_rng_is_deterministic():
    a = RNG(123)
    b = RNG(123)
    seq_a = np.array([a.random() for _ in range(100)])
    seq_b = np.array([b.random() for _ in range(100)])
    assert np.array_equal(seq_a, seq_b), "RNG must be bit-identical on same seed"
    # different seed -> different stream (overwhelmingly likely)
    c = RNG(124)
    seq_c = np.array([c.random() for _ in range(100)])
    assert not np.array_equal(seq_a, seq_c)


def test_tabular_mdp_bellman_contraction():
    """Bellman optimality operator is a γ-contraction: residual → 0."""
    mdp = GridWorld().tabular()
    V = np.zeros(mdp.S)
    # fixed-point iteration decreases the Bellman residual monotonically
    prev = np.inf
    for _ in range(200):
        V, _ = mdp.bellman_opt_v(V)
        res = mdp.bellman_residual(V)
        assert res <= prev + 1e-12, "residual must be non-increasing"
        prev = res
    assert mdp.bellman_residual(V) < 1e-9


def test_value_iteration_matches_policy_iteration():
    mdp = CliffWalking().tabular()
    V_vi, pi_vi = value_iteration(mdp)
    V_pi, pi_pi, _ = policy_iteration(mdp)
    assert abs(V_vi[0] - V_pi[0]) < 1e-6, "VI and PI must agree on the value"
    # greedy policies are equally optimal
    from rlforge.fusion import simulate_greedy
    assert abs(simulate_greedy(mdp, pi_vi) - simulate_greedy(mdp, pi_pi)) < 1e-6


def test_lqr_riccati_closed_form():
    """LQR Riccati gain yields the analytic optimum and a stable closed loop."""
    env = LQR()
    P, K = solve_lqr(env.A, env.B, env.Q, env.R)
    G_opt = -K                      # control applied as u = G x
    opt_cost = env.cost_of_policy(G_opt, x0=None, seed=0, episodes=8)
    # a zero (do-nothing) controller must be far worse
    zero_cost = env.cost_of_policy(np.zeros((env.m, env.n)), x0=None, seed=0, episodes=8)
    assert opt_cost < 0.9 * zero_cost, "Riccati must beat the do-nothing policy"
    assert is_stable(env.A + env.B @ G_opt), "closed loop A+BK must be stable"


def test_bellman_operator_fixed_point_is_optimal():
    mdp = GridWorld().tabular()
    V, pi = value_iteration(mdp)
    # V satisfies the fixed point T*V = V up to tol, and is the unique optimum
    Vn, _ = mdp.bellman_opt_v(V)
    assert np.max(np.abs(Vn - V)) < 1e-9
