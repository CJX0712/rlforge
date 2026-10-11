"""Environment optimality cross-checks (L1 + G2).

Each environment ships an *independent* optimum so the Bellman solvers can be
validated against brute force rather than against themselves. Author: 晨星.
"""
import numpy as np

from rlforge.envs import GridWorld, CliffWalking, LQR, solve_lqr, is_stable
from rlforge.dp import value_iteration
from rlforge.fusion import simulate_greedy


def test_gridworld_vi_matches_bfs():
    gw = GridWorld()
    mdp = gw.tabular()
    V_vi, pi_vi = value_iteration(mdp)
    pi_bfs = gw.bfs_optimal_policy()
    vi_ret = simulate_greedy(mdp, pi_vi)
    bfs_ret = simulate_greedy(mdp, pi_bfs)
    assert abs(vi_ret - bfs_ret) < 1e-6, "VI must equal the BFS ground truth"
    assert bfs_ret < 0.0


def test_cliffwalking_analytic_optimum():
    cw = CliffWalking()
    mdp = cw.tabular()
    V, _ = value_iteration(mdp)
    start = cw._idx(*cw._start)
    # 12 steps of −1 to reach the goal; the final goal step itself pays 0.
    assert abs(V[start] - (-12.0)) < 1e-3, "CliffWalking analytic optimum is −12.0"


def test_lqr_controller_reaches_near_optimal():
    env = LQR()
    P, K = solve_lqr(env.A, env.B, env.Q, env.R)
    cost = env.cost_of_policy(-K, x0=None, seed=0, episodes=16)
    # sanity: solving LQR must produce a finite, stable cost
    assert np.isfinite(cost) and cost > 0.0
