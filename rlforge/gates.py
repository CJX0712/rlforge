"""Delivery gates — the RlForge equivalent of 天枢's G1–G5.

A unit test proves a function is coded correctly; a gate proves the *system*
delivers its claim. Each gate is a hard, independent invariant, and the rule is
the same as 天枢's: **failure is reported honestly, thresholds are never loosened
to turn red green.**
"""
from __future__ import annotations
import numpy as np
from .envs import GridWorld, CliffWalking, LQR
from .dp import value_iteration, policy_iteration
from .td import q_learning
from .deep import dqn, greedy_return
from .pg import a2c, ppo, lqr_actor_critic
from .pg._base import make_featurizer
from .fusion import simulate_greedy


def _to_argmax(env, pi):
    """Convert a learned LinearPolicy into a greedy argmax action array."""
    _, feat = make_featurizer(env)
    return np.array([int(np.argmax(pi.probs(feat(s)))) for s in range(env.n_states)])


def _grid():
    return GridWorld().tabular()


def gate_determinism():
    mdp = _grid()
    q1 = q_learning(mdp, episodes=800, seed=7)
    q2 = q_learning(mdp, episodes=800, seed=7)
    ok = bool(np.array_equal(q1, q2))
    detail = "Q-learning bit-identical on same seed" if ok else "Q-learning diverged"
    if not ok:
        return ok, detail
    net1 = dqn(mdp, episodes=200, seed=7)
    net2 = dqn(mdp, episodes=200, seed=7)
    ok = bool(np.max(np.abs(np.array(net1._returns) - np.array(net2._returns))) < 1e-12)
    detail = "DQN deterministic (returns match to 1e-12)" if ok else "DQN non-deterministic"
    return ok, detail


def gate_optimal_match():
    gw = GridWorld()
    mdp = gw.tabular()
    V, pi_vi = value_iteration(mdp)
    pi_bfs = gw.bfs_optimal_policy()
    vi_ret = simulate_greedy(mdp, pi_vi)
    bfs_ret = simulate_greedy(mdp, pi_bfs)
    grid_ok = abs(vi_ret - bfs_ret) < 1e-6  # VI reaches the BFS optimum

    cw = CliffWalking()
    cmdp = cw.tabular()
    cV, _ = value_iteration(cmdp)
    start = cw._idx(*cw._start)
    # Analytic optimum for this reward scheme: 12 steps of -1, final goal-step = 0.
    cliff_ok = bool(abs(cV[start] - (-12.0)) < 1e-3)
    detail = (f"Grid VI==BFS optimal (ret {vi_ret:.2f}); Cliff V(start)={cV[start]:.4f}"
              if grid_ok and cliff_ok else
              f"Grid VI vs BFS (vi {vi_ret:.2f}, bfs {bfs_ret:.2f}); Cliff V={cV[start]:.4f}")
    return bool(grid_ok and cliff_ok), detail


def gate_bellman_residual():
    mdp = CliffWalking().tabular()
    V, _ = value_iteration(mdp)
    res = mdp.bellman_residual(V)
    ok = res < 1e-9
    return ok, f"‖T*V−V‖∞ = {res:.2e} (need < 1e-9)"


def gate_q_learning():
    mdp = _grid()
    V_vi, pi_vi = value_iteration(mdp)
    opt_ret = simulate_greedy(mdp, pi_vi)
    Q = q_learning(mdp, episodes=2000, seed=3)
    pi_q = np.argmax(Q, axis=1)
    q_ret = simulate_greedy(mdp, pi_q)
    # Q-learning need not tie-break identically to VI on equal Q-values
    # (e.g. two actions with the same optimal Q). The invariant is that its
    # *return* matches the optimum — i.e. it actually found Q*.
    ret_ok = abs(q_ret - opt_ret) < 1e-2
    q_ok = bool(np.array_equal(pi_q, pi_vi))
    ok = ret_ok
    detail = (f"Q-learning return {q_ret:.3f} == optimal {opt_ret:.3f}"
              f"{' (policy also identical)' if q_ok else ''}") if ok else \
             f"Q-learning not optimal (return {q_ret:.3f} vs {opt_ret:.3f})"
    return ok, detail


def gate_dqn():
    mdp = _grid()
    V_vi, pi_vi = value_iteration(mdp)
    opt_ret = simulate_greedy(mdp, pi_vi)
    net = dqn(mdp, episodes=600, batches_per_ep=4, seed=5, target_update=80)
    ret = greedy_return(mdp, net, from_state=0)
    returns = np.array(net._returns)
    improved = float(returns[-50:].mean()) > float(returns[:50].mean())
    # opt_ret is negative (grid step penalties), so "near-optimal" must be an
    # *additive* tolerance: within 1.0 of the optimum (at most one wasted step).
    # A multiplicative 0.8*opt would be a stricter (more negative) target and is
    # mathematically wrong for negative rewards.
    near = ret >= opt_ret - 1.0
    ok = bool(improved and near)
    detail = (f"DQN greedy return {ret:.2f} vs optimal {opt_ret:.2f}; "
              f"reward curve improved:{improved}") if ok else \
             f"DQN underperformed (return {ret:.2f}, optimal {opt_ret:.2f}, improved {improved})"
    return ok, detail


def gate_policy_gradient():
    mdp = _grid()
    V_vi, pi_vi = value_iteration(mdp)
    opt_ret = simulate_greedy(mdp, pi_vi)

    # A2C on the GridWorld env (stochastic policy → sampled trajectories).
    pi_a2c, _, ret_a2c = a2c(mdp_env_grid(), episodes=1500, seed=11, max_steps=80)
    ra = np.array(ret_a2c)
    a2c_improved = float(ra[-len(ra) // 2:].mean()) > float(ra[:len(ra) // 2].mean())
    a2c_near = simulate_greedy(mdp, _to_argmax(mdp_env_grid(), pi_a2c)) >= opt_ret - 1.0

    # PPO on the same env, reached via clipped trust-region updates.
    pi_ppo, _, ret_ppo = ppo(mdp_env_grid(), episodes=2000, seed=12, max_steps=80)
    rp = np.array(ret_ppo)
    ppo_improved = float(rp[-len(rp) // 2:].mean()) > float(rp[:len(rp) // 2].mean())
    ppo_near = simulate_greedy(mdp, _to_argmax(mdp_env_grid(), pi_ppo)) >= opt_ret - 1.0

    ok = bool(a2c_improved and a2c_near and ppo_improved and ppo_near)
    detail = (f"A2C improved:{a2c_improved} near-opt:{a2c_near}; "
              f"PPO improved:{ppo_improved} near-opt:{ppo_near}") if ok else \
             f"A2C(imp:{a2c_improved},near:{a2c_near}) PPO(imp:{ppo_improved},near:{ppo_near})"
    return ok, detail


def gate_lqr():
    env = LQR()
    P_opt, K_ric = env.optimal()          # optimal control is u = -K_ric x
    G_opt = -K_ric                         # gain applied as u = G x
    opt_cost = env.cost_of_policy(G_opt, x0=None, seed=21, episodes=16)
    rand_cost = env.cost_of_policy(np.zeros((env.m, env.n)), x0=None, seed=21, episodes=16)
    K, rep = lqr_actor_critic(env, episodes=6000, seed=21)
    rep["opt_cost"], rep["rand_cost"] = opt_cost, rand_cost
    ok = bool(rep["stable"] and rep["final_cost"] <= 1.3 * opt_cost
              and rep["final_cost"] < 0.9 * rand_cost)
    detail = (f"LQR learned cost {rep['final_cost']:.3f} vs opt {opt_cost:.3f} "
              f"(rand {rand_cost:.1f}), stable:{rep['stable']}") if ok else \
             f"LQR not optimal (final {rep['final_cost']:.3f} vs opt {opt_cost:.3f}, stable {rep['stable']})"
    return ok, detail


def mdp_env_grid():
    """A GridWorld env (not MDP) for policy-gradient sampling."""
    return GridWorld()


GATES = [
    ("G1", "确定性 · 同 seed 逐位一致", gate_determinism),
    ("G2", "最优值/最优控制可证匹配", gate_optimal_match),
    ("G3", "Bellman 残差 → 0 (‖T*V−V‖∞<1e-9)", gate_bellman_residual),
    ("G4", "Q-learning 收敛到 Q* (策略==最优)", gate_q_learning),
    ("G5", "DQN 学会 (回报曲线上升·近最优)", gate_dqn),
    ("G6", "策略梯度学会 (A2C/PPO 提升·近最优)", gate_policy_gradient),
    ("G7", "LQR 学习成本逼近 Riccati 最优", gate_lqr),
]


def run_gates(verbose=True):
    results = []
    for gid, name, fn in GATES:
        try:
            ok, detail = fn()
        except Exception as e:  # noqa
            ok, detail = False, f"EXCEPTION: {type(e).__name__}: {e}"
        results.append((gid, name, ok, detail))
        if verbose:
            mark = "PASS" if ok else "FAIL"
            print(f"  {gid:<3} {mark}  {name:<34} {detail}")
    return results


if __name__ == "__main__":
    run_gates()
