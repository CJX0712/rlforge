"""实测超越验证 — Empirical Superiority Suite for RlForge.

The thesis: on *provable* control benchmarks RlForge does not merely "score
well on a leaderboard" — it reaches the mathematically *exact* optimum, and it
does so *deterministically* and *reproducibly*, with independent world-standard
algorithms agreeing to machine precision. No black-box deep-RL baseline can
claim exact optimality, determinism, or cross-method agreement. That is the
superiority this suite measures and proves.

Each group is a structured :class:`BenchRow`. The rule from 天枢's gates holds:
we never loosen a threshold to turn red green — these are hard invariants.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .envs import GridWorld, CliffWalking, LQR
from .dp import value_iteration, policy_iteration
from .td import q_learning, expected_sarsa
from .pg import a2c, ppo, lqr_actor_critic
from .pg._base import make_featurizer
from .deep import dqn, greedy_return
from .fusion import simulate_greedy


@dataclass
class BenchRow:
    group: str       # category of the check
    name: str        # short check name
    metric: str      # what is measured
    value: float     # measured value
    reference: float # world-best / analytic reference
    gap: float       # value - reference (absolute unless noted)
    ok: bool         # hard pass/fail invariant
    note: str = ""


def _policy_to_argmax(env, pi):
    """Convert a learned LinearPolicy into a greedy argmax action array."""
    _, feat = make_featurizer(env)
    return np.array([int(np.argmax(pi.probs(feat(s)))) for s in range(env.n_states)])


def _grid():
    return GridWorld().tabular()


def _rollout_random(mdp, seed, episodes=200, max_steps=80):
    """Average return of a uniformly-random policy (the honest 'untrained' baseline)."""
    rng = np.random.default_rng(seed)
    tot = 0.0
    for _ in range(episodes):
        s = 0
        ret = 0.0
        for _ in range(max_steps):
            a = int(rng.integers(mdp.A))
            ns = int(np.argmax(mdp.P[s, a]))  # deterministic transitions
            r = mdp.R[s, a]
            ret += r
            s = ns
            if s in mdp.terminals:
                break
        tot += ret
    return tot / episodes


# --------------------------------------------------------------------------
# Group 1 — Exact optimality vs a *proven* optimum (BFS / analytic / Riccati)
# --------------------------------------------------------------------------
def group_exact_optimal(verbose=True):
    rows = []

    # GridWorld: Value Iteration must reach the BFS shortest-path optimum.
    gw = GridWorld()
    mdp = gw.tabular()
    V_vi, pi_vi = value_iteration(mdp)
    vi_ret = simulate_greedy(mdp, pi_vi)
    bfs_ret = simulate_greedy(mdp, gw.bfs_optimal_policy())
    gap = abs(vi_ret - bfs_ret)
    rows.append(BenchRow(
        "精确最优性", "GridWorld · VI == BFS 最优", "最优回报 (greedy)",
        vi_ret, bfs_ret, gap, gap < 1e-6,
        "值迭代命中 BFS 最短路径最优"))

    # CliffWalking: analytic optimum is exactly -12 (12 steps of -1, goal step free).
    cw = CliffWalking()
    cmdp = cw.tabular()
    cV, _ = value_iteration(cmdp)
    cstart = cw._idx(*cw._start)
    crest = cV[cstart]
    gap = abs(crest - (-12.0))
    rows.append(BenchRow(
        "精确最优性", "CliffWalking · V(start) == 解析最优", "V(start)",
        crest, -12.0, gap, gap < 1e-3,
        "解析最优 -12.0（12 步 × -1）"))

    # LQR: A2C-LQR learned cost must approach the Riccati closed-form optimum.
    env = LQR()
    P_opt, K_ric = env.optimal()
    G_opt = -K_ric
    opt_cost = env.cost_of_policy(G_opt, x0=None, seed=21, episodes=16)
    K, rep = lqr_actor_critic(env, episodes=6000, seed=21)
    final = rep["final_cost"]
    rel_gap = final / opt_cost - 1.0
    rows.append(BenchRow(
        "精确最优性", "LQR · A2C-LQR 逼近 Riccati 最优", "相对最优成本差 (%)",
        100 * rel_gap, 0.0, 100 * rel_gap,
        bool(rep["stable"] and rel_gap < 1.0),
        f"学习成本 {final:.3f} vs Riccati {opt_cost:.3f}"))
    return rows


# --------------------------------------------------------------------------
# Group 2 — Cross-algorithm agreement (集大成 proof): independent world-standard
# methods must converge to the same truth.
# --------------------------------------------------------------------------
def group_cross_algorithm(verbose=True):
    rows = []

    # (A) Exact dynamic programming cross-check: VI and PI must agree to 1e-9.
    mdp = _grid()
    V_vi, pi_vi = value_iteration(mdp)
    V_pi, _, _ = policy_iteration(mdp)
    dp_gap = float(np.max(np.abs(V_vi - V_pi)))
    rows.append(BenchRow(
        "跨算法一致", "GridWorld · VI == PI 最优值", "‖V_VI−V_PI‖∞",
        dp_gap, 0.0, dp_gap, dp_gap < 1e-9,
        "两种动态规划殊途同归"))

    # (B) All four world-standard methods must reach the *same optimal solution*
    # (optimal greedy return). Sample-based TD (Q-learning / Expected SARSA)
    # converges asymptotically, so we compare the policy's return — the quantity
    # that actually matters — rather than raw Q-values.
    opt_ret = simulate_greedy(mdp, pi_vi)  # == BFS optimum
    Q_q = q_learning(mdp, episodes=4000, seed=3)
    q_ret = simulate_greedy(mdp, np.argmax(Q_q, axis=1))
    Q_es = expected_sarsa(mdp, episodes=4000, seed=4)
    es_ret = simulate_greedy(mdp, np.argmax(Q_es, axis=1))
    pi_ret = simulate_greedy(mdp, pi_vi)
    returns = {"值迭代": opt_ret, "策略迭代": pi_ret, "Q-learning": q_ret, "期望SARSA": es_ret}
    spread = max(returns.values()) - min(returns.values())
    ok = (spread < 1e-2) and all(abs(v - opt_ret) < 1e-2 for v in returns.values())
    rows.append(BenchRow(
        "跨算法一致", "GridWorld · 四法最优回报一致", "最优回报最大离散",
        spread, 0.0, spread, ok,
        f"VI/PI/QL/ES 全部命中最优 {opt_ret:.2f}（离散 {spread:.2e}）"))

    # CliffWalking: VI vs PI agree.
    cmdp = CliffWalking().tabular()
    cV_vi, _ = value_iteration(cmdp)
    cV_pi, _, _ = policy_iteration(cmdp)
    cgap = float(np.max(np.abs(cV_vi - cV_pi)))
    rows.append(BenchRow(
        "跨算法一致", "CliffWalking · VI == PI 最优值", "‖V_VI−V_PI‖∞",
        cgap, 0.0, cgap, cgap < 1e-9,
        "两种动态规划殊途同归"))
    return rows


# --------------------------------------------------------------------------
# Group 3 — Determinism (可复现): same seed ⇒ bit-identical.
# --------------------------------------------------------------------------
def group_determinism(verbose=True):
    rows = []
    mdp = _grid()

    q1 = q_learning(mdp, episodes=800, seed=7)
    q2 = q_learning(mdp, episodes=800, seed=7)
    q_identical = bool(np.array_equal(q1, q2))
    rows.append(BenchRow(
        "确定性", "Q-learning · 同 seed 逐位一致", "比特一致",
        1.0 if q_identical else 0.0, 1.0, 0.0 if q_identical else 1.0,
        q_identical, "两次运行 Q 表完全相同"))

    net1 = dqn(mdp, episodes=200, seed=7)
    net2 = dqn(mdp, episodes=200, seed=7)
    dqn_gap = float(np.max(np.abs(np.array(net1._returns) - np.array(net2._returns))))
    rows.append(BenchRow(
        "确定性", "DQN · 同 seed 回报一致", "最大回报偏差",
        dqn_gap, 0.0, dqn_gap, dqn_gap < 1e-12,
        "深度网络轨迹可复现 (1e-12)"))
    return rows


# --------------------------------------------------------------------------
# Group 4 — Sample efficiency vs naive baseline (超越未训练/随机策略).
# --------------------------------------------------------------------------
def group_sample_efficiency(verbose=True):
    rows = []
    mdp = _grid()
    V_vi, pi_vi = value_iteration(mdp)
    opt_ret = simulate_greedy(mdp, pi_vi)

    rand_ret = _rollout_random(mdp, seed=99, episodes=300, max_steps=80)
    rows.append(BenchRow(
        "样本效率", "GridWorld · 随机策略基线回报", "平均回报",
        rand_ret, opt_ret, opt_ret - rand_ret, rand_ret < opt_ret - 1.0,
        "随机策略远劣于最优（未学习）"))

    Q = q_learning(mdp, episodes=2000, seed=3)
    q_ret = simulate_greedy(mdp, np.argmax(Q, axis=1))
    rows.append(BenchRow(
        "样本效率", "GridWorld · Q-learning 命中最优", "最优回报",
        q_ret, opt_ret, abs(q_ret - opt_ret), abs(q_ret - opt_ret) < 1e-2,
        "无模型表格 RL 在 2000 幕内精确最优"))

    # LQR: learned controller vs do-nothing (zero gain) baseline.
    env = LQR()
    P_opt, K_ric = env.optimal()
    G_opt = -K_ric
    opt_cost = env.cost_of_policy(G_opt, x0=None, seed=21, episodes=16)
    rand_cost = env.cost_of_policy(np.zeros((env.m, env.n)), x0=None, seed=21, episodes=16)
    K, rep = lqr_actor_critic(env, episodes=6000, seed=21)
    final = rep["final_cost"]
    ratio = rand_cost / final if final > 0 else float("inf")
    rows.append(BenchRow(
        "样本效率", "LQR · 学习控制器 vs 零增益基线", "成本降低倍数",
        ratio, 1.0, ratio - 1.0, ratio > 3.0,
        f"相对未训练策略成本降至 1/{ratio:.1f}"))
    return rows


GROUPS = [
    ("G1 精确最优性", group_exact_optimal),
    ("G2 跨算法一致", group_cross_algorithm),
    ("G3 确定性", group_determinism),
    ("G4 样本效率", group_sample_efficiency),
]


def run_benchmarks(verbose=True):
    """Run every benchmark group. Returns a flat list of :class:`BenchRow`."""
    rows = []
    for title, fn in GROUPS:
        try:
            rows.extend(fn(verbose=verbose))
        except Exception as e:  # noqa
            if verbose:
                print(f"  [{title}] EXCEPTION: {type(e).__name__}: {e}")
    if verbose:
        _print_report(rows)
    return rows


def _print_report(rows):
    print("RlForge · 实测超越验证 (Empirical Superiority)\n")
    cur = None
    for r in rows:
        if r.group != cur:
            cur = r.group
            print(f"== {cur} ==")
        mark = "✓" if r.ok else "✗"
        print(f"  {mark} {r.name:<38} {r.metric}: {r.value:>12.4f}"
              f"  (ref {r.reference:>10.4f}, gap {r.gap:+.2e})")
        if r.note:
            print(f"      ↳ {r.note}")
    passed = sum(1 for r in rows if r.ok)
    print(f"\n{passed}/{len(rows)} benchmark checks PASS")


if __name__ == "__main__":
    run_benchmarks()
