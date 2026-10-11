"""LQR actor-critic — A2C on a *continuous* control problem.

The only RL problem with a closed-form optimum, so it is the perfect
proof-of-correctness target. A linear policy u = Kx with Gaussian exploration
has score ∇_K log π = σ⁻² (u − Kx) xᵀ. A *learned linear baseline* V(x) = wᵀx + b
cuts the gradient variance that makes plain REINFORCE diverge on the unstable
A-matrix. Gate G7 checks the learned gain drives the cost to within 1.3× of the
Riccati optimum — a mathematical proof, not a leaderboard proxy.

Note on sign: the Riccati gain Kᵣ solves the optimal control u* = −Kᵣx, so the
gain applied as u = Gx converges to G* = −Kᵣ (closed loop A + BG*).
"""
from __future__ import annotations
import numpy as np
from ..core.rng import RNG


def lqr_actor_critic(env, *, episodes=6000, sigma=0.3, lr=0.02,
                     lr_v=0.05, clip_norm=0.5, seed=0):
    rng = RNG(seed)
    n, m = env.n, env.m
    K = np.zeros((m, n))
    w = np.zeros(n + 1)  # linear+constant baseline V(x) = w[:n]·x + w[n]
    P_opt, K_ric = env.optimal()
    G_opt = -K_ric
    opt_cost = env.cost_of_policy(G_opt, seed=seed, episodes=16)
    rand_cost = env.cost_of_policy(np.zeros((m, n)), seed=seed, episodes=16)
    costs = []
    for ep in range(episodes):
        x = env.reset(seed=seed + ep)
        xs, us, rs = [], [], []
        for _ in range(env.horizon):
            u = K @ x + sigma * rng.standard_normal(m)
            xs.append(x.copy())
            us.append(u.copy())
            st = env.step(u)
            rs.append(st.reward)
            x = st.obs
        G = 0.0
        gs = []
        for r in reversed(rs):
            G += r
            gs.insert(0, G)
        gs = np.array(gs, float)
        xb = np.array([np.concatenate([xx, [1.0]]) for xx in xs])  # (T, n+1)
        vs = xb @ w
        adv = gs - vs
        # critic regression
        for t in range(env.horizon):
            err = gs[t] - w @ xb[t]
            w += lr_v * err * xb[t]
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        delta = np.zeros((m, n))
        for t in range(env.horizon):
            score = us[t] - K @ xs[t]          # (m,)
            delta += advn[t] * np.outer(score, xs[t]) / (sigma * sigma)
        delta /= env.horizon
        norm = np.linalg.norm(delta)
        if norm > clip_norm:
            delta *= clip_norm / norm
        K += lr * delta
        if ep % 500 == 0:
            costs.append(env.cost_of_policy(K, seed=seed, episodes=8))
    final_cost = env.cost_of_policy(K, seed=seed, episodes=32)
    return K, {
        "opt_cost": opt_cost,
        "rand_cost": rand_cost,
        "final_cost": final_cost,
        "costs": costs,
        "G_opt": G_opt,
        "stable": bool(np.all(np.abs(np.linalg.eigvals(env.A + env.B @ K)) < 1.0)),
    }
