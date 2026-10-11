"""REINFORCE (Williams 1992) — the original policy-gradient theorem in action.

Gradient = E[ ∇ log π(a|s) · (return − baseline) ]. The baseline cuts variance
without adding bias, so the *mean* episode return must climb monotonically in
expectation; gate G6 asserts the second half of training beats the first.
"""
from __future__ import annotations
import numpy as np
from ..core.rng import RNG
from ._base import LinearPolicy, make_featurizer


def reinforce(env, *, episodes=2000, lr=0.03, gamma=0.99, seed=0, max_steps=200):
    rng = RNG(seed)
    n_in, feat = make_featurizer(env)
    pi = LinearPolicy(n_in, env.n_actions, seed)
    returns = []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        x = feat(obs)
        traj = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            p = pi.probs(x).flatten()
            a = int(rng.choice(env.n_actions, p=p))
            st = env.step(a)
            traj.append((x, a, st.reward))
            obs = st.obs
            done = st.done
            steps += 1
            if not done:
                x = feat(obs)
        G = 0.0
        gs = []
        for _, _, r in reversed(traj):
            G = r + gamma * G
            gs.insert(0, G)
        gs = np.array(gs)
        base = gs.mean()
        for (xx, a, _), Gt in zip(traj, gs):
            dW, db = pi.grad_logp(xx, a)
            pi.W[0] += lr * (Gt - base) * dW
            pi.b[0] += lr * (Gt - base) * db
        returns.append(gs[0])
    return pi, returns
