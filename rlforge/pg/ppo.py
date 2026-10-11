"""PPO-Clip (Schulman et al. 2017) — trust-region-lite policy optimisation.

The clipped surrogate prevents the new policy from drifting too far from the
one that collected the data: ratio π/π_old is clamped to [1−ε, 1+ε], and the
gradient is switched off once it would push past the clamp. Gate G6 asserts
PPO both improves over the baseline and lands near the optimum — the same claim
as A2C, reached by a different, more sample-efficient route.
"""
from __future__ import annotations
import numpy as np
from ..core.rng import RNG
from ._base import LinearPolicy, make_featurizer


def ppo(env, *, episodes=800, epochs=4, mb=32, lr=0.02, gamma=0.99,
        clip=0.2, seed=0, max_steps=200):
    rng = RNG(seed)
    n_in, feat = make_featurizer(env)
    pi = LinearPolicy(n_in, env.n_actions, seed)
    V = LinearPolicy(n_in, 1, seed + 1)
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
            lp = pi.logp(x, a)
            st = env.step(a)
            traj.append((x, a, st.reward, lp))
            obs = st.obs
            done = st.done
            steps += 1
            if not done:
                x = feat(obs)
        G = 0.0
        gs = []
        for _, _, r, _ in reversed(traj):
            G = r + gamma * G
            gs.insert(0, G)
        gs = np.array(gs, float)
        xs = np.stack([t[0] for t in traj])
        acts = np.array([t[1] for t in traj])
        old_lp = np.array([t[3] for t in traj], float)
        vals = np.array([V.predict(xx)[0] for xx in xs], float)
        adv = gs - vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(traj)
        order = rng.permutation(n)
        for _ in range(epochs):
            rng.shuffle(order)
            for start in range(0, n, mb):
                for i in order[start:start + mb]:
                    xx = xs[i]
                    a = int(acts[i])
                    A = adv[i]
                    Gt = gs[i]
                    ratio = np.exp(pi.logp(xx, a) - old_lp[i])
                    gate = 1.0
                    if A > 0 and ratio > 1 + clip:
                        gate = 0.0
                    elif A < 0 and ratio < 1 - clip:
                        gate = 0.0
                    dW, db = pi.grad_logp(xx, a)
                    pi.W[0] += lr * A * gate * dW
                    pi.b[0] += lr * A * gate * db
                    err = V.predict(xx)[0] - Gt
                    V.W[0] -= lr * err * xx.reshape(1, -1)
                    V.b[0] -= lr * err
        returns.append(gs[0])
    return pi, V, returns
