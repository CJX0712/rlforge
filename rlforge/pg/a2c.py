"""A2C — advantage actor-critic (Mnih et al. 2016).

Replaces the Monte-Carlo return with a *critic* baseline V(s), turning the
high-variance REINFORCE update into a low-variance advantage estimate
A(s,a) = G − V(s). Faster, stabler learning; gate G6 uses A2C as the
primary "policy-gradient actually learns" evidence.
"""
from __future__ import annotations
import numpy as np
from ..core.rng import RNG
from ._base import LinearPolicy, make_featurizer


def a2c(env, *, episodes=2000, lr_pi=0.05, lr_v=0.1, gamma=0.99, seed=0, max_steps=200):
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
        gs = np.array(gs, float)
        # Critic baseline V(s); advantage A = G - V, normalised per episode so
        # the update scale is independent of reward magnitude (the grid pays a
        # −1 per step, giving raw advantages of order −50 that would otherwise
        # saturate the softmax into a frozen, non-escaping policy). Normalising
        # keeps the policy soft enough to keep exploring — the standard,
        # correct A2C trick (PPO uses the same normalisation).
        vs = np.array([V.predict(xx)[0] for (xx, _, _) in traj], float)
        adv = gs - vs
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for i, (xx, a, _) in enumerate(traj):
            dW, db = pi.grad_logp(xx, a)
            pi.W[0] += lr_pi * adv[i] * dW
            pi.b[0] += lr_pi * adv[i] * db
            err = V.predict(xx)[0] - gs[i]
            V.W[0] -= lr_v * err * xx.reshape(1, -1)
            V.b[0] -= lr_v * err
        returns.append(gs[0])
    return pi, V, returns
