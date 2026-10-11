"""DQN (Mnih et al. 2015) — deep Q-learning with experience replay + target net.

Two ideas make DQN learn stably where plain Q-learning with a neural net
diverges: (1) a *replay buffer* breaks the temporal correlation of online
samples; (2) a *slowly-updated target network* keeps the bootstrap target
stationary. Gate G5 asserts the episode return climbs and the final greedy
policy is near-optimal.
"""
from __future__ import annotations
import numpy as np
from ..core.rng import RNG
from ._net import MLP


def _onehot(s, n):
    x = np.zeros(n)
    x[s] = 1.0
    return x


def dqn(mdp, *, hidden=32, episodes=700, batches_per_ep=4, batch=64,
        buffer_size=50000, lr=0.02, gamma=None, target_update=100,
        seed=0, eps0=0.3, eps_min=0.05):
    gamma = mdp.gamma if gamma is None else gamma
    rng = RNG(seed)
    net = MLP([mdp.S, hidden, mdp.A], seed=seed)
    target = net.copy()
    buf = {"s": [], "a": [], "r": [], "s2": [], "d": []}
    returns = []
    nS = mdp.S

    def act(s, eps):
        if rng.random() < eps:
            return int(rng.integers(mdp.A))
        return int(np.argmax(net.forward(_onehot(s, nS)).flatten()))

    for ep in range(episodes):
        eps = max(eps_min, eps0 * (1.0 - ep / episodes))
        s = int(rng.integers(nS))
        if s in mdp.terminals:
            continue
        G = 0.0
        while s not in mdp.terminals:
            a = act(s, eps)
            s2 = int(rng.choice(nS, p=mdp.P[s, a]))
            r = mdp.R[s, a]
            d = s2 in mdp.terminals
            for k, v in (("s", s), ("a", a), ("r", r), ("s2", s2), ("d", d)):
                buf[k].append(v)
            if len(buf["s"]) > buffer_size:
                for k in buf:
                    buf[k].pop(0)
            G += r
            s = s2
        returns.append(G)
        if len(buf["s"]) < batch:
            continue
        for _ in range(batches_per_ep):
            idx = rng.choice(len(buf["s"]), size=batch, replace=False)
            X = np.stack([_onehot(buf["s"][i], nS) for i in idx])
            A = np.array([buf["a"][i] for i in idx])
            R = np.array([buf["r"][i] for i in idx], float)
            S2 = np.stack([_onehot(buf["s2"][i], nS) for i in idx])
            D = np.array([buf["d"][i] for i in idx], float)
            qp = net.forward(X)
            qa = qp[np.arange(batch), A]
            max_next = np.max(target.forward(S2), axis=1)
            y = R + gamma * max_next * (1.0 - D)
            dout = np.zeros((batch, mdp.A))
            dout[np.arange(batch), A] = (qa - y) / batch
            net.sgd(net.backward(dout), lr)
        if ep % target_update == 0:
            target = net.copy()

    net._returns = returns
    return net


def greedy_return(mdp, net, from_state=0, max_steps=200):
    """Evaluate the greedy policy on the *deterministic* transition (mode of P)."""
    nS = mdp.S
    s = from_state
    total = 0.0
    for _ in range(max_steps):
        if s in mdp.terminals:
            break
        a = int(np.argmax(net.forward(_onehot(s, nS)).flatten()))
        s2 = int(np.argmax(mdp.P[s, a]))  # deterministic next state
        total += mdp.R[s, a]
        s = s2
    return total
