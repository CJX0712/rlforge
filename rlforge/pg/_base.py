"""Policy-gradient base: a linear softmax policy + environment feature adapter.

Kept deliberately simple (one linear layer) so the gradient maths is exactly
the textbook REINFORCE / actor-critic update — verifiable by hand, no hidden
autodiff. Deeper policies in RlForge go through the MLP in ``deep``.
"""
from __future__ import annotations
import numpy as np
from ..core.rng import RNG


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def make_featurizer(env):
    """Return (state_dim, featurize) for grid (one-hot) or continuous (raw)."""
    if getattr(env, "n_states", None):
        n = env.n_states

        def f(s):
            x = np.zeros(n)
            x[int(s)] = 1.0
            return x

        return n, f
    n = len(np.asarray(env.reset(seed=0), float))

    def f(s):
        return np.asarray(s, float)

    return n, f


class LinearPolicy:
    def __init__(self, n_in, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.standard_normal((n_out, n_in)) * 0.01]
        self.b = [np.zeros(n_out)]
        self.n_in, self.n_out = n_in, n_out

    def logits(self, x):
        return x @ self.W[0].T + self.b[0]

    def probs(self, x):
        return softmax(self.logits(x))

    def predict(self, x):  # scalar-value form (n_out == 1)
        return self.logits(x)

    def logp(self, x, a):
        p = self.probs(x)
        return float(np.log(p[a] + 1e-12))

    def grad_logp(self, x, a):
        p = self.probs(x)
        d = np.zeros(self.n_out)
        d[a] = 1.0
        d -= p
        return np.outer(d, x), d

    def copy(self):
        c = LinearPolicy(self.n_in, self.n_out)
        c.W = [w.copy() for w in self.W]
        c.b = [b.copy() for b in self.b]
        return c
