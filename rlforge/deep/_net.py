"""A tiny, dependency-free MLP with hand-written forward/backward.

RlForge reuses NumPy — not PyTorch — so the whole stack stays offline,
CPU-only and bit-reproducible. The maths is the textbook chain rule; keeping
it explicit makes the gradient checks in the test-suite trustworthy.
"""
from __future__ import annotations
import numpy as np


class MLP:
    def __init__(self, dims, seed: int = 0):
        self.dims = list(dims)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            std = np.sqrt(2.0 / dims[i])  # He init
            self.W.append(rng.standard_normal((dims[i + 1], dims[i])) * std)
            self.b.append(np.zeros(dims[i + 1]))
        self._cache = None

    def forward(self, X):
        if X.ndim == 1:
            X = X[None, :]
        acts = [X]
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i].T + self.b[i]
            a = np.maximum(0.0, z) if i < len(self.W) - 1 else z
            acts.append((z, a))
        self._cache = acts
        return a

    def backward(self, dout):
        acts = self._cache
        gW, gb = [], []
        da = dout
        for i in reversed(range(len(self.W))):
            z, a = acts[i + 1]
            dz = da * (z > 0).astype(float) if i < len(self.W) - 1 else da
            # input to layer i is the post-activation of the previous layer
            # (stored as acts[i][1]), except for i==0 where it is the raw input.
            a_prev = acts[i][1] if i >= 1 else acts[i]
            gW.insert(0, dz.T @ a_prev)
            gb.insert(0, dz.sum(0))
            da = dz @ self.W[i]
        return {"W": gW, "b": gb}

    def sgd(self, grads, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * grads["W"][i]
            self.b[i] -= lr * grads["b"][i]

    def copy(self):
        n = MLP(self.dims)
        n.W = [w.copy() for w in self.W]
        n.b = [b.copy() for b in self.b]
        return n
