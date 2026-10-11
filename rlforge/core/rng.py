"""Deterministic RNG — PCG64 behind a SeedSequence, the same backbone the
Forge family uses so every run is bit-identical given a seed."""
from __future__ import annotations
import numpy as np


class RNG:
    """Thin, seed-stable wrapper over ``numpy.random.Generator``.

    Why PCG64 + SeedSequence: independent seeds never collide into
    correlated streams, and the stream is reproducible across machines —
    a hard prerequisite for the "determinism gate" (G1)."""

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._g = np.random.default_rng(np.random.SeedSequence(self.seed))

    def spawn(self, n: int = 1) -> "list[RNG]":
        children = self._g.spawn(n)
        out = []
        for c in children:
            r = RNG.__new__(RNG)
            r.seed = self.seed
            r._g = np.random.Generator(c)
            out.append(r)
        return out

    def integers(self, low, high=None, size=None):
        return self._g.integers(low, high, size=size)

    def random(self, size=None):
        return self._g.random(size=size)

    def standard_normal(self, size=None):
        return self._g.standard_normal(size=size)

    def choice(self, a, size=None, p=None, replace=True):
        return self._g.choice(a, size=size, p=p, replace=replace)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._g.uniform(low, high, size=size)

    def permutation(self, n):
        return self._g.permutation(n)

    def shuffle(self, x):
        self._g.shuffle(x)
        return x
