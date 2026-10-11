"""Environment contract.

Every environment in RlForge obeys one rule: ``reset`` then ``step`` must be
deterministic when a seed is fixed. That single property is what lets the
invariant gates prove an algorithm *actually* solved the problem instead of
getting lucky.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Step:
    obs: Any
    reward: float
    done: bool
    info: dict


class Env(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None) -> Any:
        """Return the initial observation. Seed fixes the trajectory RNG."""

    @abstractmethod
    def step(self, action: int) -> Step:
        """Advance one step. Returns (obs, reward, done, info)."""

    @property
    @abstractmethod
    def n_actions(self) -> int:
        ...

    @property
    def n_states(self) -> int | None:
        """Number of discrete states, or None for continuous / large spaces."""

    @abstractmethod
    def render(self) -> str:
        ...
