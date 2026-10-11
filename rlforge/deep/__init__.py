"""Deep RL (function approximation)."""
from ._net import MLP
from .dqn import dqn, greedy_return

__all__ = ["MLP", "dqn", "greedy_return"]
