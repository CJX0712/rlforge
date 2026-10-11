"""RlForge — world-class reinforcement learning, pure NumPy, verifiable.

Ten-layer lineage (mirrors 天枢): L0 core · L1 envs · L2 dynamic programming ·
L3 tabular TD · L4 deep (DQN) · L5 policy gradients · L6 fusion (RlFuse) ·
L7 gates · L8 CLI · L9 reproducibility · L10 this package.

Author: 晨星 (CJX0712). MIT. CPU-only, no GPU / network / API key required.
"""
from .core import RNG, Env, Step, TabularMDP
from .envs import GridWorld, CliffWalking, LQR, solve_lqr, is_stable
from .dp import value_iteration, policy_iteration
from .td import q_learning, sarsa, expected_sarsa
from .deep import MLP, dqn, greedy_return
from .pg import reinforce, a2c, ppo, lqr_actor_critic
from .fusion import rlfuse, simulate_greedy

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = [
    "RNG", "Env", "Step", "TabularMDP",
    "GridWorld", "CliffWalking", "LQR", "solve_lqr", "is_stable",
    "value_iteration", "policy_iteration",
    "q_learning", "sarsa", "expected_sarsa",
    "MLP", "dqn", "greedy_return",
    "reinforce", "a2c", "ppo", "lqr_actor_critic",
    "rlfuse", "simulate_greedy",
]
