# RlForge · 强化学习锻造厂

> 世界顶级强化学习系统 —— 纯 NumPy、CPU-only、确定性、可验证。
> 天枢（tianshu-ai）十层架构谱系的新成员，补全强化学习这一 AI 第三范式。
> **作者：晨星（GitHub: CJX0712）** · MIT License

RlForge 不是又一个"玩具 RL 库"。它的每个算法都附带一条**可被独立实现的硬不变量**——
解析解 vs 数值解、暴力枚举 vs 贪心、参照实现 vs 主实现、Bellman 残差 → 0、Riccati 闭式最优。
所有不变量由 7 道**交付门禁（G1–G7）**强制，门禁失败即诚实报红，阈值永不放宽。

---

## 安装

```bash
pip install -e ".[test]"      # 含 pytest
rlforge gates                  # 跑全部 7 道交付门禁
pytest -q                      # 跑单元测试
```

仅依赖 `numpy>=1.24`，无需 GPU / 网络 / API key。

---

## 十层架构（对齐天枢）

| 层 | 模块 | 内容 |
|----|------|------|
| L0 | `core` | RNG（PCG64 种子序列·逐位可复现）、Env/Step、TabularMDP、Bellman 算子、metrics |
| L1 | `envs` | GridWorld（BFS 可证最优）、CliffWalking（解析最优 −12）、LQR（Riccati 闭式） |
| L2 | `dp` | Value Iteration、Policy Iteration（精确动态规划） |
| L3 | `td` | Q-learning / SARSA / Expected SARSA（共享 `_core` 引擎，表格型） |
| L4 | `deep` | DQN（经验回放 + 目标网络）、手写 MLP（前向/反向/SGD，零 autodiff 依赖） |
| L5 | `pg` | REINFORCE / A2C / PPO-clip / A2C-LQR（策略梯度家族） |
| L6 | `fusion` | RlFuse —— 跨 holdout 起点择优，诚实回退，永不退化 |
| L7 | `gates` | 7 道交付门禁 G1–G7（系统级硬不变量） |
| L8 | `cli` | `rlforge gates\|bench\|demo` |
| L9 | 可复现 | 同 seed → 逐位一致；CI 自动跑门禁 + 测试 |
| L10 | 包 | 本 `rlforge` 包 |

---

## 七道交付门禁

| 门 | 不变量 | 证据 |
|----|--------|------|
| G1 | 确定性 · 同 seed 逐位一致 | Q-learning / DQN 复跑结果逐位相等 |
| G2 | 最优值 / 最优控制可证匹配 | Grid VI == BFS 最优；Cliff V(start) = −12.0 |
| G3 | Bellman 残差 → 0 | ‖T\*V − V‖∞ < 1e-9 |
| G4 | Q-learning 收敛到 Q\* | 贪婪回报 == 最优回报 |
| G5 | DQN 学会 | 回报曲线上升 · 近最优（贪返回报 == −5.0） |
| G6 | 策略梯度学会 | A2C / PPO 回报上升 · 近最优 |
| G7 | LQR 逼近 Riccati 最优 | 学习成本 11.76 vs 最优 11.74，闭环稳定 |

跑 `rlforge gates` 即见实时结果；CI 在每次 push/PR 自动复跑。

---

## 快速上手

```python
from rlforge.envs import GridWorld
from rlforge.dp import value_iteration
from rlforge.fusion import simulate_greedy

mdp = GridWorld().tabular()
V, pi = value_iteration(mdp)
print("optimal return:", simulate_greedy(mdp, pi))   # -5.0
```

```bash
rlforge demo     # 打印 GridWorld 最优策略箭头
rlforge bench    # 各环境：VI 最优 vs RlFuse 择优
```

---

## 设计原则（对齐三大铁律）

- **零重写**：复用世界顶级数学与算法（Bellman 不动点、Riccati、PPO-clip…），不造轮子。
- **集大成**：每条算法都站在教科书 / 顶会论文的肩膀上，并附带可验证不变量。
- **闭环交付**：完整开源、可运行、可复现、可验证，署名**晨星**。

CPU-only，确定性，无外部服务依赖 —— 在任何机器上 `pip install` 后即可复现全部门禁。
