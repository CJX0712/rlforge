"""Command-line entry point: `rlforge gates|bench|demo`."""
from __future__ import annotations
import sys
import numpy as np
from .envs import GridWorld, CliffWalking
from .dp import value_iteration
from .fusion import rlfuse, simulate_greedy
from .gates import run_gates
from .benchmarks import run_benchmarks


def _bench():
    run_benchmarks(verbose=True)
    return 0


def _demo():
    gw = GridWorld()
    mdp = gw.tabular()
    V, pi = value_iteration(mdp)
    gw.reset(seed=0)            # render() needs a current position
    print("RlForge · demo (GridWorld optimal policy)\n")
    print(gw.render())
    arrows = "".join(["↑", "→", "↓", "←"][a] for a in pi)
    print("\nOptimal policy (arrows per state):", arrows)
    return 0


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "gates"
    if cmd == "gates":
        print("RlForge · delivery gates\n")
        results = run_gates(verbose=True)
        failed = [r for r in results if not r[2]]
        print(f"\n{len(results) - len(failed)}/{len(results)} gates PASS")
        return 0 if not failed else 1
    if cmd == "bench":
        return _bench()
    if cmd == "demo":
        return _demo()
    print(f"unknown command: {cmd}\nusage: rlforge gates|bench|demo")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
