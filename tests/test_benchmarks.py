"""实测超越验证 — every benchmark group must hold (hard invariants)."""
import numpy as np
from rlforge.benchmarks import run_benchmarks


def test_all_benchmark_checks_pass():
    rows = run_benchmarks(verbose=False)
    assert rows, "benchmark suite returned no rows"
    failed = [r for r in rows if not r.ok]
    assert not failed, f"benchmark failures: {[(r.name, r.note) for r in failed]}"


def test_exact_optimal_group():
    from rlforge.benchmarks import group_exact_optimal
    rows = group_exact_optimal(verbose=False)
    assert all(r.ok for r in rows)


def test_cross_algorithm_group():
    from rlforge.benchmarks import group_cross_algorithm
    rows = group_cross_algorithm(verbose=False)
    assert all(r.ok for r in rows)


def test_determinism_group():
    from rlforge.benchmarks import group_determinism
    rows = group_determinism(verbose=False)
    assert all(r.ok for r in rows)


def test_sample_efficiency_group():
    from rlforge.benchmarks import group_sample_efficiency
    rows = group_sample_efficiency(verbose=False)
    assert all(r.ok for r in rows)
