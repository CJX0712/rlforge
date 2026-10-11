"""Delivery-gate acceptance tests (L7).

Each gate is a hard, independent invariant. The rule (mirrors 天枢) is that a
failing gate is reported honestly — thresholds are never loosened to turn red
green. These tests encode exactly the claims RlForge makes. Author: 晨星.
"""
from rlforge.gates import (
    gate_determinism,
    gate_optimal_match,
    gate_bellman_residual,
    gate_q_learning,
    gate_dqn,
    gate_policy_gradient,
    gate_lqr,
    run_gates,
)


def test_g1_determinism():
    ok, detail = gate_determinism()
    assert ok, detail


def test_g2_optimal_match():
    ok, detail = gate_optimal_match()
    assert ok, detail


def test_g3_bellman_residual():
    ok, detail = gate_bellman_residual()
    assert ok, detail


def test_g4_q_learning():
    ok, detail = gate_q_learning()
    assert ok, detail


def test_g5_dqn():
    ok, detail = gate_dqn()
    assert ok, detail


def test_g6_policy_gradient():
    ok, detail = gate_policy_gradient()
    assert ok, detail


def test_g7_lqr():
    ok, detail = gate_lqr()
    assert ok, detail


def test_all_gates_pass():
    results = run_gates(verbose=False)
    failed = [(gid, name, detail) for gid, name, ok, detail in results if not ok]
    assert not failed, f"gates failed: {failed}"
    assert len(results) == 7
