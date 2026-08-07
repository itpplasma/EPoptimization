from __future__ import annotations

import pytest

from validate_barrier_optimum import best_feasible_record


def record(evaluation_id, value, constraints):
    return {
        "evaluation_id": evaluation_id,
        "observation": {"value": value, "constraints": constraints},
    }


def test_best_feasible_beats_a_lower_infeasible_value() -> None:
    ledger = {
        "records": [
            record(0, 0.30, [-1.0, -1.0, -1.0]),
            record(1, 0.05, [0.4, -1.0, -1.0]),
            record(2, 0.12, [-1.0, -1.0, -1.0]),
        ]
    }
    assert best_feasible_record(ledger)["evaluation_id"] == 2


def test_falls_back_to_the_best_infeasible_when_nothing_is_feasible() -> None:
    ledger = {
        "records": [
            record(0, 0.30, [0.1, -1.0, -1.0]),
            record(1, 0.05, [0.4, -1.0, -1.0]),
        ]
    }
    assert best_feasible_record(ledger)["evaluation_id"] == 1


def test_failed_evaluations_are_ignored() -> None:
    ledger = {
        "records": [
            {"evaluation_id": 0, "observation": None},
            record(1, 0.2, [-1.0, -1.0, -1.0]),
        ]
    }
    assert best_feasible_record(ledger)["evaluation_id"] == 1


def test_a_ledger_without_successes_is_rejected() -> None:
    with pytest.raises(ValueError):
        best_feasible_record({"records": [{"evaluation_id": 0, "observation": None}]})
