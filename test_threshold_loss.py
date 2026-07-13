#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pytest

from evaluate_threshold_loss import parser, score_trace
from loss_threshold_objective import threshold_crossing, threshold_objective


def test_threshold_crossing_interpolates_monotone_curve() -> None:
    curve = np.array(
        [
            [0.00, 1.00, 0.00],
            [0.01, 0.90, 0.00],
            [0.02, 0.70, 0.00],
            [0.03, 0.55, 0.00],
        ]
    )
    crossing = threshold_crossing(curve, 0.20)
    assert crossing == pytest.approx(0.015)
    assert threshold_crossing(curve, 0.50) is None


def test_threshold_objective_is_continuous_and_rewards_lower_completed_loss() -> None:
    threshold = 0.38
    trace_time = 0.3
    at_crossing = threshold_objective(
        threshold, trace_time, threshold, crossing_time=trace_time
    )
    at_completion = threshold_objective(
        threshold, trace_time, threshold, crossing_time=None
    )
    better = threshold_objective(0.35, trace_time, threshold, crossing_time=None)
    assert at_crossing == pytest.approx(at_completion)
    assert better < at_completion


def test_threshold_objective_rejects_inconsistent_crossing() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        threshold_objective(0.2, 0.3, 0.38, crossing_time=0.1)


def test_threshold_evaluator_defaults_freeze_search_contract() -> None:
    args = parser().parse_args(
        [
            "--wout",
            "/tmp/wout.nc",
            "--out",
            "/tmp/output",
            "--simple-executable",
            "/tmp/simple.x",
            "--simple-sha256",
            "0" * 64,
            "--wout-sha256",
            "1" * 64,
        ]
    )
    assert args.particles == 1024
    assert args.trace_time == 0.3
    assert args.loss_threshold == 0.38
    assert args.prompt_time == 1.0e-3


def test_full_trace_search_objective_is_total_loss() -> None:
    args = parser().parse_args(
        [
            "--wout",
            "/tmp/wout.nc",
            "--out",
            "/tmp/output",
            "--simple-executable",
            "/tmp/simple.x",
            "--simple-sha256",
            "0" * 64,
            "--wout-sha256",
            "1" * 64,
        ]
    )
    curve = np.array([[0.001, 0.8, 0.0], [0.3, 0.55, 0.0]])

    scores = score_trace(curve, 0.45, args)

    assert scores["objective"] == 0.45
    assert scores["threshold_score"] != scores["objective"]
