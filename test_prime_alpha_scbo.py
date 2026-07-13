from __future__ import annotations

import argparse

from prime_alpha_scbo import campaign_configuration, local_priming_rows, priming_rows


def test_priming_rows_add_reference_and_preserve_failures() -> None:
    scout = {
        "late_reduction_target": 0.01,
        "cases": [
            {
                "unit": [0.2, 0.8],
                "status": "ok",
                "total": {"change": -0.05},
                "observation": {
                    "value": 0.3,
                    "variance": 0.002,
                    "constraints": [-0.02],
                    "constraint_variances": [0.001],
                },
            },
            {
                "unit": [0.7, 0.3],
                "status": "failed",
                "failure_kind": "equilibrium_failure",
            },
        ],
    }

    extra = [
        {
            "candidate_id": 3,
            "unit_x": [0.4, 0.6],
            "observation": {
                "value": -0.02,
                "variance": 0.001,
                "constraints": [-0.01],
                "constraint_variances": [0.001],
            },
            "status": "ok",
            "failure_kind": None,
        }
    ]

    rows = priming_rows(scout, extra)

    assert [row["candidate_id"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["unit_x"] == [0.5, 0.5]
    assert rows[0]["observation"]["constraints"] == [0.01]
    assert rows[1]["observation"]["value"] == -0.05
    assert rows[2]["observation"] is None
    assert rows[2]["failure_kind"] == "equilibrium_failure"
    assert rows[3] == extra[0]

    local = local_priming_rows(scout, extra)
    assert [row["candidate_id"] for row in local] == [0, 1]
    assert local[1]["unit_x"] == extra[0]["unit_x"]


def test_local_configuration_accepts_all_resolved_priming_rows() -> None:
    rows = [{"unit_x": [0.5, 0.5]} for _ in range(17)]
    args = argparse.Namespace(
        local_only=True,
        workers=8,
        new_calls=128,
        seed=7102,
        initial_length=0.3,
    )

    configuration = campaign_configuration(args, rows)

    assert configuration["initial_points"] == 17
    assert configuration["budget"] == 145
