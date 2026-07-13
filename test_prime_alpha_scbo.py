from __future__ import annotations

from prime_alpha_scbo import priming_rows


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

    rows = priming_rows(scout)

    assert [row["candidate_id"] for row in rows] == [0, 1, 2]
    assert rows[0]["unit_x"] == [0.5, 0.5]
    assert rows[0]["observation"]["constraints"] == [0.01]
    assert rows[1]["observation"]["value"] == -0.05
    assert rows[2]["observation"] is None
    assert rows[2]["failure_kind"] == "equilibrium_failure"
