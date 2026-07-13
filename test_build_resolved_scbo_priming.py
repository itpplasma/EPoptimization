from __future__ import annotations

import argparse
import json

from build_resolved_scbo_priming import build


def test_build_preserves_paired_variances_and_late_constraint(tmp_path) -> None:
    search = tmp_path / "search" / "wave00" / "candidates" / "candidate-00000007"
    search.mkdir(parents=True)
    (search / "request.json").write_text(json.dumps({"unit_x": [0.2, 0.3]}))
    promotion = {
        "candidates": {
            "7": {
                "aggregate": {
                    "total": {"change": -0.03, "paired_se": 0.004},
                    "late": {"change": -0.02, "paired_se": 0.003},
                }
            }
        }
    }
    blend = {
        "cases": [
            {
                "unit": [0.4, 0.5],
                "total": {"change": -0.01, "paired_se": 0.02},
                "late": {"change": 0.01, "paired_se": 0.03},
            }
        ]
    }
    promotion_path = tmp_path / "promotion.json"
    blend_path = tmp_path / "blend.json"
    promotion_path.write_text(json.dumps(promotion))
    blend_path.write_text(json.dumps(blend))
    args = argparse.Namespace(
        promotion=promotion_path,
        blend=blend_path,
        search_root=tmp_path / "search",
        out=tmp_path / "out",
    )

    rows = build(args)

    assert [row["candidate_id"] for row in rows] == [1, 2]
    assert rows[0]["observation"] == {
        "value": -0.03,
        "variance": 0.004**2,
        "constraints": [-0.01],
        "constraint_variances": [0.003**2],
    }
    assert rows[1]["observation"]["constraints"] == [0.02]
