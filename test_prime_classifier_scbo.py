import argparse
import json

import pytest

from prime_classifier_scbo import configuration, load_responses


def _response(candidate_id: int, constraints: int = 9) -> dict:
    return {
        "candidate_id": candidate_id,
        "unit_x": [0.5] * 8,
        "observation": {
            "value": 0.0,
            "variance": 0.0,
            "constraints": [0.0] * constraints,
            "constraint_variances": [0.0] * constraints,
        },
        "status": "ok",
        "failure_kind": None,
    }


def test_classifier_priming_infers_all_proxy_constraints(tmp_path) -> None:
    for candidate_id in range(3):
        case = tmp_path / str(candidate_id)
        case.mkdir()
        (case / "response.json").write_text(json.dumps(_response(candidate_id)))
    rows = load_responses(tmp_path)
    args = argparse.Namespace(
        workers=8, new_calls=32, seed=7, initial_length=0.2
    )

    result = configuration(args, rows)

    assert result["constraint_count"] == 9
    assert result["dimension"] == 8
    assert result["initial_points"] == 16
    assert result["budget"] == 35


def test_classifier_priming_rejects_mixed_constraint_counts(tmp_path) -> None:
    for candidate_id, constraints in ((0, 9), (1, 8)):
        case = tmp_path / str(candidate_id)
        case.mkdir()
        (case / "response.json").write_text(
            json.dumps(_response(candidate_id, constraints))
        )

    with pytest.raises(ValueError, match="constraint counts"):
        load_responses(tmp_path)
