import json

import pytest

from rebuild_shell_total_scbo import load_responses, total_loss_row


def response(candidate_id, constraints=None):
    observation = None
    pareto = None
    if constraints is not None:
        observation = {
            "value": -0.03,
            "variance": 0.002,
            "constraints": constraints,
            "constraint_variances": [0.0] * len(constraints),
        }
        pareto = {
            "values": [0.01, 0.02, -0.06],
            "variances": [0.0, 0.0, 0.0],
            "constraints": [-0.004],
            "constraint_variances": [0.0],
        }
    return {
        "candidate_id": candidate_id,
        "unit_x": [0.4, 0.6],
        "observation": observation,
        "pareto_observation": pareto,
        "status": "ok" if observation else "failed",
        "failure_kind": None if observation else "equilibrium_failure",
    }


def test_total_loss_row_keeps_value_and_only_gamma_guard():
    row = total_loss_row(response(2, [-0.1, 0.2, -0.3, 0.4, -0.004]))
    assert row["observation"] == {
        "value": -0.03,
        "variance": 0.002,
        "constraints": [-0.004],
        "constraint_variances": [0.0],
    }


def test_load_responses_combines_roots_and_requires_contiguous_ids(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    for root, candidate_id in zip(roots, (0, 1), strict=True):
        case = root / f"candidate-{candidate_id:08d}"
        case.mkdir(parents=True)
        (case / "response.json").write_text(json.dumps(response(candidate_id, [0.0])))
    assert [row["candidate_id"] for row in load_responses(roots)] == [0, 1]

    old = roots[1] / "candidate-00000001/response.json"
    old.unlink()
    case = roots[1] / "candidate-00000002"
    case.mkdir()
    (case / "response.json").write_text(json.dumps(response(2, [0.0])))
    with pytest.raises(ValueError, match="contiguous"):
        load_responses(roots)
