import json

from advance_alpha_nsga2 import pending_responses
from prime_response_scbo import load_rows


def response(candidate_id):
    return {
        "candidate_id": candidate_id,
        "unit_x": [0.5, 0.5],
        "status": "ok",
        "failure_kind": None,
        "observation": {
            "value": 0.0,
            "variance": 0.0,
            "constraints": [-1.0],
            "constraint_variances": [0.0],
        },
        "pareto_observation": {
            "values": [0.0, 0.0, 0.0],
            "variances": [0.0, 0.0, 0.0],
            "constraints": [-1.0],
            "constraint_variances": [0.0],
        },
    }


def test_load_rows_requires_contiguous_successes(tmp_path):
    for candidate_id in range(2):
        case = tmp_path / str(candidate_id)
        case.mkdir()
        (case / "response.json").write_text(json.dumps(response(candidate_id)))
    assert [row["candidate_id"] for row in load_rows(tmp_path)] == [0, 1]


def test_nsga_pending_responses_preserves_generation(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    row = {**response(5), "generation": 2}
    (case / "response.json").write_text(json.dumps(row))
    state = {"pending": [{"candidate_id": 5}]}
    assert pending_responses(state, tmp_path) == [row]
