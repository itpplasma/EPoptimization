from __future__ import annotations

import json

import pytest

from generate_scbo_candidates import load_requests


def test_load_requests_preserves_ordered_bounded_coordinates(tmp_path) -> None:
    path = tmp_path / "requests.json"
    path.write_text(
        json.dumps(
            {
                "requests": [
                    {"candidate_id": 12, "unit_x": [0.2, 0.8]},
                    {"candidate_id": 13, "unit_x": [0.4, 0.6]},
                ]
            }
        )
    )

    rows = load_requests(path, 2)

    assert [row["candidate_id"] for row in rows] == [12, 13]
    assert rows[0]["unit_x"] == [0.2, 0.8]


def test_load_requests_rejects_duplicate_candidate(tmp_path) -> None:
    path = tmp_path / "requests.json"
    path.write_text(
        json.dumps(
            [
                {"candidate_id": 2, "unit_x": [0.2]},
                {"candidate_id": 2, "unit_x": [0.3]},
            ]
        )
    )

    with pytest.raises(ValueError, match="unique"):
        load_requests(path, 1)
