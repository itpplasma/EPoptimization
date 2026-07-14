import numpy as np

from validate_classifier_heads import validate_predictions


def _labels() -> dict:
    return {
        "birth_surface": 0.25,
        "candidates": {
            "74": {
                "wout_sha256": "w74",
                "aggregate": {
                    "prompt": {"change": 0.078, "paired_se": 0.007},
                    "late": {"change": -0.075, "paired_se": 0.007},
                },
            },
            "83": {
                "wout_sha256": "w83",
                "aggregate": {
                    "prompt": {"change": 0.040, "paired_se": 0.007},
                    "late": {"change": -0.032, "paired_se": 0.007},
                },
            },
            "88": {
                "wout_sha256": "w88",
                "aggregate": {
                    "prompt": {"change": 0.033, "paired_se": 0.007},
                    "late": {"change": -0.035, "paired_se": 0.007},
                },
            },
        },
    }


def _predictions() -> dict:
    return {
        "74": {
            "wout_sha256": "w74",
            "predictions": {"prompt": [0.08, 0.07], "late": [-0.08, -0.07]},
            "fractal_features": [],
        },
        "83": {
            "wout_sha256": "w83",
            "predictions": {"prompt": [0.04, 0.04], "late": [-0.03, -0.03]},
            "fractal_features": [],
        },
        "88": {
            "wout_sha256": "w88",
            "predictions": {"prompt": [0.03, 0.03], "late": [-0.04, -0.04]},
            "fractal_features": [],
        },
    }


def test_heldout_gate_passes_shiftwise_sign_and_resolved_ordering() -> None:
    result = validate_predictions(_labels(), _predictions())
    assert result["passes"]
    assert result["prompt"]["passes"]
    assert result["late"]["passes"]
    assert np.allclose(result["late"]["ordering_spearman"], [1.0, 1.0])


def test_heldout_gate_rejects_a_false_safe_shift() -> None:
    predictions = _predictions()
    predictions["74"]["predictions"]["prompt"][1] = -0.01
    result = validate_predictions(_labels(), predictions)
    assert not result["passes"]
    assert result["prompt"]["false_safe"] == [[False, True], [False, False], [False, False]]
