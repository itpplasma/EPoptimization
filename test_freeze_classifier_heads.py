import pytest

from freeze_classifier_heads import freeze


def _fit(passes: bool, slope: float = 2.0) -> dict:
    return {"passes": passes, "slope": slope, "spearman": 0.9}


def test_freeze_selects_smallest_passing_nonfractal_heads() -> None:
    result = freeze(
        {
            "fractal_features": [],
            "fits": {
                "prompt_topology_nonideal": _fit(True),
                "prompt_jpar_nonideal": _fit(True),
                "prompt_unclassified": _fit(True),
                "late_topology_escape": _fit(False),
                "late_jpar_escape": _fit(False),
                "late_topology_nonideal": _fit(True, -3.0),
                "late_jpar_nonideal": _fit(True, -3.0),
            },
        }
    )
    assert result["prompt"]["feature"] == "prompt_topology_nonideal"
    assert result["late"]["feature"] == "late_topology_nonideal"
    assert result["fractal_features"] == []


def test_freeze_rejects_missing_head_or_fractal_input() -> None:
    fits = {
        name: _fit(False)
        for name in (
            "prompt_topology_nonideal",
            "prompt_jpar_nonideal",
            "prompt_unclassified",
            "late_topology_escape",
            "late_jpar_escape",
            "late_topology_nonideal",
            "late_jpar_nonideal",
        )
    }
    with pytest.raises(ValueError, match="prompt"):
        freeze({"fractal_features": [], "fits": fits})
    with pytest.raises(ValueError, match="fractal"):
        freeze({"fractal_features": ["dimension"], "fits": fits})
