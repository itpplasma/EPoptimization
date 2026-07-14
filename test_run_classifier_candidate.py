from pathlib import Path


def test_classifier_candidate_uses_two_heads_without_fractal_or_direct_loss() -> None:
    text = Path("run_classifier_candidate.sh").read_text()

    assert 'json.load(open(sys.argv[1]))["radial_surfaces"]' in text
    assert '--surfaces "${surface_values[@]}"' in text
    assert "--classifier topology" in text
    assert "--classifier jpar" in text
    assert "--metric gamma_c" in text
    assert "evaluate_direct_loss" not in text
    assert "evaluate_threshold_loss" not in text
    assert "fractal" not in text.lower()
