from pathlib import Path


def test_shell_candidate_uses_fixed_shell_and_short_prompt_trace() -> None:
    text = Path("run_shell_candidate.sh").read_text()

    assert "--surfaces 0.25 0.675 0.8" in text
    assert "--shell-inner 0.675" in text
    assert "--shell-outer 0.8" in text
    assert "--particles \"${PROMPT_PARTICLES:-128}\"" in text
    assert "--birth-surface 0.25" in text
    assert "--trace-time \"${PROMPT_TRACE_TIME:-0.0011}\"" in text
    assert "--classifier topology" in text
    assert 'test -f "$proxy/design/manifest.json"' in text
    assert 'rm -rf "$proxy/design"' in text
    assert "--classifier jpar" not in text
    assert "escape" not in text.lower()
    assert "fractal" not in text.lower()


def test_incremental_validation_metadata_uses_script_arguments() -> None:
    text = Path("run_shell_candidate.sh").read_text()

    assert '"$candidate_id" "$case_root" "$(basename "$wout")" "$wout_sha"' in text
    assert '--export="ALL,CANDIDATE_ID=' not in text
