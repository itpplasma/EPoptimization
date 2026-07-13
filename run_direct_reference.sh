#!/bin/bash
set -euo pipefail

index=${1:?reference manifest index required}
line=$(sed -n "$((index + 1))p" "$REFERENCE_ROOT/manifest.tsv")
IFS=$'\t' read -r -a fields <<< "$line"
case_name=${fields[0]}
if test "${#fields[@]}" -eq 2; then
    seed=${fields[1]}
    wout_relative=
    manifest_wout_sha=
elif test "${#fields[@]}" -eq 5; then
    wout_relative=${fields[2]}
    manifest_wout_sha=${fields[3]}
    seed=${fields[4]}
else
    echo "manifest row must have two or five columns" >&2
    exit 2
fi
test -n "$case_name"
if test -n "$wout_relative"; then
    wout=$REFERENCE_ROOT/$wout_relative
    wout_sha=$manifest_wout_sha
else
    wout=$WOUT
    wout_sha=$WOUT_SHA256
fi
test "$(sha256sum "$wout" | cut -d' ' -f1)" = "$wout_sha"
test "$(sha256sum "$SIMPLE_X" | cut -d' ' -f1)" = "$SIMPLE_SHA256"
test "$(sha256sum "$CODE_ROOT/evaluate_threshold_loss.py" | cut -d' ' -f1)" = "$EVALUATOR_SHA256"
export OMP_NUM_THREADS=$ALLOCATED_CPUS
python3 "$CODE_ROOT/evaluate_threshold_loss.py" \
    --wout "$wout" \
    --wout-sha256 "$wout_sha" \
    --out "$REFERENCE_ROOT/results/$case_name" \
    --simple-executable "$SIMPLE_X" \
    --simple-sha256 "$SIMPLE_SHA256" \
    --particles "${PARTICLES:-1024}" \
    --seed "$seed" \
    --birth-surface "${BIRTH_SURFACE:-0.25}" \
    --trace-time "${TRACE_TIME:-0.3}" \
    --loss-threshold "${LOSS_THRESHOLD:-0.38}"
