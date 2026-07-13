#!/bin/bash
set -euo pipefail

index=${1:?reference manifest index required}
line=$(sed -n "$((index + 1))p" "$REFERENCE_ROOT/manifest.tsv")
IFS=$'\t' read -r case_name seed <<< "$line"
test -n "$case_name"
test "$(sha256sum "$WOUT" | cut -d' ' -f1)" = "$WOUT_SHA256"
test "$(sha256sum "$SIMPLE_X" | cut -d' ' -f1)" = "$SIMPLE_SHA256"
test "$(sha256sum "$CODE_ROOT/evaluate_threshold_loss.py" | cut -d' ' -f1)" = "$EVALUATOR_SHA256"
export OMP_NUM_THREADS=$ALLOCATED_CPUS
python3 "$CODE_ROOT/evaluate_threshold_loss.py" \
    --wout "$WOUT" \
    --wout-sha256 "$WOUT_SHA256" \
    --out "$REFERENCE_ROOT/results/$case_name" \
    --simple-executable "$SIMPLE_X" \
    --simple-sha256 "$SIMPLE_SHA256" \
    --particles "${PARTICLES:-1024}" \
    --seed "$seed" \
    --birth-surface "${BIRTH_SURFACE:-0.25}" \
    --trace-time "${TRACE_TIME:-0.3}" \
    --loss-threshold "${LOSS_THRESHOLD:-0.38}"
