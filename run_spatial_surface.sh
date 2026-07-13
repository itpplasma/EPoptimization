#!/bin/bash
set -euo pipefail

index=${1:?surface manifest index required}
line=$(sed -n "$((index + 1))p" "$ATLAS_ROOT/manifest.tsv")
IFS=$'\t' read -r case_name design_relative design_sha start_sha wout_relative wout_sha output_relative <<< "$line"
test -n "$case_name"
design=$ATLAS_ROOT/$design_relative
output=$ATLAS_ROOT/$output_relative
wout=$ATLAS_ROOT/$wout_relative
if test -f "$output/topology.npz"; then
    exit 0
fi
test "$(sha256sum "$design/design.npz" | cut -d' ' -f1)" = "$design_sha"
test "$(sha256sum "$design/start.dat" | cut -d' ' -f1)" = "$start_sha"
test "$(sha256sum "$wout" | cut -d' ' -f1)" = "$wout_sha"
test "$(sha256sum "$SIMPLE_X" | cut -d' ' -f1)" = "$SIMPLE_SHA256"
test "$(sha256sum "$CODE_ROOT/evaluate_spatial_surface.py" | cut -d' ' -f1)" = "$EVALUATOR_SHA256"
test "$(sha256sum "$CODE_ROOT/spatial_grid.py" | cut -d' ' -f1)" = "$SPATIAL_GRID_SHA256"
test "$(sha256sum "$CODE_ROOT/simple_barrier.py" | cut -d' ' -f1)" = "$SIMPLE_BARRIER_SHA256"
export OMP_NUM_THREADS=$ALLOCATED_CPUS
python3 "$CODE_ROOT/evaluate_spatial_surface.py" \
    --wout "$wout" \
    --design "$design" \
    --out "$output" \
    --simple-executable "$SIMPLE_X" \
    --trace-time "${TRACE_TIME:-0.02}" \
    --timeout "${SPATIAL_TIMEOUT_SECONDS:-3600}"
