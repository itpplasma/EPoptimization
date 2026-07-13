#!/bin/bash
set -euo pipefail

index=${1:?candidate manifest index required}
wave_root=$CAMPAIGN_ROOT/$WAVE
line=$(sed -n "$((index + 1))p" "$wave_root/manifest.tsv")
IFS=$'\t' read -r case_name relative_input input_sha <<< "$line"
test -n "$case_name"
case_root=$wave_root/candidates/$case_name
input_name=$(basename "$relative_input")
test "$(sha256sum "$wave_root/candidates/$relative_input" | cut -d' ' -f1)" = "$input_sha"
test "$(sha256sum "$XVMEC" | cut -d' ' -f1)" = "$XVMEC_SHA256"
test "$(sha256sum "$SIMPLE_X" | cut -d' ' -f1)" = "$SIMPLE_SHA256"

set +e
(cd "$case_root" && timeout "${VMEC_TIMEOUT_SECONDS:-600}" "$XVMEC" "$input_name" > vmec.stdout 2>&1)
vmec_exit=$?
set -e
if test "$vmec_exit" -ne 0 || ! grep -q "EXECUTION TERMINATED NORMALLY" "$case_root/vmec.stdout"; then
    python3 "$CODE_ROOT/build_scbo_response.py" \
        --request "$case_root/request.json" \
        --out "$case_root/response.json" \
        --failure-kind equilibrium_failure
    exit 0
fi

wout=$case_root/wout_${case_name}.nc
wout_sha=$(sha256sum "$wout" | cut -d' ' -f1)
export OMP_NUM_THREADS=$ALLOCATED_CPUS
python3 "$CODE_ROOT/evaluate_threshold_loss.py" \
    --wout "$wout" \
    --wout-sha256 "$wout_sha" \
    --out "$case_root/direct_result" \
    --simple-executable "$SIMPLE_X" \
    --simple-sha256 "$SIMPLE_SHA256" \
    --particles "${PARTICLES:-256}" \
    --seed "${PARTICLE_SEED:-12345}" \
    --birth-surface "${BIRTH_SURFACE:-0.3}" \
    --trace-time "${TRACE_TIME:-0.3}" \
    --loss-threshold "${LOSS_THRESHOLD:-0.38}"
python3 "$CODE_ROOT/build_scbo_response.py" \
    --request "$case_root/request.json" \
    --result "$case_root/direct_result/result.json" \
    --reference-times "$REFERENCE_TIMES" \
    --out "$case_root/response.json" \
    --late-target "${LATE_TARGET:-0.01}"
