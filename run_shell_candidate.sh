#!/bin/bash
set -Eeuo pipefail

index=${1:?candidate manifest index required}
: "${SHELL_HEAD:?}"
wave_root=$CAMPAIGN_ROOT/$WAVE
line=$(sed -n "$((index + 1))p" "$wave_root/manifest.tsv")
IFS=$'\t' read -r case_name relative_input input_sha <<< "$line"
test -n "$case_name"
case_root=$wave_root/candidates/$case_name
input_name=$(basename "$relative_input")

failure_response() {
    if ! test -f "$case_root/response.json"; then
        python3 "$CODE_ROOT/build_shell_scbo_response.py" \
            --request "$case_root/request.json" \
            --out "$case_root/response.json" \
            --failure-kind shell_proxy_failure
    fi
}
trap failure_response ERR

test "$(sha256sum "$wave_root/candidates/$relative_input" | cut -d' ' -f1)" = "$input_sha"
test "$(sha256sum "$XVMEC" | cut -d' ' -f1)" = "$XVMEC_SHA256"
test "$(sha256sum "$CLASSIFIER_SIMPLE_X" | cut -d' ' -f1)" = "$CLASSIFIER_SIMPLE_SHA256"
test "$(sha256sum "$DIRECT_SIMPLE_X" | cut -d' ' -f1)" = "$DIRECT_SIMPLE_SHA256"
test "$(sha256sum "$CODE_MANIFEST" | cut -d' ' -f1)" = "$CODE_MANIFEST_SHA256"
(cd "$CODE_ROOT" && sha256sum -c "$CODE_MANIFEST")

set +e
(cd "$case_root" && timeout "${VMEC_TIMEOUT_SECONDS:-600}" "$XVMEC" "$input_name" > vmec.stdout 2>&1)
vmec_exit=$?
set -e
if test "$vmec_exit" -ne 0 || ! grep -q "EXECUTION TERMINATED NORMALLY" "$case_root/vmec.stdout"; then
    trap - ERR
    python3 "$CODE_ROOT/build_shell_scbo_response.py" \
        --request "$case_root/request.json" \
        --out "$case_root/response.json" \
        --failure-kind equilibrium_failure
    exit 0
fi

wout=$case_root/wout_${case_name}.nc
wout_sha=$(sha256sum "$wout" | cut -d' ' -f1)
proxy=$case_root/shell_proxy
if ! test -f "$proxy/design/manifest.json"; then
    rm -rf "$proxy/design"
    python3 "$CODE_ROOT/generate_spatial_grid.py" \
        --wout "$wout" \
        --out "$proxy/design" \
        --surfaces 0.25 0.675 0.8 \
        --ntheta 16 \
        --nzeta 16 \
        --nmu 9
fi

export OMP_NUM_THREADS=$ALLOCATED_CPUS
for surface in s0p25000 s0p67500 s0p80000; do
    if ! test -f "$proxy/surfaces/$surface/topology.npz"; then
        python3 "$CODE_ROOT/evaluate_spatial_surface.py" \
            --wout "$wout" \
            --design "$proxy/design/$surface" \
            --out "$proxy/surfaces/$surface" \
            --simple-executable "$CLASSIFIER_SIMPLE_X" \
            --trace-time 0.02 \
            --timeout "${SPATIAL_TIMEOUT_SECONDS:-3600}"
    fi
done

python3 "$CODE_ROOT/evaluate_spatial_atlas.py" \
    --topology \
        "$proxy/surfaces/s0p25000/topology.npz" \
        "$proxy/surfaces/s0p67500/topology.npz" \
        "$proxy/surfaces/s0p80000/topology.npz" \
    --classifier topology \
    --shell-inner 0.675 \
    --shell-outer 0.8 \
    --out "$proxy/shell.json" \
    --risk-out "$proxy/risk.npz"

python3 "$CODE_ROOT/evaluate_direct_loss.py" \
    --wout "$wout" \
    --wout-sha256 "$wout_sha" \
    --out "$case_root/short_result" \
    --simple-executable "$DIRECT_SIMPLE_X" \
    --simple-sha256 "$DIRECT_SIMPLE_SHA256" \
    --particles "${PROMPT_PARTICLES:-128}" \
    --seed "${PROMPT_SEED:-12345}" \
    --birth-surface 0.25 \
    --prompt-time 0.0001 \
    --trace-time "${PROMPT_TRACE_TIME:-0.0011}"

"$DESC_PYTHON" "$CODE_ROOT/reactor_proxy_calibration.py" desc-worker \
    --metric gamma_c \
    --wout "$wout" > "$proxy/gamma-c.json"
gamma_c=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["values"][0])' "$proxy/gamma-c.json")

trap - ERR
python3 "$CODE_ROOT/build_shell_scbo_response.py" \
    --request "$case_root/request.json" \
    --shell "$proxy/shell.json" \
    --reference-shell "$REFERENCE_SHELL" \
    --shell-head "$SHELL_HEAD" \
    --short-result "$case_root/short_result/result.json" \
    --reference-short-times "$REFERENCE_SHORT_TIMES" \
    --gamma-c "$gamma_c" \
    --reference-gamma-c "$REFERENCE_GAMMA_C" \
    --prompt-tolerance "${PROMPT_TOLERANCE:-0.005}" \
    --early-tolerance "${EARLY_TOLERANCE:-0.005}" \
    --late-tolerance "${LATE_TOLERANCE:-0.0}" \
    --out "$case_root/response.json"
