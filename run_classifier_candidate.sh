#!/bin/bash
set -Eeuo pipefail

index=${1:?candidate manifest index required}
wave_root=$CAMPAIGN_ROOT/$WAVE
line=$(sed -n "$((index + 1))p" "$wave_root/manifest.tsv")
IFS=$'\t' read -r case_name relative_input input_sha <<< "$line"
test -n "$case_name"
case_root=$wave_root/candidates/$case_name
input_name=$(basename "$relative_input")

failure_response() {
    if ! test -f "$case_root/response.json"; then
        python3 "$CODE_ROOT/build_classifier_scbo_response.py" \
            --request "$case_root/request.json" \
            --out "$case_root/response.json" \
            --failure-kind classifier_proxy_failure
    fi
}
trap failure_response ERR

test "$(sha256sum "$wave_root/candidates/$relative_input" | cut -d' ' -f1)" = "$input_sha"
test "$(sha256sum "$XVMEC" | cut -d' ' -f1)" = "$XVMEC_SHA256"
test "$(sha256sum "$SIMPLE_X" | cut -d' ' -f1)" = "$SIMPLE_SHA256"
test "$(sha256sum "$CODE_MANIFEST" | cut -d' ' -f1)" = "$CODE_MANIFEST_SHA256"
(cd "$CODE_ROOT" && sha256sum -c "$CODE_MANIFEST")

set +e
(cd "$case_root" && timeout "${VMEC_TIMEOUT_SECONDS:-600}" "$XVMEC" "$input_name" > vmec.stdout 2>&1)
vmec_exit=$?
set -e
if test "$vmec_exit" -ne 0 || ! grep -q "EXECUTION TERMINATED NORMALLY" "$case_root/vmec.stdout"; then
    trap - ERR
    python3 "$CODE_ROOT/build_classifier_scbo_response.py" \
        --request "$case_root/request.json" \
        --out "$case_root/response.json" \
        --failure-kind equilibrium_failure
    exit 0
fi

wout=$case_root/wout_${case_name}.nc
proxy=$case_root/classifier_proxy
if ! test -d "$proxy/design"; then
    python3 "$CODE_ROOT/generate_spatial_grid.py" \
        --wout "$wout" \
        --out "$proxy/design" \
        --surfaces 0.25 0.30 0.425 0.4875 0.55 0.80 \
        --ntheta 16 \
        --nzeta 16 \
        --nmu 9
fi

export OMP_NUM_THREADS=$ALLOCATED_CPUS
for surface in s0p25000 s0p30000 s0p42500 s0p48750 s0p55000 s0p80000; do
    if ! test -f "$proxy/surfaces/$surface/topology.npz"; then
        python3 "$CODE_ROOT/evaluate_spatial_surface.py" \
            --wout "$wout" \
            --design "$proxy/design/$surface" \
            --out "$proxy/surfaces/$surface" \
            --simple-executable "$SIMPLE_X" \
            --trace-time 0.02 \
            --timeout "${SPATIAL_TIMEOUT_SECONDS:-3600}"
    fi
done

python3 "$CODE_ROOT/evaluate_classifier_proxy.py" \
    --topology "$proxy/surfaces/s0p25000/topology.npz" \
    --out "$proxy/prompt.json"
topology_files=(
    "$proxy/surfaces/s0p30000/topology.npz"
    "$proxy/surfaces/s0p42500/topology.npz"
    "$proxy/surfaces/s0p48750/topology.npz"
    "$proxy/surfaces/s0p55000/topology.npz"
    "$proxy/surfaces/s0p80000/topology.npz"
)
python3 "$CODE_ROOT/evaluate_spatial_atlas.py" \
    --topology "${topology_files[@]}" \
    --classifier topology \
    --out "$proxy/late-topology.json" \
    --risk-out "$proxy/late-topology-risk.npz"
python3 "$CODE_ROOT/evaluate_spatial_atlas.py" \
    --topology "${topology_files[@]}" \
    --classifier jpar \
    --out "$proxy/late-jpar.json" \
    --risk-out "$proxy/late-jpar-risk.npz"
"$DESC_PYTHON" "$CODE_ROOT/reactor_proxy_calibration.py" desc-worker \
    --metric gamma_c \
    --wout "$wout" > "$proxy/gamma-c.json"
gamma_c=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["values"][0])' "$proxy/gamma-c.json")

trap - ERR
python3 "$CODE_ROOT/build_classifier_scbo_response.py" \
    --request "$case_root/request.json" \
    --topology "$proxy/late-topology.json" \
    --reference-topology "$REFERENCE_TOPOLOGY" \
    --jpar "$proxy/late-jpar.json" \
    --reference-jpar "$REFERENCE_JPAR" \
    --prompt "$proxy/prompt.json" \
    --reference-prompt "$REFERENCE_PROMPT" \
    --gamma-c "$gamma_c" \
    --reference-gamma-c "$REFERENCE_GAMMA_C" \
    --prompt-tolerance "${PROMPT_TOLERANCE:-0.005}" \
    --out "$case_root/response.json"
