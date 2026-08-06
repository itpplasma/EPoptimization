#!/bin/bash
set -Eeuo pipefail

index=${1:?candidate manifest index required}
wave_root=$CAMPAIGN_ROOT/$WAVE
line=$(sed -n "$((index + 1))p" "$wave_root/manifest.tsv")
IFS=$'\t' read -r case_name relative_input input_sha <<< "$line"
test -n "$case_name"
case_root=$wave_root/candidates/$case_name
input_name=$(basename "$relative_input")

failure_kind=worker_failure
failure_response() {
    if ! test -f "$case_root/response.json"; then
        python3 "$CODE_ROOT/build_barrier_scbo_response.py" \
            --request "$case_root/request.json" \
            --out "$case_root/response.json" \
            --failure-kind "$failure_kind"
    fi
}
trap failure_response ERR

test "$(sha256sum "$wave_root/candidates/$relative_input" | cut -d' ' -f1)" = "$input_sha"
test "$(sha256sum "$XVMEC" | cut -d' ' -f1)" = "$XVMEC_SHA256"
test "$(sha256sum "$SIMPLE_X" | cut -d' ' -f1)" = "$SIMPLE_SHA256"
test "$(sha256sum "$CODE_MANIFEST" | cut -d' ' -f1)" = "$CODE_MANIFEST_SHA256"
(cd "$CODE_ROOT" && sha256sum -c "$CODE_MANIFEST")

failure_kind=equilibrium_failure
set +e
(cd "$case_root" && timeout "${VMEC_TIMEOUT_SECONDS:-600}" "$XVMEC" "$input_name" > vmec.stdout 2>&1)
vmec_exit=$?
set -e
if test "$vmec_exit" -ne 0 || ! grep -q "EXECUTION TERMINATED NORMALLY" "$case_root/vmec.stdout"; then
    trap - ERR
    failure_response
    exit 0
fi

wout=$case_root/wout_${case_name}.nc
wout_sha=$(sha256sum "$wout" | cut -d' ' -f1)

failure_kind=geometry_failure
python3 "$CODE_ROOT/evaluate_vmec_geometry.py" \
    --wout "$wout" \
    --mirror-limit "${MIRROR_LIMIT:-0.20}" \
    --elongation-limit "${ELONGATION_LIMIT:-6.0}" \
    --out "$case_root/geometry.json"

failure_kind=barrier_overlap_failure
export OMP_NUM_THREADS=$ALLOCATED_CPUS
python3 "$CODE_ROOT/evaluate_barrier_overlap.py" \
    --wout "$wout" \
    --wout-sha256 "$wout_sha" \
    --out "$case_root/barrier_result" \
    --simple-executable "$SIMPLE_X" \
    --simple-sha256 "$SIMPLE_SHA256" \
    --surfaces "${SURFACES:-0.25,0.4,0.55,0.7}" \
    --ntheta "${NTHETA:-8}" \
    --nzeta "${NZETA:-8}" \
    --npitch "${NPITCH:-16}" \
    --mu-nodes "${MU_NODES:-24}" \
    --trace-time "${TRACE_TIME:-0.02}" \
    --prompt-time "${PROMPT_TIME:-0.001}" \
    --nturns "${NTURNS:-8}" \
    --seed "${PARTICLE_SEED:-12345}" \
    --trapped-width "${TRAPPED_WIDTH:-0.15}" \
    --mu-width-factor "${MU_WIDTH_FACTOR:-0.75}" \
    --jpar-temperature "${JPAR_TEMPERATURE:-0.1}" \
    --rotation-temperature "${ROTATION_TEMPERATURE:-0.02}" \
    --timeout "${SIMPLE_TIMEOUT_SECONDS:-3600}"

failure_kind=response_failure
python3 "$CODE_ROOT/build_barrier_scbo_response.py" \
    --request "$case_root/request.json" \
    --result "$case_root/barrier_result/result.json" \
    --geometry "$case_root/geometry.json" \
    --inner-surface "${INNER_SURFACE:-0.25}" \
    --outer-surface "${OUTER_SURFACE:-0.7}" \
    --particles-per-surface "$(( ${NTHETA:-8} * ${NZETA:-8} * ${NPITCH:-16} ))" \
    --prompt-limit "${PROMPT_LIMIT:-0.05}" \
    --objective "${OBJECTIVE:-barrier-jpar}" \
    --out "$case_root/response.json"
trap - ERR
