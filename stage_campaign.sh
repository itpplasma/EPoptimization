#!/bin/bash
# Stage a campaign root: snapshot the code and the SIMPLE binary, pin their
# hashes, and write the environment the Slurm worker sources.
#
# A campaign must not read the live checkout or the live build tree. The worker
# verifies a code manifest and the executable hash on every candidate, so any
# edit or rebuild while a campaign runs fails every job in flight. Snapshotting
# makes a run reproducible and immune to work continuing in the repository.
set -Eeuo pipefail

root=${1:?campaign root required}
simple_x=${2:?path to simple.x required}
xvmec=${XVMEC:-/home/ert/runs/alpha_total_late_turbo_v1/toolchain/vmec2000/build-cluster/build/bin/xvmec}
code_src=${CODE_SRC:-$HOME/code/EPoptimization}

# Everything the worker executes or imports.
files=(
    barrier_overlap.py
    smooth_barrier.py
    evaluate_barrier_overlap.py
    build_barrier_scbo_response.py
    evaluate_vmec_geometry.py
    Alan_objectives.py
    raw_fourier_surface.py
    run_barrier_candidate.sh
    simple_direct.py
    validate_barrier_optimum.py
)

mkdir -p "$root/code" "$root/toolchain"
for file in "${files[@]}"; do
    cp "$code_src/$file" "$root/code/$file"
done
cp "$simple_x" "$root/toolchain/simple.x"
chmod +x "$root/toolchain/simple.x" "$root/code/run_barrier_candidate.sh"

(cd "$root/code" && sha256sum "${files[@]}" > "$root/code-manifest.sha256")
sha256sum "$root/code-manifest.sha256" | cut -d' ' -f1 > "$root/code-manifest.sha256.digest"

cat > "$root/env.sh" <<EOF
export PATH=${WORKER_VENV:-/home/ert/runs/alpha-direct-runtime/worker-venv}/bin:\$PATH
export XVMEC=$xvmec
export XVMEC_SHA256=$(sha256sum "$xvmec" | cut -d' ' -f1)
export SIMPLE_X=$root/toolchain/simple.x
export SIMPLE_SHA256=$(sha256sum "$root/toolchain/simple.x" | cut -d' ' -f1)
export CODE_ROOT=$root/code
export CODE_MANIFEST=$root/code-manifest.sha256
export CODE_MANIFEST_SHA256=$(cat "$root/code-manifest.sha256.digest")
export CAMPAIGN_ROOT=$root/campaign
export INNER_SURFACE=${INNER_SURFACE:-0.25}
export OUTER_SURFACE=${OUTER_SURFACE:-0.6}
export NTHETA=${NTHETA:-8}
export NZETA=${NZETA:-8}
export NPITCH=${NPITCH:-16}
export MU_BINS=${MU_BINS:-16}
export TRACE_TIME=${TRACE_TIME:-0.02}
export PROMPT_TIME=${PROMPT_TIME:-0.001}
export PROMPT_LIMIT=${PROMPT_LIMIT:-0.25}
export NTURNS=${NTURNS:-8}
export PARTICLE_SEED=${PARTICLE_SEED:-12345}
export CLASSIFIER=${CLASSIFIER:-topology}
export OBJECTIVE=${OBJECTIVE:-discrete}
export SMOOTH_CHAOS_WIDTH=${SMOOTH_CHAOS_WIDTH:-0.25}
export SMOOTH_TRAPPED_WIDTH=${SMOOTH_TRAPPED_WIDTH:-0.15}
export SMOOTH_BIN_WIDTH=${SMOOTH_BIN_WIDTH:-0.5}
export SIMPLE_TIMEOUT_SECONDS=${SIMPLE_TIMEOUT_SECONDS:-3600}
EOF

cp "$code_src/run_single.sbatch" "$root/run_single.sbatch"
echo "staged $root"
echo "  simple.x $(sha256sum "$root/toolchain/simple.x" | cut -c1-12)"
echo "  code     $(cat "$root/code-manifest.sha256.digest" | cut -c1-12)"
echo "  objective ${OBJECTIVE:-discrete}"
