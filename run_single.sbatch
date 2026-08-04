#!/bin/bash
#SBATCH --job-name=alpha-barrier
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
set -Eeuo pipefail

# One optimizer candidate. The driver exports METHOD, EVAL and METRIC.
: "${METHOD:?METHOD is required}"
: "${EVAL:?EVAL is required}"
METRIC=${METRIC:-barrier-overlap}

export CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-$SLURM_SUBMIT_DIR/campaign}
export WAVE="$METHOD/eval-$EVAL"
export CODE_ROOT=${CODE_ROOT:-$HOME/code/EPoptimization}
export ALLOCATED_CPUS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$ALLOCATED_CPUS

export XVMEC=${XVMEC:?XVMEC is required}
export XVMEC_SHA256=${XVMEC_SHA256:?XVMEC_SHA256 is required}
export SIMPLE_X=${SIMPLE_X:?SIMPLE_X is required}
export SIMPLE_SHA256=${SIMPLE_SHA256:?SIMPLE_SHA256 is required}
export CODE_MANIFEST=${CODE_MANIFEST:?CODE_MANIFEST is required}
export CODE_MANIFEST_SHA256=${CODE_MANIFEST_SHA256:?CODE_MANIFEST_SHA256 is required}

case "$METRIC" in
    barrier-overlap) worker=run_barrier_candidate.sh ;;
    direct-loss)     worker=run_direct_scbo_candidate.sh ;;
    *) echo "unknown METRIC: $METRIC" >&2; exit 2 ;;
esac

bash "$CODE_ROOT/$worker" 0
