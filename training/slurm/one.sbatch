#!/bin/bash
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# One job, one command, the run's environment: SFT, preference training, GRPO, the WSM merge, COBOLEval.
# Submit with submit.sh, which writes the command to a file and passes it here:
#   bash training/slurm/submit.sh run.env one 08:00:00 1 -- python training/bin.sft.py ...
# No chain and no deadline: post-training jobs of the 990M fit in one job (SFT is hours on one GPU).
# Not in the chain's ledger (ledger.py): its GPU hours are in sacct, count them against the allocation.
set -uo pipefail
: "${RUN_ENV:?RUN_ENV not set: submit with training/slurm/submit.sh}"
: "${ONE_SCRIPT:?ONE_SCRIPT not set: submit with training/slurm/submit.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck disable=SC1091
source "$REPO/training/slurm/lib.sh"
log "job ${SLURM_JOB_ID:-?} on $(hostname): $(tr '\n' ' ' < "$ONE_SCRIPT")"

# the environment of train.sbatch (step 3)
[ -n "${MODULES:-}" ] && eval "$MODULES"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True
# GnuCOBOL of build_gnucobol.sh, for COBOLEval and the COBOL rewards
if [ -n "${GNUCOBOL:-}" ]; then
    export COBC="$GNUCOBOL/bin/cobc" COB_CC="$GNUCOBOL/bin/x86_64-conda-linux-gnu-cc"
fi
cd "$REPO" || exit 2
T0=$(date +%s)
bash "$ONE_SCRIPT"
RC=$?
log "exit $RC after $(( ($(date +%s) - T0) / 60 )) min"
exit $RC
