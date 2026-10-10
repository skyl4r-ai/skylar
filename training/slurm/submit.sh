#!/bin/bash
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# Starts (or restarts) a chained run:  bash training/slurm/submit.sh my_run.env
# Day-1 checks instead (one job, debug QOS, no chain): bash training/slurm/submit.sh my_run.env preflight
# One command in the run's environment (SFT, merge, COBOLEval...), one node:
#   bash training/slurm/submit.sh my_run.env one <time> <gpus> -- <command...>
# Every later link is queued by the running job itself (train.sbatch). Safe to call on a run that
# stopped: it resumes from <out>/last. Status at any time: python training/slurm/ledger.py <out>
set -euo pipefail
RUN_ENV=$(realpath "${1:?usage: submit.sh <run.env> [preflight | one <time> <gpus> -- <command...>]}")
MODE=${2:-chain}
export RUN_ENV
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck disable=SC1091
source "$REPO/training/slurm/lib.sh"
if [ "$MODE" = "one" ]; then
    TIME=${3:?time, e.g. 08:00:00}; GPUS_PER_NODE=${4:?gpus}; shift 4
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || { echo "no command after --"; exit 2; }
    NODES=1 CPUS_PER_TASK=$(( 8 * GPUS_PER_NODE ))
    mkdir -p "$OUT/slurm"
    ONE_SCRIPT="$OUT/slurm/one.$(date +%Y%m%d_%H%M%S).sh"
    printf '%q ' "$@" > "$ONE_SCRIPT"; echo >> "$ONE_SCRIPT"
    EXPORT_EXTRA="ONE_SCRIPT=$ONE_SCRIPT" SBATCH_SCRIPT=one.sbatch JOB_NAME="$JOB_NAME-one"
    JOB=$(submit_job)
    log "submitted $JOB: 1 node, $GPUS_PER_NODE GPUs, $TIME: $(cat "$ONE_SCRIPT")"
    exit 0
fi
for v in REPO VENV OUT DATA TRAIN_ARGS NODES GPUS_PER_NODE TIME; do
    [ -n "${!v:-}" ] || { echo "missing $v in $RUN_ENV"; exit 2; }
done
[ -f "$DATA/pretokenized_meta.json" ] || { echo "no pretokenized_meta.json in $DATA"; exit 2; }
mkdir -p "$OUT/slurm" "$OUT/chain"
if [ "$MODE" = "preflight" ]; then
    NODES=${PREFLIGHT_NODES:-2} TIME=${PREFLIGHT_TIME:-00:30:00} QOS=${PREFLIGHT_QOS:-boost_qos_dbg}
    SBATCH_SCRIPT=preflight.sbatch JOB_NAME="$JOB_NAME-preflight"
    JOB=$(submit_job)
    log "preflight submitted: $JOB ($NODES nodes, $TIME). Verdicts in $OUT/preflight/$JOB/SUMMARY.txt"
    exit 0
fi
cp "$RUN_ENV" "$OUT/chain/run.env.$(date +%Y%m%d_%H%M%S)"     # what each submission ran with
rm -f "$OUT/ALERT"
if [ -f "$OUT/chain/next" ] && squeue -h -j "$(cat "$OUT/chain/next")" 2>/dev/null | grep -q .; then
    echo "a link is already queued: $(cat "$OUT/chain/next") (scancel it first to resubmit)"
    exit 1
fi
JOB=$(submit_job)
echo "$JOB" > "$OUT/chain/next"
log "submitted $JOB: $NODES nodes x $GPUS_PER_NODE GPUs, $TIME, up to $(job_max_gpu_h) GPU-h per job"
