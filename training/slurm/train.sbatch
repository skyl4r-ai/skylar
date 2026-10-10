#!/bin/bash
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# One link of a chained pretraining run on Slurm. Submit with `bash training/slurm/submit.sh run.env`:
# the #SBATCH options come from the run file, so there are none here.
#
# What one job does, in order:
#   1. stops at once if the run is done, over budget, or stuck in a crash loop (ledger.py --can-chain);
#   2. queues the next link now (--dependency=afterany: it starts when this one ends, in any state);
#   3. trains with --resume auto and a deadline: the trainer saves <out>/last and exits STOP_MARGIN_MIN
#      before the time limit, and every CKPT_EVERY_MIN in between;
#   4. runs a watchdog: a heartbeat older than WATCHDOG_STALL_MIN (a hung collective, a dead node) kills
#      the training step instead of letting it burn the allocation until the time limit;
#   5. appends its line to <out>/ledger.jsonl: nodes, wall time, GPU hours, steps and tokens done.
set -uo pipefail
: "${RUN_ENV:?RUN_ENV not set: submit with training/slurm/submit.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck disable=SC1091
source "$REPO/training/slurm/lib.sh"

JOB=${SLURM_JOB_ID:?}
JOB_START=$(date +%s)
mkdir -p "$OUT/slurm" "$OUT/chain"
echo "$JOB" >> "$OUT/chain/jobs.txt"
log "job $JOB on $SLURM_JOB_NUM_NODES nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | tr '\n' ' ')"
# where this job starts from: the ledger uses it for the progress of a job killed before its own line
read -r STEP0 TOK0 < <(resume_point)
echo "$STEP0 $TOK0" > "$OUT/chain/start.$JOB"

# 1. may this job run at all?
if [ "$(run_state)" = "done" ]; then
    log "the run is done: nothing to do"
    exit 0
fi
MAX_GPU_H=$(job_max_gpu_h)
if ! verdict=$(python3 "$REPO/training/slurm/ledger.py" "$OUT" --backfill "$JOB" --gpus-per-node "$GPUS_PER_NODE" \
                --can-chain --budget "${BUDGET_GPU_H:-0}" \
                --next-gpu-h "$MAX_GPU_H" --max-stalled "${MAX_STALLED:-2}" 2>&1); then
    log "NOT RUNNING: $verdict"
    echo "$(date '+%F %T') job $JOB not run: $verdict" >> "$OUT/ALERT"
    exit 0
fi
log "ledger: $verdict"

# 2. queue the next link now, so its wait in the queue overlaps this job
n_jobs=$(wc -l < "$OUT/chain/jobs.txt")
if [ "$n_jobs" -lt "${CHAIN_MAX:-40}" ]; then
    if NEXT=$(submit_job --dependency "afterany:$JOB"); then
        echo "$NEXT" > "$OUT/chain/next"
        log "next link queued: $NEXT"
    else
        log "could not queue the next link"
        echo "$(date '+%F %T') job $JOB: sbatch of the next link failed" >> "$OUT/ALERT"
    fi
else
    log "chain at CHAIN_MAX=$CHAIN_MAX: no next link"
fi

# 3. environment: modules, venv, caches, network, offline
[ -n "${MODULES:-}" ] && eval "$MODULES"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTORCH_ALLOC_CONF=expandable_segments:True
# shellcheck disable=SC2086
[ -n "${NCCL_ENV:-}" ] && export $NCCL_ENV
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
LEFT=$(squeue -h -j "$JOB" -o %L)
# the job's end as a unix time; empty when Slurm has no limit for it (%L = UNLIMITED/INVALID): then no deadline
DEADLINE=$(python3 -c "
import sys; s=sys.argv[1].strip(); d=0
try:
    if '-' in s: d, s = s.split('-', 1); d = int(d)
    p=[int(x) for x in s.split(':')]; p=[0]*(3-len(p))+p
    print(int(sys.argv[2]) + d*86400 + p[0]*3600 + p[1]*60 + p[2])
except ValueError:
    pass" "$LEFT" "$(date +%s)")
DEADLINE_ARG=""
[ -n "$DEADLINE" ] && DEADLINE_ARG="--deadline $DEADLINE --stop_margin_min ${STOP_MARGIN_MIN:-15}"
log "resume point: step $STEP0, $(python3 -c "print(f'{$TOK0/1e9:.3f}')")B tokens; time left $LEFT"

# 4. train (one torchrun per node), with the watchdog on the side
LAUNCH="cd '$REPO' && exec torchrun --nnodes $SLURM_JOB_NUM_NODES --nproc-per-node $GPUS_PER_NODE \
  --node-rank \$SLURM_NODEID --master-addr $MASTER --master-port ${MASTER_PORT:-29500} \
  training/bin.pretrain.py $TRAIN_ARGS --data '$DATA' --out '$OUT' --resume auto \
  $DEADLINE_ARG --ckpt_every_min ${CKPT_EVERY_MIN:-30}"
# scancel (or a preemption) sends SIGTERM to this script too, not only to the trainer: untrapped, bash would die here
# and the job would leave no ledger line. The trainer saves <out>/last on SIGTERM: wait for it, then write the line.
trap 'log "SIGTERM: waiting for the trainer to save last, then the ledger line"' TERM
srun --kill-on-bad-exit=1 --ntasks-per-node=1 --cpus-per-task="$CPUS_PER_TASK" bash -c "$LAUNCH" &
SRUN=$!
bash "$REPO/training/slurm/watchdog.sh" "$OUT" "$JOB_START" "${WATCHDOG_GRACE_MIN:-30}" \
     "${WATCHDOG_STALL_MIN:-15}" "$SRUN" "$JOB" &
WD=$!
wait "$SRUN"
RC=$?
while kill -0 "$SRUN" 2>/dev/null; do wait "$SRUN"; RC=$?; done    # a trapped signal ends the wait, not srun
kill "$WD" 2>/dev/null
wait "$WD" 2>/dev/null

# 5. the ledger line
END=$(date +%s)
read -r STEP1 TOK1 < <(resume_point)
STATE=$(run_state)
[ -f "$OUT/chain/stall.$JOB" ] && STATE="stalled"
python3 - "$OUT/ledger.jsonl" <<EOF
import json, sys
rec = {"job": "$JOB", "nodes": $SLURM_JOB_NUM_NODES, "gpus": $SLURM_JOB_NUM_NODES * $GPUS_PER_NODE,
       "start": $JOB_START, "end": $END, "elapsed_s": $END - $JOB_START,
       "gpu_h": round(($END - $JOB_START) * $SLURM_JOB_NUM_NODES * $GPUS_PER_NODE / 3600, 3),
       "step_start": $STEP0, "step_end": $STEP1, "tokens_start": $TOK0, "tokens_end": $TOK1,
       "state": "$STATE", "exit": $RC}
open(sys.argv[1], "a").write(json.dumps(rec) + "\n")
EOF
log "job $JOB ended: exit $RC, state '$STATE', step $STEP0 -> $STEP1"
python3 "$REPO/training/slurm/ledger.py" "$OUT" | tail -4

# the run is finished: the queued link has nothing to do
if [ "$STATE" = "done" ] && [ -f "$OUT/chain/next" ]; then
    scancel "$(cat "$OUT/chain/next")" 2>/dev/null && log "run done: cancelled the queued link"
fi
exit 0
