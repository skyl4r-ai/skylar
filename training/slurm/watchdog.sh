#!/bin/bash
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# Kills a hung training step. A dead node or a stuck collective freezes every rank, and the job would sit
# on all its GPUs until the time limit: on 32 nodes that is 128 GPU hours per hour, for nothing. Rank 0
# rewrites <out>/heartbeat.json every few seconds while it trains; when it stops changing, this kills the
# step, the job ends, and the next link of the chain resumes from <out>/last.
#
#   watchdog.sh <out> <job_start_unix> <grace_min> <stall_min> <pid_to_kill> [slurm_job_id]
# grace: allowed before the first training heartbeat of this job (startup, data, torch.compile);
# stall: allowed between heartbeats afterwards (eval, checkpoint saves included).
OUT=$1 JOB_START=$2 PID=$5 JOB=${6:-}
GRACE=$(LC_ALL=C awk -v m="$3" 'BEGIN{print int(m*60)}') STALL=$(LC_ALL=C awk -v m="$4" 'BEGIN{print int(m*60)}')   # minutes may be fractional
HB="$OUT/heartbeat.json"
POLL=${WATCHDOG_POLL_S:-30}

while kill -0 "$PID" 2>/dev/null; do
    sleep "$POLL"
    now=$(date +%s)
    if [ -f "$HB" ] && [ "$(stat -c %Y "$HB")" -ge "$JOB_START" ]; then
        ref=$(stat -c %Y "$HB")
        if grep -q '"state": "starting"' "$HB"; then limit=$GRACE; else limit=$STALL; fi
    else
        ref=$JOB_START; limit=$GRACE
    fi
    age=$(( now - ref ))
    if [ "$age" -gt "$limit" ]; then
        msg="$(date '+%F %T') watchdog: heartbeat ${age}s old (limit ${limit}s), killing the training step"
        echo "$msg"
        mkdir -p "$OUT/chain"
        echo "$msg" > "$OUT/chain/stall.${JOB:-local}"
        echo "$msg (job ${JOB:-local})" >> "$OUT/ALERT"
        kill -TERM "$PID" 2>/dev/null
        for _ in $(seq 1 12); do kill -0 "$PID" 2>/dev/null || exit 0; sleep 5; done
        kill -KILL "$PID" 2>/dev/null
        [ -n "$JOB" ] && scancel --signal=KILL "${JOB}.0" 2>/dev/null
        exit 0
    fi
done
