#!/bin/bash
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# Mirrors the small files of a remote run (the trainer's --out) into a local directory, to watch it with
# training/bin.monitor.py. Never copies a checkpoint or a snapshot: metrics, heartbeat, telemetry, samples, the
# ledger and the Slurm logs, a few MB.
#
#   bash training/monitor/pull.sh <host>:<remote out> <local dir> [every_s]       # every_s 0 = once (default 300)
#   SSH_OPTS="-p 40123 -i ~/.runpod/ssh/key" bash training/monitor/pull.sh root@1.2.3.4:/workspace/runs/x mirror/x
#
# <host> is anything ssh accepts: an alias of ~/.ssh/config (a cluster login behind a 12-hour certificate, a pod).
# The remote needs rsync. The JSONL files only grow, so they are appended in place and the monitor reads only the
# new lines; the JSON files are rewritten whole. <local dir>/.pulled holds the time of the last good pull: if pulls
# fail (the certificate expired, the network), the monitor says the mirror is stale instead of calling the run stalled.
set -uo pipefail
SRC=${1:?usage: pull.sh <host>:<remote out> <local dir> [every_s]}
DST=${2:?usage: pull.sh <host>:<remote out> <local dir> [every_s]}
EVERY=${3:-300}
mkdir -p "$DST"
RSH="ssh -o BatchMode=yes -o ConnectTimeout=20 ${SSH_OPTS:-}"
while true; do
    # 1. append-only JSONL (train_steps, metrics, bpb, samples, ledger, telemetry/<host>)
    rsync -rt --append-verify --inplace --timeout=120 -e "$RSH" \
        --include='/*.jsonl' --include='/telemetry/' --include='/telemetry/*.jsonl' --exclude='*' \
        "$SRC/" "$DST/" && ok1=1 || ok1=0
    # 2. rewritten files (status, heartbeat, the resume point) and the Slurm logs
    rsync -rt --timeout=120 -e "$RSH" \
        --include='/*.json' --include='/ALERT' --include='/last/' --include='/last/progress.json' \
        --include='/slurm/' --include='/slurm/*.out' --include='/chain/' --include='/chain/**' --exclude='*' \
        "$SRC/" "$DST/" && ok2=1 || ok2=0
    if [ $ok1 = 1 ] && [ $ok2 = 1 ]; then
        date +%s > "$DST/.pulled"
    else
        echo "[pull] $(date '+%F %T') failed (ssh or rsync); the monitor will report a stale mirror" >&2
    fi
    [ "$EVERY" = 0 ] && break
    sleep "$EVERY"
done
