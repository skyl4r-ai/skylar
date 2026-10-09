# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# Shared by submit.sh and train.sbatch. Needs the run file sourced (RUN_ENV).

log() { echo "[$(date '+%F %T')] [chain] $*"; }

# GPU hours one job of this run may use at most (its wall time x its GPUs)
job_max_gpu_h() {
    local t=$TIME d=0 h m s
    [[ $t == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r h m s <<< "$t"
    LC_ALL=C awk -v sec="$(( d * 86400 + 10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${s:-0} ))" -v g="$(( NODES * GPUS_PER_NODE ))" \
        'BEGIN{printf "%.2f\n", sec * g / 3600}'
}

# submit_job [extra sbatch args...] -> prints the job id
submit_job() {
    local args=(--parsable --job-name "$JOB_NAME" --nodes "$NODES" --ntasks-per-node 1
                --gres "gpu:$GPUS_PER_NODE" --cpus-per-task "$CPUS_PER_TASK" --time "$TIME"
                --output "$OUT/slurm/%j.out" --export "ALL,RUN_ENV=$RUN_ENV")
    [ -n "${ACCOUNT:-}" ] && args+=(--account "$ACCOUNT")
    [ -n "${PARTITION:-}" ] && args+=(--partition "$PARTITION")
    [ -n "${QOS:-}" ] && args+=(--qos "$QOS")
    [ -n "${EXCLUDE:-}" ] && args+=(--exclude "$EXCLUDE")
    # shellcheck disable=SC2086
    sbatch "${args[@]}" ${EXTRA_SBATCH:-} "$@" "$REPO/training/slurm/${SBATCH_SCRIPT:-train.sbatch}"
}

# step/tokens of the resume point (<out>/last/progress.json), "0 0" if none
resume_point() {
    local f="$OUT/last/progress.json"
    [ -f "$f" ] || f="$OUT/last.prev/progress.json"
    if [ -f "$f" ]; then
        python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('step',0), d.get('tokens',0))" "$f"
    else
        echo "0 0"
    fi
}

run_state() {   # state field of <out>/status.json ("" if none)
    [ -f "$OUT/status.json" ] || { echo ""; return; }
    python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('state',''))" "$OUT/status.json"
}
