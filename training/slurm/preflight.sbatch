#!/bin/bash
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# Day-1 checks on a cluster, before any long job: bash training/slurm/submit.sh run.env preflight
# Runs on PREFLIGHT_NODES nodes (default 2) of the debug QOS for at most 30 minutes (~4 GPU hours) and
# writes everything to <out>/preflight/<job>/, with a one-line verdict per phase in SUMMARY.txt:
#   env       check_env.py on every node: versions, driver vs wheel, Triton, the KDA kernels, NCCL
#   gates     eval/bin.gate_arch_v2.py on one GPU (numerical equivalence of the kernels)
#   allreduce bandwidth inside one node and across all nodes; across nodes NCCL must use NET/IB
#   disk      write+fsync of a checkpoint-sized file, random 16 KiB reads from the shards (the loader)
#   train     the real model on all nodes, stopped by a deadline (saves last), then resumed on ONE node:
#             tok/s per GPU on 1 node vs N nodes = the first measured scaling efficiency
# PREFLIGHT_PHASES="env gates allreduce disk train" selects phases.
set -uo pipefail
: "${RUN_ENV:?}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck disable=SC1091
source "$REPO/training/slurm/lib.sh"
JOB=${SLURM_JOB_ID:?}
P="$OUT/preflight/$JOB"
mkdir -p "$P"
SUM="$P/SUMMARY.txt"
verdict() { echo "$1: $2" | tee -a "$SUM"; }
PHASES=${PREFLIGHT_PHASES:-"env gates allreduce disk train"}
[ -n "${MODULES:-}" ] && eval "$MODULES"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 PYTORCH_ALLOC_CONF=expandable_segments:True
# shellcheck disable=SC2086
[ -n "${NCCL_ENV:-}" ] && export $NCCL_ENV
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
N=$SLURM_JOB_NUM_NODES
log "preflight $JOB on $N nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
nvidia-smi > "$P/nvidia-smi.txt" 2>&1
python -c "import torch,sys; print(sys.version); print(torch.__version__, torch.version.cuda)" >> "$P/nvidia-smi.txt" 2>&1

if [[ " $PHASES " == *" env "* ]]; then
    srun --ntasks-per-node=1 bash -c "python '$REPO/training/slurm/check_env.py' --json '$P/env_\$(hostname).json' \
        > '$P/env_\$(hostname).log' 2>&1"
    bad=$(grep -L "ENV OK" "$P"/env_*.log 2>/dev/null | wc -l)
    [ "$bad" -eq 0 ] && verdict env "PASS on $N nodes" || verdict env "FAIL on $bad nodes (see env_*.log)"
fi

if [[ " $PHASES " == *" gates "* ]]; then
    (cd "$REPO" && srun -N1 -n1 python eval/bin.gate_arch_v2.py > "$P/gates.log" 2>&1) \
        && verdict gates "PASS ($(grep -c PASS "$P/gates.log") gates)" \
        || verdict gates "FAIL (see gates.log)"
fi

if [[ " $PHASES " == *" allreduce "* ]]; then
    AR="$REPO/training/slurm/allreduce_bench.py"
    srun -N1 --ntasks-per-node=1 bash -c "NCCL_DEBUG=INFO torchrun --nproc-per-node $GPUS_PER_NODE --standalone \
        '$AR' --out '$P/allreduce_1node.json'" > "$P/allreduce_1node.log" 2>&1
    if [ "$N" -gt 1 ]; then
        srun --ntasks-per-node=1 bash -c "NCCL_DEBUG=INFO torchrun --nnodes $N --nproc-per-node $GPUS_PER_NODE \
            --node-rank \$SLURM_NODEID --master-addr $MASTER --master-port 29600 '$AR' \
            --out '$P/allreduce_${N}nodes.json'" > "$P/allreduce_${N}nodes.log" 2>&1
        if grep -q "NET/IB" "$P/allreduce_${N}nodes.log"; then
            verdict allreduce "PASS: NET/IB between nodes; $(grep 'gradient all-reduce' "$P/allreduce_${N}nodes.log")"
        else
            verdict allreduce "FAIL: no NET/IB in the NCCL log ($(grep -o 'NET/[A-Za-z]*' "$P/allreduce_${N}nodes.log" \
                | sort -u | tr '\n' ' ')): fix NCCL_ENV before any multi-node run"
        fi
    else
        verdict allreduce "one node only: $(grep 'gradient all-reduce' "$P/allreduce_1node.log")"
    fi
fi

if [[ " $PHASES " == *" disk "* ]]; then
    python - "$P" "$DATA" > "$P/disk.log" 2>&1 <<'EOF'
import json, os, random, sys, time
p, data = sys.argv[1], sys.argv[2]
f = os.path.join(p, "disk_probe.bin")
buf = os.urandom(64 << 20)
t = time.time()
with open(f, "wb") as h:
    for _ in range(64):                      # 4 GiB, like a slice of a checkpoint
        h.write(buf)
    h.flush(); os.fsync(h.fileno())
w = 4 / (time.time() - t)
os.remove(f)
meta = json.load(open(os.path.join(data, "pretokenized_meta.json")))
shards = [os.path.join(data, "shards", s["filename"]) for s in meta["shards"]]
random.seed(0)
t, n = time.time(), 0
while time.time() - t < 20:
    s = random.choice(shards)
    with open(s, "rb") as h:
        h.seek(random.randrange(max(1, os.path.getsize(s) - 16384)))
        h.read(16384)
    n += 1
r = n / (time.time() - t)
print(json.dumps({"write_gib_s": round(w, 2), "random_16k_reads_s": round(r, 1)}))
EOF
    verdict disk "$(tail -1 "$P/disk.log") (the loader needs ~2 reads/s per GPU)"
fi

if [[ " $PHASES " == *" train "* ]]; then
    RUNP="$P/run"
    T1=${PREFLIGHT_TRAIN_MIN:-12}   # includes the first torch.compile (4-10 min on an A100 with an empty cache)
    T2=${PREFLIGHT_RESUME_MIN:-6}
    L1="cd '$REPO' && torchrun --nnodes $N --nproc-per-node $GPUS_PER_NODE --node-rank \$SLURM_NODEID \
        --master-addr $MASTER --master-port 29700 training/bin.pretrain.py $TRAIN_ARGS --data '$DATA' --out '$RUNP' \
        --deadline +${T1}m --stop_margin_min 1 --ckpt_every_min 2 --heartbeat_s 10"
    srun --ntasks-per-node=1 bash -c "$L1" > "$P/train_${N}nodes.log" 2>&1
    s1=$(python3 -c "import json;d=json.load(open('$RUNP/status.json'));print(d['state'],d['step'])" 2>/dev/null)
    tps1=$(python3 -c "import json;print(round(json.load(open('$RUNP/heartbeat.json'))['tok_s'] or 0))" 2>/dev/null)
    L2="cd '$REPO' && torchrun --nproc-per-node $GPUS_PER_NODE --standalone training/bin.pretrain.py $TRAIN_ARGS \
        --data '$DATA' --out '$RUNP' --resume auto --deadline +${T2}m --stop_margin_min 1 --heartbeat_s 10"
    srun -N1 --ntasks-per-node=1 bash -c "$L2" > "$P/train_1node_resume.log" 2>&1
    s2=$(python3 -c "import json;d=json.load(open('$RUNP/status.json'));print(d['state'],d['step'])" 2>/dev/null)
    tps2=$(python3 -c "import json;print(round(json.load(open('$RUNP/heartbeat.json'))['tok_s'] or 0))" 2>/dev/null)
    eff=$(python3 -c "a,b=$tps1/($N*$GPUS_PER_NODE),$tps2/$GPUS_PER_NODE; print(f'{a:.0f} vs {b:.0f} tok/s per GPU, efficiency {a/b:.0%}')" 2>/dev/null)
    if [[ "$s1" == deadline* ]] && [[ "$s2" == deadline* ]] && grep -q "\[resume\]" "$P/train_1node_resume.log"; then
        verdict train "PASS: $N nodes '$s1' -> 1 node '$s2'; $eff"
    else
        verdict train "FAIL: $N nodes '$s1', 1 node '$s2' (see train_*.log)"
    fi
fi
log "preflight done: $SUM"
cat "$SUM"
