#!/usr/bin/env bash
# filepath: train.runpod.sh
# Lancia Pre-Train su RunPod da locale.
#
# Uso base:
#   bash train.runpod.sh --host root@66.92.198.50 --port 11639
#
# Con parametri training:
#   bash train.runpod.sh --host root@66.92.198.50 --port 11639 \
#       --preset medium --max_steps 50000 --bf16 --compile --wandb
#
# Tutti i parametri dopo -- vengono passati direttamente a train.py:
#   bash train.runpod.sh --host root@66.92.198.50 --port 11639 \
#       -- --preset large --batch_size 4 --grad_accum 8 --lr 1e-4
#
set -euo pipefail

# ── Defaults ──
RUNPOD_HOST=""
RUNPOD_PORT=""
LOCAL_DIR="$HOME/htdocs/sophia-core-server/model/skylar"
REMOTE_DIR="/workspace/skylar"

# S3 (default, sovrascrivibili)
S3_BUCKET="sophia-ai-dset"
S3_PREFIX="skylar/pretrain_v1"
S3_REGION="eu-south-1"

# AWS (legge da env locale se presenti)
AWS_KEY="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET="${AWS_SECRET_ACCESS_KEY:-}"

# Train defaults
PRESET="medium"
MAX_STEPS="50000"
BATCH_SIZE=""
GRAD_ACCUM=""
LR=""
SEQ_LEN=""
DROPOUT=""
WARMUP_STEPS=""
WEIGHT_DECAY=""
GRAD_CLIP=""
LR_SCHEDULE=""
LR_DECAY_RATIO=""
LOG_EVERY=""
EVAL_EVERY=""
SAVE_EVERY=""
SAMPLE_EVERY=""
SEED=""
NUM_WORKERS=""
RESUME=""
OUT_DIR="/workspace/checkpoints"
MUP_BASE=""
WANDB_PROJECT=""
WANDB_RUN=""

# Flags (default off)
BF16=false
FP16=false
COMPILE=false
WANDB=false
MULTI_GPU=false
GRAD_CKPT=false
NO_PACKING=false

# Extra args (passati direttamente a train.py)
EXTRA_ARGS=""

# ── Parse argomenti ──
while [[ $# -gt 0 ]]; do
    case $1 in
        # Connessione
        --host)             RUNPOD_HOST="$2"; shift 2 ;;
        --port)             RUNPOD_PORT="$2"; shift 2 ;;
        --dir)              LOCAL_DIR="$2"; shift 2 ;;
        --remote_dir)       REMOTE_DIR="$2"; shift 2 ;;

        # S3
        --s3_bucket)        S3_BUCKET="$2"; shift 2 ;;
        --s3_prefix)        S3_PREFIX="$2"; shift 2 ;;
        --s3_region)        S3_REGION="$2"; shift 2 ;;

        # AWS credentials
        --aws_key)          AWS_KEY="$2"; shift 2 ;;
        --aws_secret)       AWS_SECRET="$2"; shift 2 ;;

        # Model
        --preset)           PRESET="$2"; shift 2 ;;
        --seq_len)          SEQ_LEN="$2"; shift 2 ;;
        --dropout)          DROPOUT="$2"; shift 2 ;;
        --mup_base_d_model) MUP_BASE="$2"; shift 2 ;;

        # Training
        --max_steps)        MAX_STEPS="$2"; shift 2 ;;
        --batch_size)       BATCH_SIZE="$2"; shift 2 ;;
        --grad_accum)       GRAD_ACCUM="$2"; shift 2 ;;
        --lr)               LR="$2"; shift 2 ;;
        --warmup_steps)     WARMUP_STEPS="$2"; shift 2 ;;
        --weight_decay)     WEIGHT_DECAY="$2"; shift 2 ;;
        --grad_clip)        GRAD_CLIP="$2"; shift 2 ;;
        --lr_schedule)      LR_SCHEDULE="$2"; shift 2 ;;
        --lr_decay_ratio)   LR_DECAY_RATIO="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --num_workers)      NUM_WORKERS="$2"; shift 2 ;;
        --resume)           RESUME="$2"; shift 2 ;;
        --out_dir)          OUT_DIR="$2"; shift 2 ;;

        # Logging
        --log_every)        LOG_EVERY="$2"; shift 2 ;;
        --eval_every)       EVAL_EVERY="$2"; shift 2 ;;
        --save_every)       SAVE_EVERY="$2"; shift 2 ;;
        --sample_every)     SAMPLE_EVERY="$2"; shift 2 ;;

        # WandB
        --wandb)            WANDB=true; shift ;;
        --wandb_project)    WANDB_PROJECT="$2"; shift 2 ;;
        --wandb_run)        WANDB_RUN="$2"; shift 2 ;;

        # Flags
        --bf16)             BF16=true; shift ;;
        --fp16)             FP16=true; shift ;;
        --compile)          COMPILE=true; shift ;;
        --multi_gpu)        MULTI_GPU=true; shift ;;
        --gradient_checkpointing) GRAD_CKPT=true; shift ;;
        --no_packing)       NO_PACKING=true; shift ;;

        # Passthrough: tutto dopo -- va diritto a train.py
        --)                 shift; EXTRA_ARGS="$*"; break ;;

        *) echo "ERRORE: argomento sconosciuto: $1"; exit 1 ;;
    esac
done

# ── Validazione ──
if [[ -z "$RUNPOD_HOST" || -z "$RUNPOD_PORT" ]]; then
    echo "Uso: bash train.runpod.sh --host <user@ip> --port <port> [opzioni]"
    echo ""
    echo "  Connessione:"
    echo "    --host              Host SSH RunPod (es: root@66.92.198.50)"
    echo "    --port              Porta SSH RunPod (es: 11639)"
    echo "    --dir               Directory locale (default: \$HOME/htdocs/sophia-core-server/model/skylar)"
    echo ""
    echo "  Model:"
    echo "    --preset            test|small|small_plus|medium|large|1B|4b|8b|... (default: medium)"
    echo "    --seq_len           Override max sequence length"
    echo "    --bf16 / --fp16     Mixed precision"
    echo "    --compile           torch.compile()"
    echo ""
    echo "  Training:"
    echo "    --max_steps         (default: 50000)"
    echo "    --batch_size        (default: auto)"
    echo "    --grad_accum        (default: 4)"
    echo "    --lr                (default: 3e-4)"
    echo "    --lr_schedule       cosine|wsd"
    echo "    --warmup_steps      (default: 200)"
    echo "    --out_dir           Output directory (default: /workspace/checkpoints)"
    echo "    --wandb             Abilita WandB tracking"
    echo "    --multi_gpu         Multi-GPU con Accelerate"
    echo ""
    echo "  Passthrough:"
    echo "    -- <args>           Passa argomenti extra direttamente a train.py"
    exit 1
fi

SSH="ssh -p ${RUNPOD_PORT} ${RUNPOD_HOST}"
SCP="scp -P ${RUNPOD_PORT}"

# ── Costruisci train args ──
TRAIN_ARGS=""
TRAIN_ARGS+=" --s3_bucket ${S3_BUCKET}"
TRAIN_ARGS+=" --s3_prefix ${S3_PREFIX}"
TRAIN_ARGS+=" --s3_region ${S3_REGION}"
TRAIN_ARGS+=" --preset ${PRESET}"
TRAIN_ARGS+=" --max_steps ${MAX_STEPS}"
TRAIN_ARGS+=" --out_dir ${OUT_DIR}"

# Parametri opzionali (aggiungi solo se settati)
[[ -n "$BATCH_SIZE" ]]     && TRAIN_ARGS+=" --batch_size ${BATCH_SIZE}"
[[ -n "$GRAD_ACCUM" ]]     && TRAIN_ARGS+=" --grad_accum ${GRAD_ACCUM}"
[[ -n "$LR" ]]             && TRAIN_ARGS+=" --lr ${LR}"
[[ -n "$SEQ_LEN" ]]        && TRAIN_ARGS+=" --seq_len ${SEQ_LEN}"
[[ -n "$DROPOUT" ]]        && TRAIN_ARGS+=" --dropout ${DROPOUT}"
[[ -n "$WARMUP_STEPS" ]]   && TRAIN_ARGS+=" --warmup_steps ${WARMUP_STEPS}"
[[ -n "$WEIGHT_DECAY" ]]   && TRAIN_ARGS+=" --weight_decay ${WEIGHT_DECAY}"
[[ -n "$GRAD_CLIP" ]]      && TRAIN_ARGS+=" --grad_clip ${GRAD_CLIP}"
[[ -n "$LR_SCHEDULE" ]]    && TRAIN_ARGS+=" --lr_schedule ${LR_SCHEDULE}"
[[ -n "$LR_DECAY_RATIO" ]] && TRAIN_ARGS+=" --lr_decay_ratio ${LR_DECAY_RATIO}"
[[ -n "$LOG_EVERY" ]]      && TRAIN_ARGS+=" --log_every ${LOG_EVERY}"
[[ -n "$EVAL_EVERY" ]]     && TRAIN_ARGS+=" --eval_every ${EVAL_EVERY}"
[[ -n "$SAVE_EVERY" ]]     && TRAIN_ARGS+=" --save_every ${SAVE_EVERY}"
[[ -n "$SAMPLE_EVERY" ]]   && TRAIN_ARGS+=" --sample_every ${SAMPLE_EVERY}"
[[ -n "$SEED" ]]           && TRAIN_ARGS+=" --seed ${SEED}"
[[ -n "$NUM_WORKERS" ]]    && TRAIN_ARGS+=" --num_workers ${NUM_WORKERS}"
[[ -n "$RESUME" ]]         && TRAIN_ARGS+=" --resume ${RESUME}"
[[ -n "$MUP_BASE" ]]       && TRAIN_ARGS+=" --mup_base_d_model ${MUP_BASE}"
[[ -n "$WANDB_PROJECT" ]]  && TRAIN_ARGS+=" --wandb_project ${WANDB_PROJECT}"
[[ -n "$WANDB_RUN" ]]      && TRAIN_ARGS+=" --wandb_run ${WANDB_RUN}"

# Flags
$BF16       && TRAIN_ARGS+=" --bf16"
$FP16       && TRAIN_ARGS+=" --fp16"
$COMPILE    && TRAIN_ARGS+=" --compile"
$WANDB      && TRAIN_ARGS+=" --wandb"
$MULTI_GPU  && TRAIN_ARGS+=" --multi_gpu"
$GRAD_CKPT  && TRAIN_ARGS+=" --gradient_checkpointing"
$NO_PACKING && TRAIN_ARGS+=" --no-packing"

# Extra passthrough
[[ -n "$EXTRA_ARGS" ]] && TRAIN_ARGS+=" ${EXTRA_ARGS}"

echo "════════════════════════════════════════════════════════"
echo "  🚀 Skylar Pre-Train → RunPod"
echo "  Host:    ${RUNPOD_HOST}:${RUNPOD_PORT}"
echo "  Preset:  ${PRESET}"
echo "  Dir:     ${LOCAL_DIR}"
echo "  Remote:  ${REMOTE_DIR}"
echo "  Output:  ${OUT_DIR}"
echo "════════════════════════════════════════════════════════"

# ── Step 1: Connessione ──
echo ""
echo "[1/4] Verifica connessione..."
${SSH} "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || {
    echo "ERRORE: impossibile connettersi." && exit 1
}
echo "  ✓ Connesso"

# ── Step 2: Verifica file locali ──
echo ""
echo "[2/4] Verifica file locali..."

FILES=(
    "train.py"
    "train_sft.py"
    "chat_format.py"
    "config.py"
    "model.py"
    "chat.py"
    "generate.py"
    "requirements.txt"
    "./data/sft_train_mixed.jsonl"
)

for f in "${FILES[@]}"; do
    if [[ ! -f "${LOCAL_DIR}/${f}" ]]; then
        echo "  ✗ ERRORE: file non trovato: ${LOCAL_DIR}/${f}"
        exit 1
    fi
    echo "  ✓ ${f}"
done
echo "  OK!"

# ── Step 3: Copia file ──
echo ""
echo "[3/4] Copia file su RunPod..."

# Crea directory remota
${SSH} "mkdir -p ${REMOTE_DIR}"

# Copia tutti i file in un colpo solo
${SCP} ${FILES[@]/#/${LOCAL_DIR}/} "${RUNPOD_HOST}:${REMOTE_DIR}/"

echo "  ✓ ${#FILES[@]} file copiati in ${REMOTE_DIR}/"

# ── Validazione AWS ──
if [[ -z "$AWS_KEY" || -z "$AWS_SECRET" ]]; then
    echo ""
    echo "  ⚠  Credenziali AWS non trovate!"
    echo "     Passa --aws_key e --aws_secret, oppure esporta:"
    echo "     export AWS_ACCESS_KEY_ID=..."
    echo "     export AWS_SECRET_ACCESS_KEY=..."
    exit 1
fi

# ── Step 4: Setup + Pre Train ──
echo ""
echo "[4/4] Setup e avvio..."
echo "  Train args: ${TRAIN_ARGS}"
echo ""

${SSH} "bash -s" << REMOTE
set -euo pipefail
cd ${REMOTE_DIR}

echo "\$(date '+%H:%M:%S') | Installazione dipendenze..."
pip install -q --break-system-packages -r requirements.txt \
    2>&1 | tail -5

echo "\$(date '+%H:%M:%S') | Verifica ambiente..."
python3 -c "
import torch, tokenizers, transformers
n = torch.cuda.device_count()
vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(n))
print(f'  GPU: {n}x {torch.cuda.get_device_name(0)} | VRAM: {vram/1e9:.0f} GB')
print(f'  torch={torch.__version__} tokenizers={tokenizers.__version__} transformers={transformers.__version__}')
"

echo "\$(date '+%H:%M:%S') | Avvio pre-train..."

# AWS credentials per S3
export AWS_ACCESS_KEY_ID="${AWS_KEY}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET}"

MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=268435456 \
HF_HOME=/workspace/hf_cache \
PYTHONUNBUFFERED=1 nohup python3 train.py \
  ${TRAIN_ARGS} \
    > /workspace/train.log 2>&1 &

TRAIN_PID=\$!
disown

echo "\$(date '+%H:%M:%S') | PID: \${TRAIN_PID}"
echo ""

# Aspetta che il log esista e abbia contenuto
for i in {1..15}; do
    if [[ -s /workspace/train.log ]]; then break; fi
    sleep 2
done
echo "── Ultimi log ──"
tail -30 /workspace/train.log 2>/dev/null || echo "(log non ancora disponibile)"
REMOTE

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅ Pre-train avviato!"
echo ""
echo "  Monitora:"
echo "    ${SSH} 'tail -f /workspace/train.log'"
echo ""
echo "  Stato GPU:"
echo "    ${SSH} 'nvidia-smi'"
echo ""
echo "  Ferma:"
echo "    ${SSH} 'pkill -f train.py'"
echo ""
echo "  ⚠  SPEGNI il pod quando hai finito!"
echo "════════════════════════════════════════════════════════"