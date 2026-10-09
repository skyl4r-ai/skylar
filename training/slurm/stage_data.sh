#!/bin/bash
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# Puts a tokenized corpus where the training reads it, and proves every byte arrived.
#
#   bash training/slurm/stage_data.sh hf <hf://buckets/org/name/path> <dst> [parallel]
#       from a Hugging Face bucket (needs the `hf` CLI, huggingface_hub >= 1.18, and a token);
#   bash training/slurm/stage_data.sh verify <dst> [parallel]
#       only the check, e.g. after an rsync from another machine.
#
# Resumable: a shard already there with the right sha256 is not downloaded again; a broken or partial one
# is. The corpus format is the one the trainer reads: pretokenized_meta.json, tokenizer.json,
# checksums.sha256 ("<sha256>  shards/<file>") and shards/. Writes <dst>/STAGED with the counts when complete.
# On a cluster with a CPU-time limit on login nodes, run it on a data-transfer node or as a serial job.
set -uo pipefail
MODE=${1:?usage: stage_data.sh hf <bucket_url> <dst> [parallel] | verify <dst> [parallel]}
if [ "$MODE" = hf ]; then SRC=${2:?bucket url}; DST=${3:?destination}; P=${4:-8}
elif [ "$MODE" = verify ]; then DST=${2:?destination}; P=${3:-8}
else echo "unknown mode $MODE"; exit 2; fi
HF=${HF:-hf}
mkdir -p "$DST/shards"
cd "$DST" || exit 2

bad_list() {   # shards whose sha256 is wrong or missing -> stdout
    grep " shards/" checksums.sha256 | xargs -P "$P" -L 200 sh -c '
        for i in $(seq 1 2 $#); do eval h=\${$i}; eval f=\${$((i+1))}
            if [ ! -s "$f" ] || [ "$(sha256sum "$f" | cut -d" " -f1)" != "$h" ]; then echo "$f"; fi
        done' sh
}

if [ "$MODE" = hf ]; then
    for f in pretokenized_meta.json tokenizer.json checksums.sha256; do
        [ -s "$f" ] || "$HF" buckets cp "$SRC/$f" "$f" >/dev/null || { echo "cannot download $f"; exit 1; }
    done
    for attempt in 1 2 3; do
        bad_list > .todo
        n=$(wc -l < .todo)
        [ "$n" -eq 0 ] && break
        echo "attempt $attempt: $n shards to download"
        < .todo xargs -P "$P" -I{} sh -c 'rm -f "{}"; "'"$HF"'" buckets cp "'"$SRC"'/{}" "{}" >/dev/null 2>&1 || echo "failed {}"'
    done
fi

bad_list > .todo
n_bad=$(wc -l < .todo)
n_all=$(grep -c " shards/" checksums.sha256)
if [ "$n_bad" -eq 0 ]; then
    python3 - <<'EOF'
import json
m = json.load(open("pretokenized_meta.json"))
print(f"complete: {len(m['shards'])} shards, {m['total_tokens'] / 1e9:.2f}B tokens, every sha256 checked")
EOF
    echo "$(date '+%F %T') $n_all shards verified" > STAGED
    rm -f .todo
else
    echo "INCOMPLETE: $n_bad of $n_all shards missing or corrupt (list in $DST/.todo); run again"
    exit 1
fi
