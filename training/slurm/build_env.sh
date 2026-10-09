#!/bin/bash
# =================================================================
# @copyright: A. Ivanovitch | CEO SKYL4R | 2026
# =================================================================
# The training environment, pinned to constraints.txt, on a cluster where compute nodes may have no
# internet and login nodes may limit CPU time. Three ways:
#
#   bash training/slurm/build_env.sh online   <venv> [cu128|cu126]            # one step, needs internet
#   bash training/slurm/build_env.sh download <wheelhouse> [cu128|cu126] [3.11] # where there is internet
#   bash training/slurm/build_env.sh install  <venv> <wheelhouse>              # no network needed
#
# `download` can run on any Linux x86_64 machine: it fetches wheels for the cluster's Python version
# (third argument, e.g. 3.11), then the wheelhouse is copied over (rsync/scp to a data-transfer node).
# cu128 is the validated build. cu126 is the fallback when the driver is too old for it: check_env.py says
# which one works. Both are torch 2.10.0; torch 2.10 has no cu124 build.
# Triton compiles with its own ptxas; on an old driver, point it at the cluster's (module load cuda/...):
#   export TRITON_PTXAS_PATH=$(which ptxas)
set -euo pipefail
# a venv with pip; where python3 has no venv module (some images), uv makes one
mkvenv() { python3 -m venv "$1" 2>/dev/null || { command -v uv >/dev/null && uv venv -q --seed "$1"; } \
           || { echo "cannot create a venv: python3 -m venv failed and uv is missing"; exit 2; }; }
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MODE=${1:?usage: build_env.sh online|download|install ...}
CU=cu128
PINS="$ROOT/constraints.txt"
# what the training imports, beyond the pins: the framework itself is installed from the checkout (-e)
REQS=(torch triton transformers tokenizers accelerate numpy safetensors huggingface_hub
      flash-linear-attention fla-core einops)

case "$MODE" in
  online)
    VENV=${2:?venv dir}; CU=${3:-cu128}
    mkvenv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    "$VENV/bin/pip" install -q torch==2.10.0 --index-url "https://download.pytorch.org/whl/$CU"
    "$VENV/bin/pip" install -q -c "$PINS" "${REQS[@]}"
    "$VENV/bin/pip" install -q --no-deps -e "$ROOT"
    echo "environment ready in $VENV ($CU). Check it on a GPU node: $VENV/bin/python $ROOT/training/slurm/check_env.py"
    ;;
  download)
    WH=${2:?wheelhouse dir}; CU=${3:-cu128}; PYV=${4:-}
    mkdir -p "$WH"
    plat=()
    if [ -n "$PYV" ]; then
        plat=(--python-version "$PYV" --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64
              --platform manylinux_2_17_x86_64 --only-binary=:all:)
    fi
    pip download -q -d "$WH" "${plat[@]}" torch==2.10.0 --index-url "https://download.pytorch.org/whl/$CU" \
        --extra-index-url https://pypi.org/simple
    pip download -q -d "$WH" "${plat[@]}" -c "$PINS" "${REQS[@]}" pip setuptools wheel
    echo "$CU" > "$WH/CUDA_BUILD"
    (cd "$WH" && sha256sum ./*.whl > SHA256SUMS)
    echo "wheelhouse $WH: $(ls "$WH"/*.whl | wc -l) wheels, $(du -sh "$WH" | cut -f1), torch build $CU"
    ;;
  install)
    VENV=${2:?venv dir}; WH=${3:?wheelhouse dir}
    (cd "$WH" && sha256sum -c --quiet SHA256SUMS)
    mkvenv "$VENV"
    "$VENV/bin/pip" install -q --no-index --find-links "$WH" --upgrade pip
    "$VENV/bin/pip" install -q --no-index --find-links "$WH" -c "$PINS" "${REQS[@]}" setuptools wheel
    "$VENV/bin/pip" install -q --no-index --no-deps --no-build-isolation -e "$ROOT" \
        || echo "note: framework not installed with -e; scripts still run from the checkout (sys.path)"
    echo "environment ready in $VENV ($(cat "$WH/CUDA_BUILD")). Check it on a GPU node: $VENV/bin/python $ROOT/training/slurm/check_env.py"
    ;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
