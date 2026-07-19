#!/usr/bin/env bash

# Source this file before running inference optimization benchmarks.
# By default it tries /data/$USER/cache, then falls back to this project.
# You can override it before sourcing:
#   export INFERENCE_OPT_CACHE_ROOT=/data/your_name/cache

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CACHE_ROOT="/data/${USER}/cache"
export INFERENCE_OPT_CACHE_ROOT="${INFERENCE_OPT_CACHE_ROOT:-${TOKEN_COMPRESSION_CACHE_ROOT:-$DEFAULT_CACHE_ROOT}}"

if ! mkdir -p "$INFERENCE_OPT_CACHE_ROOT" 2>/dev/null; then
  export INFERENCE_OPT_CACHE_ROOT="$SCRIPT_DIR/.cache"
  mkdir -p "$INFERENCE_OPT_CACHE_ROOT"
fi

export TOKEN_COMPRESSION_CACHE_ROOT="$INFERENCE_OPT_CACHE_ROOT"
export HF_HOME="$INFERENCE_OPT_CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$INFERENCE_OPT_CACHE_ROOT/huggingface/hub"
export HF_DATASETS_CACHE="$INFERENCE_OPT_CACHE_ROOT/huggingface/datasets"
export TRANSFORMERS_CACHE="$INFERENCE_OPT_CACHE_ROOT/huggingface/transformers"
export NLTK_DATA="$INFERENCE_OPT_CACHE_ROOT/nltk"
export XDG_CACHE_HOME="$INFERENCE_OPT_CACHE_ROOT"
export TMPDIR="${INFERENCE_OPT_TMPDIR:-${TOKEN_COMPRESSION_TMPDIR:-$INFERENCE_OPT_CACHE_ROOT/tmp}}"
export PIP_CACHE_DIR="$INFERENCE_OPT_CACHE_ROOT/pip"

mkdir -p "$HF_HOME" \
         "$HF_HUB_CACHE" \
         "$HF_DATASETS_CACHE" \
         "$TRANSFORMERS_CACHE" \
         "$NLTK_DATA" \
         "$XDG_CACHE_HOME" \
         "$TMPDIR" \
         "$PIP_CACHE_DIR" 2>/dev/null || true
