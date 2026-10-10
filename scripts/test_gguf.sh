#!/usr/bin/env bash
# test_gguf.sh — valida o GGUF com o llama.cpp e gera uma amostra de texto.
# Uso: LLAMA_CPP_DIR=/tmp/llama.cpp bash scripts/test_gguf.sh models/gpt-machado-q8_0.gguf
set -euo pipefail

MODEL="${1:?uso: test_gguf.sh <modelo.gguf>}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/tmp/llama.cpp}"
BIN="$LLAMA_CPP_DIR/build/bin/llama-cli"

"$BIN" -m "$MODEL" \
  -p "— Capitu, você me" \
  -n 220 --temp 0.85 --top-k 60 --seed 42 -ngl 0 --no-warmup --simple-io
