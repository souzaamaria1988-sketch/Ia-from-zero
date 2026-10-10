#!/usr/bin/env bash
# quantize.sh — cria versões quantizadas do GGUF para celular.
# Uso: LLAMA_CPP_DIR=/tmp/llama.cpp bash scripts/quantize.sh models/gpt-machado-f16.gguf
set -euo pipefail

SRC="${1:?uso: quantize.sh <modelo-f16.gguf>}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/tmp/llama.cpp}"
QBIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
BASE="$(dirname "$SRC")/$(basename "$SRC" .gguf)"

if [ ! -x "$QBIN" ]; then
  echo "ERRO: llama-quantize não encontrado em $QBIN"
  echo "Compile: cmake -B build -S $LLAMA_CPP_DIR && cmake --build build --target llama-quantize -j"
  exit 1
fi

echo "== Q8_0 (recomendado para celular: ~1 byte/param, perda mínima) =="
"$QBIN" "$SRC" "$BASE-q8_0.gguf" q8_0

echo "== Q4_K_M (~4,8 bits/param, melhor equilíbrio qualidade/tamanho) =="
"$QBIN" "$SRC" "$BASE-q4_k_m.gguf" q4_k_m || echo "(Q4_K_M indisponível — normal em modelos minúsculos)"

echo "== Q4_0 (o menor ainda utilizável) =="
"$QBIN" "$SRC" "$BASE-q4_0.gguf" q4_0

echo "== Resultado =="
ls -la "$BASE"*.gguf
