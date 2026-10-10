#!/usr/bin/env bash
# run_all.sh — pipeline completo: dados -> treino -> HF -> GGUF -> quantização.
# Pré-requisitos: pip install -r requirements.txt; clone do llama.cpp em LLAMA_CPP_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/tmp/llama.cpp}"
ITERS="${ITERS:-2400}"

python3 scripts/prepare_data.py
python3 scripts/train.py --max-iters "$ITERS"
python3 scripts/export_hf_checkpoint.py

mkdir -p models
python3 scripts/convert_hf_to_gguf_custom_bpe.py out/hf \
  --outfile models/gpt-machado-f16.gguf --outtype f16
LLAMA_CPP_DIR="$LLAMA_CPP_DIR" bash scripts/quantize.sh models/gpt-machado-f16.gguf
echo "[OK] pipeline completo. Modelos em models/"
