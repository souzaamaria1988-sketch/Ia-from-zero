#!/usr/bin/env bash
# Copia o GGUF quantizado para os assets do app (o APK embute o modelo).
set -euo pipefail
cd "$(dirname "$0")/../../.."   # raiz do repositório

SRC="models/gpt-machado-q8_0.gguf"
DST="android/app/src/main/assets/gpt-machado-q8_0.gguf"

if [ ! -f "$SRC" ]; then
  echo "ERRO: $SRC não existe. Rode o pipeline primeiro (scripts/run_all.sh)."
  exit 1
fi

mkdir -p "$(dirname "$DST")"
cp "$SRC" "$DST"
echo "OK: modelo embutido em $DST ($(du -h "$DST" | cut -f1))"
