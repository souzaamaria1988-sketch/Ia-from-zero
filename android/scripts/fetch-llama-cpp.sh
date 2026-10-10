#!/usr/bin/env bash
# Clona o llama.cpp (versão fixa) na pasta que o CMake do app espera.
set -euo pipefail
cd "$(dirname "$0")/../app/src/main/cpp"

if [ -d llama.cpp ]; then
  echo "llama.cpp já presente em $PWD/llama.cpp"
  exit 0
fi

TAG="v0.6.0"
echo "Baixando llama.cpp $TAG ..."
git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp
echo "OK: $(pwd)/llama.cpp"
