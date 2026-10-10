"""
convert_hf_to_gguf_custom_bpe.py — executa o conversor OFICIAL do llama.cpp
(convert_hf_to_gguf.py) com um ajuste necessário para tokenizers BPE
treinados do zero: o conversor identifica o pré-tokenizador comparando o
hash dos tokens de teste com uma lista de modelos conhecidos. Como nosso
tokenizer é inédito, registramos manualmente o mapeamento correto:

  ByteLevel (estilo GPT-2)  ->  tokenizer.ggml.pre = "gpt2"

que é exatamente o pré-tokenizador usado pelos modelos GPT-2 reais.

Requer a variável LLAMA_CPP_DIR apontando para um clone do llama.cpp
(conversor puro-Python; não precisa compilar nada para a conversão).

Uso:
  python3 scripts/convert_hf_to_gguf_custom_bpe.py out/hf --outfile modelo.gguf --outtype f16
"""

import os
import sys
from pathlib import Path

LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", "/tmp/llama.cpp")
if not Path(LLAMA_CPP_DIR, "convert_hf_to_gguf.py").exists():
    sys.exit(
        f"ERRO: conversor não encontrado em {LLAMA_CPP_DIR}.\n"
        "Clone o llama.cpp e/ou defina LLAMA_CPP_DIR:\n"
        "  git clone https://github.com/ggml-org/llama.cpp /tmp/llama.cpp"
    )

sys.path.insert(0, LLAMA_CPP_DIR)

from conversion.base import TextModel  # noqa: E402


def _pre_custom(self, tokenizer) -> str:
    # Nosso BPE usa ByteLevel (mesmo esquema do GPT-2) -> pre "gpt2"
    return "gpt2"


TextModel.get_vocab_base_pre = _pre_custom
print("[wrapper] pré-tokenizador registrado: gpt2 (ByteLevel, estilo GPT-2)")

if __name__ == "__main__":
    import runpy
    script = str(Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py")
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
