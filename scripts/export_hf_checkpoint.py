"""
export_hf_checkpoint.py — exporta o checkpoint do treino para o formato
HuggingFace GPT-2 (config.json + model.safetensors + tokenizer), que é o que
o conversor oficial do llama.cpp (convert_hf_to_gguf.py) consome.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig, hf_state_dict

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=CKPT_DIR / "last.pt")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "out" / "hf")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[export] checkpoint iter {ck['iter']} | val {ck['val_loss']:.4f} | "
          f"{sum(p.numel() for p in model.parameters()):,} parâmetros")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"],
        "vocab_size": cfg.vocab_size,
        "n_positions": cfg.block_size,
        "n_ctx": cfg.block_size,
        "n_embd": cfg.n_embd,
        "n_layer": cfg.n_layer,
        "n_head": cfg.n_head,
        "layer_norm_epsilon": 1e-5,
        "activation_function": "gelu",
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 0,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    save_file(hf_state_dict(model), str(out / "model.safetensors"))

    # tokenizer treinado do zero -> arquivos clássicos do GPT-2
    tok = Tokenizer.from_file(str(REPO_ROOT / "data" / "tokenizer_machado.json"))
    tok.save(str(out / "tokenizer.json"))
    bpe = tok.model
    (out / "vocab.json").write_text(
        json.dumps(bpe.get_vocab(), ensure_ascii=False), encoding="utf-8"
    )
    merges = "\n".join(" ".join(m) for m in bpe.get_merges())
    (out / "merges.txt").write_text("#version: 0.2\n" + merges + "\n", encoding="utf-8")
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "GPT2TokenizerFast",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "add_prefix_space": False,
    }, indent=2), encoding="utf-8")
    (out / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
    }, indent=2), encoding="utf-8")

    print(f"[export] checkpoint HF salvo em {out}/")
    print("  arquivos:", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
