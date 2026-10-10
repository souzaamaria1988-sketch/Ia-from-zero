"""
demo_interativo.py — converse/continue texto com o GPT-Machado usando só PyTorch
(sem llama.cpp). Carrega o checkpoint treinado e completa o que você digitar.

Uso:
  python3 scripts/demo_interativo.py                 # checkpoint best.pt
  python3 scripts/demo_interativo.py --temp 1.0      # mais criativo
  echo "— Capitu," | python3 scripts/demo_interativo.py --max 200
"""

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO_ROOT / "checkpoints" / "best.pt")
    ap.add_argument("--tok", type=Path, default=REPO_ROOT / "data" / "tokenizer_machado.json")
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--max", type=int, default=250, help="novos tokens por geração")
    ap.add_argument("--prompt", type=str, default="", help="gera uma vez e sai (lê stdin se vazio)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ck["cfg"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = Tokenizer.from_file(str(args.tok))
    eos = tok.token_to_id("<|endoftext|>")
    print(f"[GPT-Machado] {sum(p.numel() for p in model.parameters()):,} parâmetros | "
          f"val {ck['val_loss']:.3f} | temp {args.temp} | top_k {args.top_k}")
    print("Digite o início de uma frase em português. Ctrl-D para sair.\n")

    def gerar(prompt: str):
        ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long)
        y = model.generate(ids, args.max, temperature=args.temp, top_k=args.top_k, eos_id=eos)
        print("machado>", tok.decode(y[0].tolist()))

    if args.prompt:
        gerar(args.prompt)
        return
    if sys.stdin.isatty():
        while True:
            try:
                p = input("você> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[tchau!]")
                break
            if p:
                gerar(p)
    else:
        p = sys.stdin.read().strip()
        if p:
            gerar(p)


if __name__ == "__main__":
    main()
