"""
train.py — treina o GPT do zero em CPU (funciona igual em GPU).

Uso:
  python3 scripts/train.py --max-iters 3000
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CKPT_DIR = REPO_ROOT / "checkpoints"


def get_batch(split: str, batch_size: int, block_size: int, data: dict, rng: np.random.Generator):
    arr = data[split]
    ix = rng.integers(0, len(arr) - block_size - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(arr[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(arr[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss(model, data, cfg_params, eval_iters: int, rng):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, cfg_params["bs"], cfg_params["ctx"], data, rng)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(it: int, base_lr: float, warmup: int, max_iters: int) -> float:
    if it < warmup:
        return base_lr * (it + 1) / warmup
    if it > max_iters:
        return base_lr * 0.1
    decay_ratio = (it - warmup) / max(1, max_iters - warmup)
    return base_lr * 0.1 + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * decay_ratio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--timeout-min", type=float, default=0.0, help="0 = sem limite")
    ap.add_argument("--gen-tokens", type=int, default=400)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    data = {
        "train": np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r"),
        "val": np.memmap(DATA_DIR / "val.bin", dtype=np.uint16, mode="r"),
    }
    vocab_size = int(data["train"].max()) + 1
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenizer_machado.json"))
    vocab_size = tok.get_vocab_size()

    cfg = GPTConfig(
        vocab_size=vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
    )
    model = GPT(cfg)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[modelo] {nparams:,} parâmetros | {args.n_layer} camadas | d={args.n_embd} | ctx={args.block_size} | vocab={vocab_size}")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    cfg_params = {"bs": args.batch_size, "ctx": args.block_size}
    CKPT_DIR.mkdir(exist_ok=True)
    best_val = float("inf")
    t0 = time.time()
    log_path = CKPT_DIR / "train_log.jsonl"
    log_f = open(log_path, "w")

    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, data, cfg_params, args.eval_iters, rng)
            dt = time.time() - t0
            best_val = min(best_val, losses["val"])
            rec = {"iter": it, **losses, "lr": lr_at(it, args.lr, args.warmup, args.max_iters),
                   "s": round(dt, 1), "ms_per_step": round(dt * 1000 / max(it, 1), 1)}
            print(f"it {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"lr {rec['lr']:.2e} | {dt:7.1f}s decorridos", flush=True)
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "iter": it,
                        "val_loss": losses["val"]}, CKPT_DIR / "last.pt")
            if losses["val"] <= best_val:
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "iter": it,
                            "val_loss": losses["val"]}, CKPT_DIR / "best.pt")
        if args.timeout_min and (time.time() - t0) > args.timeout_min * 60:
            print(f"[treino] timeout de {args.timeout_min} min atingido no iter {it}; encerrando.")
            break
        if it == args.max_iters:
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(it, args.lr, args.warmup, args.max_iters)
        x, y = get_batch("train", args.batch_size, args.block_size, data, rng)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    log_f.close()

    # amostra de geração direto do PyTorch (sanidade)
    model.eval()
    ctx_ids = torch.zeros((1, 1), dtype=torch.long)
    y = model.generate(ctx_ids, args.gen_tokens, temperature=0.85, top_k=64, eos_id=0)
    texto = tok.decode(y[0].tolist())
    print("\n===== AMOSTRA GERADA (PyTorch, temp 0.85, top_k 64) =====")
    print(texto)
    print("==========================================================")
    (CKPT_DIR / "sample_torch.txt").write_text(texto, encoding="utf-8")
    print(f"[OK] treino concluído em {(time.time()-t0)/60:.1f} min | melhor val: {best_val:.4f}")


from dataclasses import asdict  # noqa: E402  (usado acima)

if __name__ == "__main__":
    main()
