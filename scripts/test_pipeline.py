"""
test_pipeline.py — testes de sanidade do pipeline completo (sem dependências
além das já usadas pelo projeto). Rode: python3 scripts/test_pipeline.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig, hf_state_dict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
MODELS = REPO_ROOT / "models"

OK = []


def check(nome, cond, extra=""):
    status = "✅" if cond else "❌"
    OK.append(bool(cond))
    print(f"{status} {nome}{(' — ' + extra) if extra else ''}")


def test_tokenizer_roundtrip():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(DATA / "tokenizer_machado.json"))
    frase = "— Capitu, você me olhava com olhos de ressaca; o dela eram os meus."
    ids = tok.encode(frase).ids
    check("tokenizer: round-trip ida-e-volta", tok.decode(ids) == frase,
          f"{len(ids)} tokens")
    check("tokenizer: vocabulário 512", tok.get_vocab_size() == 512)


def test_tokenizer_v3():
    tok3 = DATA / "tokenizer_machado-v3.json"
    if not tok3.exists():
        print("⏭️  tokenizer v3 ainda não gerado (rode prepare_data --tag v3)")
        return
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok3))
    check("tokenizer v3: vocabulário 1024", tok.get_vocab_size() == 1024)
    base = Tokenizer.from_file(str(DATA / "tokenizer_machado.json"))
    txt = "Numa rua do Cateti há uma casquinha de mós azul."
    check("tokenizer v3 comprime melhor que o de 512",
          len(tok.encode(txt).ids) <= len(base.encode(txt).ids))


def test_model_forward():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=512, block_size=64, n_layer=2, n_head=2, n_embd=64)
    m = GPT(cfg)
    x = torch.randint(0, 512, (4, 64))
    logits, loss = m(x, x)
    check("modelo: shape dos logits", tuple(logits.shape) == (4, 64, 512))
    check("modelo: loss inicial ≈ ln(512)",
          abs(loss.item() - math.log(512)) < 0.35, f"loss={loss.item():.3f}")


def test_treino_aprende():
    """30 passos num lote FIXO têm que memorizar (loss cai bastante)."""
    torch.manual_seed(1)
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    m = GPT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 256, (8, 64))
    _, loss0 = m(x, x)
    for _ in range(30):
        _, loss = m(x, x)
        opt.zero_grad(); loss.backward(); opt.step()
    check("treino: loss cai ao aprender (30 passos, lote fixo)",
          loss.item() < loss0.item() - 1.0,
          f"{loss0.item():.2f} -> {loss.item():.2f}")


def test_hf_layout():
    cfg = GPTConfig(vocab_size=256, block_size=64, n_layer=2, n_head=2, n_embd=64)
    m = GPT(cfg)
    sd = hf_state_dict(m)
    w = sd["transformer.h.0.attn.c_attn.weight"]
    check("export HF: Conv1D transposto [in,out]",
          w.shape[0] == 64 and w.shape[1] == 3 * 64, str(tuple(w.shape)))
    check("export HF: lm_head presente (não amarrado)",
          "lm_head.weight" in sd and sd["lm_head.weight"].shape == (256, 64))


def test_checkpoints():
    ck = REPO_ROOT / "checkpoints" / "best.pt"
    if not ck.exists():
        print("⏭️  checkpoints ainda não treinados")
        return
    obj = torch.load(ck, map_location="cpu", weights_only=False)
    m = GPT(GPTConfig(**obj["cfg"]))
    m.load_state_dict(obj["model"])
    if obj["iter"] < 1000:
        print(f"⏭️  checkpoint em início de treino (iter {obj['iter']}) — rode de novo no fim")
        return
    check("checkpoint: carrega e é mais esperto que aleatório",
          obj["val_loss"] < math.log(m.cfg.vocab_size) - 1.0,
          f"val {obj['val_loss']:.3f} (iter {obj['iter']})")


def test_gguf():
    f16 = MODELS / "gpt-machado-f16.gguf"
    q8 = MODELS / "gpt-machado-q8_0.gguf"
    if not f16.exists() or not q8.exists():
        print("⏭️  GGUFs ainda não exportados")
        return
    import gguf
    r = gguf.GGUFReader(str(q8))
    magic = bytes(r.data[0:4]) if hasattr(r, "data") else b"GGUF"
    nomes = {t.name for t in r.tensors}
    check("GGUF: tensores essenciais presentes",
          {"token_embd.weight", "output.weight", "blk.0.attn_qkv.weight"} <= nomes)
    check("GGUF: quantizado menor que F16", q8.stat().st_size < f16.stat().st_size,
          f"{q8.stat().st_size/1e6:.2f} MB < {f16.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    test_tokenizer_v3()
    test_model_forward()
    test_treino_aprende()
    test_hf_layout()
    test_checkpoints()
    test_gguf()
    n_ok = sum(OK)
    print(f"\n{n_ok}/{len(OK)} testes passaram")
    sys.exit(0 if n_ok == len(OK) else 1)
