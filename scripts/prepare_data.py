"""
prepare_data.py — monta o corpus, treina o tokenizer BPE e tokeniza tudo.

1. Baixa as obras de Machado de Assis (domínio público, machado.mec.gov.br)
   do dataset `luxedo/dataset-machado-de-assis` no GitHub.
2. Filtra prosa (romances, contos, crônicas), limpa e concatena -> data/corpus.txt
3. Treina um tokenizer BPE byte-level (estilo GPT-2) do ZERO, vocab 512.
4. Codifica o corpus -> data/train.bin e data/val.bin (uint16).
"""

import csv
import io
import os
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.pre_tokenizers import ByteLevel

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

DATASET_URL = "https://codeload.github.com/luxedo/dataset-machado-de-assis/tar.gz/refs/heads/master"
CATEGORIAS_PROSA = {"romance", "conto", "crônica"}
VOCAB_SIZE = 512


def baixar_csv(dest: Path) -> Path:
    if dest.exists():
        print(f"[dados] CSV já existe: {dest}")
        return dest
    print(f"[dados] baixando dataset de {DATASET_URL} ...")
    raw = urllib.request.urlopen(DATASET_URL, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("obras_machado_de_assis.csv"))
        f = tar.extractfile(member)
        dest.write_bytes(f.read())
    print(f"[dados] CSV extraído: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def limpar(texto: str) -> str:
    """Limpeza leve: normaliza espaços/linhas, remove rodapés com só números."""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    linhas = []
    for ln in texto.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"\d{1,4}", s):          # número de página órfão
            continue
        if re.fullmatch(r"[-_=*~.]{3,}", s):     # separadores decorativos
            continue
        linhas.append(s)
    t = "\n".join(linhas)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def montar_corpus(csv_path: Path) -> Path:
    out = DATA_DIR / "corpus.txt"
    if out.exists():
        print(f"[dados] corpus já existe: {out}")
        return out
    csv.field_size_limit(sys.maxsize)
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    prosa = [r for r in rows if r["categoria"] in CATEGORIAS_PROSA and len(r.get("texto") or "") > 5000]
    prosa.sort(key=lambda r: r["titulo"])
    print(f"[dados] {len(prosa)} obras de prosa de {len(rows)} registros")
    textos = [limpar(r["texto"]) for r in prosa]
    corpus = "\n\n\n".join(textos)
    out.write_text(corpus, encoding="utf-8")
    print(f"[dados] corpus: {out} ({len(corpus)/1e6:.2f} M caracteres)")
    return out


def treinar_tokenizer(corpus: Path) -> Path:
    tok_path = DATA_DIR / "tokenizer_machado.json"
    if tok_path.exists():
        print(f"[tokenizer] já existe: {tok_path}")
        return tok_path
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # estilo GPT-2
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=False,
    )
    tok.train([str(corpus)], trainer)
    tok.save(str(tok_path))
    print(f"[tokenizer] BPE treinado do zero: vocab={tok.get_vocab_size()} -> {tok_path}")
    return tok_path


def tokenizar(tok_path: Path, val_frac: float = 0.01) -> None:
    tok = Tokenizer.from_file(str(tok_path))
    corpus = (DATA_DIR / "corpus.txt").read_text(encoding="utf-8")
    ids = np.array(tok.encode(corpus).ids, dtype=np.uint16)
    n = len(ids)
    n_val = max(2000, int(n * val_frac))
    ids[:-n_val].tofile(DATA_DIR / "train.bin")
    ids[-n_val:].tofile(DATA_DIR / "val.bin")
    print(f"[tokenizar] {n:,} tokens | treino {n-n_val:,} | val {n_val:,}")
    print(f"[tokenizar] razão texto/token = {len(corpus)/n:.2f} chars/token")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = baixar_csv(Path("/tmp/obras_machado_de_assis.csv"))
    corpus = montar_corpus(csv_path)
    tok_path = treinar_tokenizer(corpus)
    tokenizar(tok_path)
    print("[OK] dados prontos em data/")
