"""Step 3 - turn all text into one long array of token ids.

    python tokenize_data.py

Each article becomes  <s> tokens... </s>  and they are joined end to end into
data/tokens/train.bin and val.bin (uint16, so vocab must be < 65,536).
Training then just reads random windows out of these files.
"""
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).parent
CLEAN, TOK = ROOT / "data" / "clean", ROOT / "data" / "tokens"
SP_MODEL = ROOT / "tokenizer" / "bn_sp.model"


def encode_file(sp, src: Path, dst: Path, batch: int = 1000) -> int:
    total = 0
    with open(src, encoding="utf-8") as f, open(dst, "wb") as out:
        texts = []

        def flush():
            nonlocal total
            for ids in sp.encode(texts, num_threads=8):
                arr = np.array([sp.bos_id(), *ids, sp.eos_id()], dtype=np.uint16)
                out.write(arr.tobytes())
                total += len(arr)
            texts.clear()

        for row in f:
            texts.append(json.loads(row)["text"])
            if len(texts) >= batch:
                flush()
                print(f"  {dst.name}: {total/1e6:.1f}M tokens", end="\r")
        if texts:
            flush()
    print(f"  {dst.name}: {total:,} tokens        ")
    return total


def main():
    TOK.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    assert sp.vocab_size() < 65536
    n_train = encode_file(sp, CLEAN / "train.jsonl", TOK / "train.bin")
    n_val = encode_file(sp, CLEAN / "val.jsonl", TOK / "val.bin")
    chars = sum(len(json.loads(r)["text"]) for r in open(CLEAN / "val.jsonl", encoding="utf-8"))
    meta = {"vocab_size": sp.vocab_size(), "train_tokens": n_train, "val_tokens": n_val,
            "chars_per_token": round(chars / max(n_val, 1), 2)}
    (TOK / "meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
