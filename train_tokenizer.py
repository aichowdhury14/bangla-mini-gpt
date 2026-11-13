"""Step 2 - teach a tokenizer how to cut Bangla into pieces.

    python train_tokenizer.py                     # 16k vocab
    python train_tokenizer.py --vocab 4000        # small local test

SentencePiece BPE starts from single characters and keeps merging the most
frequent neighbouring pair ("ব"+"াং" -> "বাং" ...) until it has --vocab pieces.
byte_fallback means any character it never saw (emoji, rare symbols, "\\n")
still encodes as raw bytes - nothing becomes <unk>.

Writes tokenizer/bn_sp.model (+ .vocab to read by eye).
Llama-style ids: 0=<unk> 1=<s> (start) 2=</s> (end).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).parent
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "tokenizer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--max-lines", type=int, default=2_000_000, help="paragraphs sampled for training")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    # SentencePiece reads one sentence per line, so feed it paragraphs.
    lines = []
    with open(CLEAN / "train.jsonl", encoding="utf-8") as f:
        for row in f:
            lines.extend(p for p in json.loads(row)["text"].split("\n") if len(p) > 20)
    random.Random(0).shuffle(lines)
    lines = lines[: args.max_lines]
    sample = OUT / "_train_lines.txt"
    sample.write_text("\n".join(lines), encoding="utf-8")
    print(f"training tokenizer on {len(lines):,} paragraphs, vocab {args.vocab}")

    spm.SentencePieceTrainer.train(
        input=str(sample),
        model_prefix=str(OUT / "bn_sp"),
        model_type="bpe",
        vocab_size=args.vocab,
        character_coverage=0.9995,     # keep virtually every Bangla character as its own piece
        byte_fallback=True,
        split_digits=True,              # 2024 -> 2 0 2 4, helps with numbers
        normalization_rule_name="identity",  # text is already NFC-normalised
        unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
        num_threads=8,
        max_sentence_length=8192,
        minloglevel=2,
    )
    sample.unlink()

    sp = spm.SentencePieceProcessor(model_file=str(OUT / "bn_sp.model"))
    demo = "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ।"
    pieces = sp.encode(demo, out_type=str)
    print(f"\n{demo}\n-> {len(pieces)} tokens: {pieces}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
