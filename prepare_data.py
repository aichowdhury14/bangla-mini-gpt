"""Step 1 - download Bangla Wikipedia and clean it.

    python prepare_data.py                 # full (~145k articles, both shards)
    python prepare_data.py --shards 1 --max-docs 3000   # small local test

Writes data/clean/train.jsonl and data/clean/val.jsonl  ({"text": ...} per line).
"""
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "wikimedia/wikipedia"
SHARDS = [
    "20231101.bn/train-00000-of-00002.parquet",
    "20231101.bn/train-00001-of-00002.parquet",
]
ROOT = Path(__file__).parent
RAW, CLEAN = ROOT / "data" / "raw", ROOT / "data" / "clean"

BANGLA = re.compile(r"[ঀ-৿]")
LETTER = re.compile(r"[^\W\d_]")
SPACES = re.compile(r"[ \t ​]+")
BLANK_LINES = re.compile(r"\n{3,}")
# Wikipedia footer headings: References, External links, See also, Notes, Bibliography, Further reading.
# Everything after them is link lists and categories - noise for a language model.
FOOTERS = {"তথ্যসূত্র", "বহিঃসংযোগ", "আরও দেখুন", "টীকা", "গ্রন্থপঞ্জি", "আরও পড়ুন", "উৎস", "পাদটীকা"}


def clean(text: str) -> str | None:
    """Normalise one article. Returns None to drop it."""
    text = unicodedata.normalize("NFC", text)          # one spelling per character
    text = SPACES.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    for i, line in enumerate(lines):
        if line in FOOTERS:
            lines = lines[:i]
            break
    text = "\n".join(lines)
    text = BLANK_LINES.sub("\n\n", text).strip()
    if len(text) < 200:                                 # stubs teach little
        return None
    letters = LETTER.findall(text)
    if not letters or len(BANGLA.findall(text)) / len(letters) < 0.6:
        return None                                     # mostly English / lists of names
    return text


def is_val(text: str, frac: float) -> bool:
    """Stable split: the same article always lands on the same side."""
    h = int(hashlib.md5(text[:200].encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=len(SHARDS), help="how many parquet shards (1-2)")
    ap.add_argument("--max-docs", type=int, default=0, help="stop after N kept articles (0 = all)")
    ap.add_argument("--val-frac", type=float, default=0.005)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    CLEAN.mkdir(parents=True, exist_ok=True)
    kept = dropped = chars = 0
    with open(CLEAN / "train.jsonl", "w", encoding="utf-8") as ftr, \
         open(CLEAN / "val.jsonl", "w", encoding="utf-8") as fva:
        for shard in SHARDS[: args.shards]:
            path = hf_hub_download(REPO, shard, repo_type="dataset", local_dir=RAW)
            print(f"reading {Path(path).name}")
            for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2000):
                for text in batch.column("text").to_pylist():
                    text = clean(text)
                    if text is None:
                        dropped += 1
                        continue
                    out = fva if is_val(text, args.val_frac) else ftr
                    out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    kept += 1
                    chars += len(text)
                    if args.max_docs and kept >= args.max_docs:
                        break
                if args.max_docs and kept >= args.max_docs:
                    break
                print(f"  kept {kept:,}  dropped {dropped:,}", end="\r")
            if args.max_docs and kept >= args.max_docs:
                break
    print(f"\nkept {kept:,} articles ({chars/1e6:.1f}M characters), dropped {dropped:,}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
