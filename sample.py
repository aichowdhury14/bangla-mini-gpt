"""Step 5 - let the model write.

    python sample.py --size tiny --prompt "বাংলাদেশ"
    python sample.py --size base --prompt "ঢাকা" --n 3 --temperature 0.7

It is a *base* model: give it the start of a sentence and it continues,
Wikipedia-style. It has not been taught to follow instructions or chat.
"""
import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch

from minigpt import GPTConfig, MiniGPT

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--prompt", default="বাংলাদেশ")
    ap.add_argument("--n", type=int, default=1, help="how many samples")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8, help="lower = safer, higher = wilder")
    ap.add_argument("--top-k", type=int, default=50)
    a = ap.parse_args()

    ck = torch.load(a.ckpt or ROOT / "out" / a.size / "ckpt.pt", map_location="cpu", weights_only=False)
    model = MiniGPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "bn_sp.model"))

    print(f"step {ck['step']} | best val loss {ck['best_val']:.3f}\n")
    for i in range(a.n):
        idx = torch.tensor([[sp.bos_id(), *sp.encode(a.prompt)]])
        out = model.generate(idx, a.tokens, a.temperature, a.top_k, eos_id=sp.eos_id())
        print(f"[{i + 1}] {sp.decode(out[0, 1:].tolist())}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
