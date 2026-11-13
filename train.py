"""Step 4 - train the model.

    python train.py --size tiny                       # CPU check, a few minutes
    python train.py --size base --out /content/drive/MyDrive/bmg/base   # Colab
    python train.py --size base --out ... --resume    # continue after Colab disconnects

The loop, in one line: read a random window of text, ask the model to guess
each next token, measure how wrong it was (cross-entropy loss), nudge every
weight a little to be less wrong (AdamW), repeat.
"""
import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from minigpt import MiniGPT, make_config

ROOT = Path(__file__).parent
TOK = ROOT / "data" / "tokens"
SP_MODEL = ROOT / "tokenizer" / "bn_sp.model"

# batch = sequences per micro-step, accum = micro-steps per weight update
DEFAULTS = {
    "tiny":  dict(batch=16, accum=1, lr=2e-3, warmup=50,  max_steps=600),
    "small": dict(batch=32, accum=2, lr=1e-3, warmup=200, max_steps=0),
    "base":  dict(batch=32, accum=4, lr=6e-4, warmup=300, max_steps=0),
}
PROMPTS = ["বাংলাদেশ", "ঢাকা শহরের", "রবীন্দ্রনাথ ঠাকুর"]


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny", choices=list(DEFAULTS))
    ap.add_argument("--out", default=None, help="checkpoint folder (default out/<size>)")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--accum", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--max-steps", type=int, help="0 = derive from --epochs")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--max-minutes", type=float, default=0, help="save and stop after N minutes")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile (faster on GPU)")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    for k, v in DEFAULTS[a.size].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    a.out = Path(a.out or ROOT / "out" / a.size)
    return a


class Batches:
    """Random windows of block_size+1 tokens: x = window[:-1], y = window[1:]."""

    def __init__(self, path: Path, block: int, batch: int, device: str):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block, self.batch, self.device = block, batch, device

    def get(self):
        ix = np.random.randint(0, len(self.data) - self.block - 1, self.batch)
        x = torch.from_numpy(np.stack([self.data[i:i + self.block] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([self.data[i + 1:i + 1 + self.block] for i in ix]).astype(np.int64))
        if self.device == "cuda":
            return x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y


def lr_at(step, a, total):
    """Linear warm-up, then cosine decay to 10% of the peak."""
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    progress = min(1.0, (step - a.warmup) / max(1, total - a.warmup))
    return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast = torch.autocast("cuda", dtype=amp_dtype)
    else:
        amp_dtype, autocast = torch.float32, nullcontext()
        torch.set_num_threads(max(1, torch.get_num_threads()))

    meta = json.loads((TOK / "meta.json").read_text())
    cfg = make_config(a.size, meta["vocab_size"])
    sp = spm.SentencePieceProcessor(model_file=str(SP_MODEL))
    train = Batches(TOK / "train.bin", cfg.block_size, a.batch, device)
    val = Batches(TOK / "val.bin", cfg.block_size, a.batch, device)

    tokens_per_step = a.batch * a.accum * cfg.block_size
    total = a.max_steps or int(a.epochs * meta["train_tokens"] / tokens_per_step)

    model = MiniGPT(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95), fused=(device == "cuda"),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    a.out.mkdir(parents=True, exist_ok=True)
    ckpt_path, log_path = a.out / "ckpt.pt", a.out / "log.jsonl"
    step, best_val = 0, float("inf")
    if a.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"resumed at step {step}")

    fwd = torch.compile(model) if a.compile else model
    print(f"{a.size}: {model.num_params()/1e6:.1f}M params | {device} {str(amp_dtype).split('.')[-1]} | "
          f"{tokens_per_step:,} tokens/step | {total:,} steps "
          f"(~{total*tokens_per_step/meta['train_tokens']:.1f} epochs of {meta['train_tokens']/1e6:.0f}M tokens)")

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for name, src in (("train", train), ("val", val)):
            losses = []
            for _ in range(a.eval_iters):
                x, y = src.get()
                with autocast:
                    losses.append(fwd(x, y)[1].item())
            out[name] = sum(losses) / len(losses)
        model.train()
        return out

    def sample(prompt):
        model.eval()
        idx = torch.tensor([[sp.bos_id(), *sp.encode(prompt)]], device=device)
        with autocast:
            ids = model.generate(idx, 40, temperature=0.8, top_k=50, eos_id=sp.eos_id())
        model.train()
        return sp.decode(ids[0, 1:].tolist())

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "best_val": best_val, "config": cfg.to_dict(), "size": a.size},
                   ckpt_path)

    start, t0, tok_seen = time.time(), time.time(), 0
    model.train()
    while step < total:
        lr = lr_at(step, a, total)
        for g in opt.param_groups:
            g["lr"] = lr

        for _ in range(a.accum):
            x, y = train.get()
            with autocast:
                _, loss = fwd(x, y)
            scaler.scale(loss / a.accum).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_seen += tokens_per_step

        if step % 10 == 0 or step == 1:
            dt = time.time() - t0
            tps = tok_seen / dt
            eta = (total - step) * tokens_per_step / tps / 60
            print(f"step {step:>6}/{total} | loss {loss.item():.3f} | lr {lr:.2e} | "
                  f"{tps/1e3:.1f}k tok/s | ETA {eta:.0f} min", flush=True)
            t0, tok_seen = time.time(), 0

        last = step == total
        out_of_time = a.max_minutes and (time.time() - start) / 60 > a.max_minutes
        if step % a.eval_every == 0 or last or out_of_time:
            ev = evaluate()
            best_val = min(best_val, ev["val"])
            text = sample(PROMPTS[(step // a.eval_every) % len(PROMPTS)])
            print(f"-- eval step {step}: train {ev['train']:.3f}  val {ev['val']:.3f}  "
                  f"(perplexity {math.exp(ev['val']):.1f})\n   ✍ {text}", flush=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "lr": lr, **ev, "sample": text}, ensure_ascii=False) + "\n")
            save()
        if out_of_time:
            print(f"time limit reached - saved at step {step}. Re-run with --resume to continue.")
            break

    print(f"done. best val loss {best_val:.3f} -> {ckpt_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
