"""Playground website for checking and reviewing the model.

    python app.py      ->  open http://127.0.0.1:8000

Finds every out/<size>/ckpt.pt automatically and reloads a checkpoint when the
file changes, so you can keep this open while training runs.
"""
import json
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from minigpt import GPTConfig, MiniGPT

ROOT = Path(__file__).parent
OUT = ROOT / "out"
sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer" / "bn_sp.model"))
app = FastAPI(title="Bangla Mini-GPT")
_cache: dict[str, tuple[float, MiniGPT, dict]] = {}


def checkpoints() -> dict[str, Path]:
    return {p.parent.name: p for p in sorted(OUT.glob("*/ckpt.pt"))}


def load(size: str) -> tuple[MiniGPT, dict]:
    path = checkpoints().get(size)
    if path is None:
        raise HTTPException(404, f"no checkpoint for {size!r}")
    mtime = path.stat().st_mtime
    if size in _cache and _cache[size][0] == mtime:
        return _cache[size][1], _cache[size][2]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MiniGPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()
    info = {"size": size, "params": model.num_params(), "step": ck["step"],
            "best_val": ck["best_val"], "config": ck["config"],
            "updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))}
    _cache[size] = (mtime, model, info)
    return model, info


# ---------- API ----------

@app.get("/api/models")
def models():
    out = []
    for size in checkpoints():
        _, info = load(size)
        out.append(info)
    return sorted(out, key=lambda m: m["params"])


@app.get("/api/log/{size}")
def log(size: str):
    path = OUT / size / "log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TokenizeIn(BaseModel):
    text: str = Field(max_length=5000)


@app.post("/api/tokenize")
def tokenize(body: TokenizeIn):
    ids = sp.encode(body.text)
    return {"pieces": [sp.id_to_piece(i) for i in ids], "ids": ids,
            "chars": len(body.text), "vocab_size": sp.vocab_size()}


class GenerateIn(BaseModel):
    size: str
    prompt: str = Field(max_length=2000)
    max_tokens: int = Field(120, ge=1, le=500)
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    top_k: int = Field(50, ge=0, le=500)
    top_p: float = Field(0.95, gt=0, le=1.0)
    repeat_penalty: float = Field(1.15, ge=1.0, le=2.0)
    seed: int | None = None


@torch.no_grad()
def stream(model: MiniGPT, body: GenerateIn):
    """Same sampling as model.generate, plus top-p and a repetition penalty,
    yielding the new text after every token (as NDJSON lines)."""
    if body.seed is not None:
        torch.manual_seed(body.seed)
    prompt_ids = [sp.bos_id(), *sp.encode(body.prompt)]
    idx = torch.tensor([prompt_ids])
    shown, t0, n = sp.decode(prompt_ids[1:]), time.time(), 0
    for _ in range(body.max_tokens):
        logits = model(idx[:, -model.cfg.block_size:])[0][0, -1].float()
        if body.repeat_penalty > 1.0:                  # make recently used tokens less likely
            seen = idx[0, -64:].unique()
            vals = logits[seen]
            logits[seen] = torch.where(vals > 0, vals / body.repeat_penalty, vals * body.repeat_penalty)
        logits = logits / body.temperature
        if body.top_k:
            kth = torch.topk(logits, min(body.top_k, logits.numel())).values[-1]
            logits[logits < kth] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        if body.top_p < 1.0:                           # keep the smallest set covering top_p of the mass
            sp_probs, order = probs.sort(descending=True)
            drop = sp_probs.cumsum(0) - sp_probs > body.top_p
            sp_probs[drop] = 0
            probs = torch.zeros_like(probs).scatter(0, order, sp_probs)
            probs /= probs.sum()
        nxt = torch.multinomial(probs, 1)
        if nxt.item() == sp.eos_id():
            break
        idx = torch.cat([idx, nxt.view(1, 1)], dim=1)
        n += 1
        text = sp.decode(idx[0, 1:].tolist())
        if len(text) > len(shown) and not text.endswith("�"):   # wait for whole characters
            yield json.dumps({"t": text[len(shown):]}, ensure_ascii=False) + "\n"
            shown = text
    secs = time.time() - t0
    yield json.dumps({"done": True, "tokens": n, "secs": round(secs, 2),
                      "tok_per_s": round(n / max(secs, 1e-6), 1)}) + "\n"


@app.post("/api/generate")
def generate(body: GenerateIn):
    model, _ = load(body.size)
    return StreamingResponse(stream(model, body), media_type="application/x-ndjson")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("Bangla Mini-GPT playground -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
