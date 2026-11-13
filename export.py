"""Step 6 - package the trained model for Ollama.

    python export.py --size tiny                 # HF folder + GGUF + Modelfile
    python export.py --size tiny --ollama        # ...and `ollama create bangla-mini-gpt-tiny`

  ckpt.pt ──► Hugging Face LlamaForCausalLM folder ──► llama.cpp converter ──► .gguf ──► Ollama

Our module names already match Llama's, so step 1 is a straight weight copy.
A parity check proves both implementations give the same outputs before we
go further.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

from minigpt import GPTConfig, MiniGPT

ROOT = Path(__file__).parent
SP_MODEL = ROOT / "tokenizer" / "bn_sp.model"
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"


def to_llama_config(cfg: GPTConfig) -> LlamaConfig:
    # transformers v5 moved rope_theta into rope_parameters; Colab may still have v4
    if int(transformers.__version__.split(".")[0]) >= 5:
        rope = {"rope_parameters": {"rope_type": "default", "rope_theta": cfg.rope_theta}}
    else:
        rope = {"rope_theta": cfg.rope_theta}
    return LlamaConfig(
        vocab_size=cfg.vocab_size, hidden_size=cfg.n_embd, intermediate_size=cfg.ffn_hidden,
        num_hidden_layers=cfg.n_layer, num_attention_heads=cfg.n_head, num_key_value_heads=cfg.n_head,
        head_dim=cfg.head_dim, max_position_embeddings=cfg.block_size, rms_norm_eps=cfg.norm_eps,
        hidden_act="silu", tie_word_embeddings=True, bos_token_id=1, eos_token_id=2,
        architectures=["LlamaForCausalLM"], **rope,
    )


def load_ours(ckpt_path: Path) -> MiniGPT:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MiniGPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def parity_check(ours: MiniGPT, hf: LlamaForCausalLM) -> float:
    ids = torch.randint(0, ours.cfg.vocab_size, (2, min(64, ours.cfg.block_size)))
    a = ours(ids)[0]
    b = hf(input_ids=ids).logits
    return (a - b).abs().max().item()


def write_hf(model: MiniGPT, hf_dir: Path) -> LlamaForCausalLM:
    hf = LlamaForCausalLM(to_llama_config(model.cfg)).eval()
    missing, unexpected = hf.load_state_dict(model.state_dict(), strict=False)
    missing = [k for k in missing if "rotary_emb" not in k]
    if missing or unexpected:
        raise RuntimeError(f"weight names differ: missing={missing} unexpected={unexpected}")
    diff = parity_check(model, hf)
    print(f"parity check: max |ours - Llama| = {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError("outputs differ - export would be wrong")

    hf_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(hf_dir, safe_serialization=True)
    shutil.copy(SP_MODEL, hf_dir / "tokenizer.model")
    (hf_dir / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "LlamaTokenizer", "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>",
        "add_bos_token": True, "add_eos_token": False, "model_max_length": model.cfg.block_size,
    }, indent=2))
    (hf_dir / "special_tokens_map.json").write_text(json.dumps(
        {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}, indent=2))
    return hf


def ensure_llama_cpp(path: Path) -> Path:
    script = path / "convert_hf_to_gguf.py"
    if not script.exists():
        print(f"cloning llama.cpp into {path} (converter scripts only, no build needed)")
        subprocess.run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, str(path)], check=True)
    return script


def write_modelfile(gguf: Path, block_size: int) -> Path:
    mf = gguf.parent / "Modelfile"
    mf.write_text(f'''FROM ./{gguf.name}

# A base model: it continues text, it does not chat. No chat template.
TEMPLATE """{{{{ .Prompt }}}}"""
PARAMETER temperature 0.8
PARAMETER top_k 50
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx {block_size}
PARAMETER num_predict 200
PARAMETER stop "</s>"
''', encoding="utf-8")
    return mf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--ckpt", type=Path, help="default out/<size>/ckpt.pt")
    ap.add_argument("--dest", type=Path, help="default export/<size>")
    ap.add_argument("--llama-cpp", type=Path, default=ROOT / "llama.cpp")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "q8_0"])
    ap.add_argument("--ollama", action="store_true", help="also run `ollama create`")
    ap.add_argument("--skip-gguf", action="store_true", help="stop after the Hugging Face folder")
    a = ap.parse_args()

    ckpt = a.ckpt or ROOT / "out" / a.size / "ckpt.pt"
    dest = a.dest or ROOT / "export" / a.size
    name = f"bangla-mini-gpt-{a.size}"

    model = load_ours(ckpt)
    print(f"loaded {ckpt} ({model.num_params()/1e6:.1f}M params)")
    write_hf(model, dest / "hf")
    print(f"Hugging Face folder -> {dest / 'hf'}")
    if a.skip_gguf:
        return

    script = ensure_llama_cpp(a.llama_cpp)
    gguf = dest / f"{name}.gguf"
    subprocess.run([sys.executable, str(script), str(dest / "hf"),
                    "--outfile", str(gguf), "--outtype", a.outtype], check=True)
    mf = write_modelfile(gguf, model.cfg.block_size)
    print(f"GGUF -> {gguf} ({gguf.stat().st_size/1e6:.1f} MB)\nModelfile -> {mf}")

    if a.ollama:
        subprocess.run(["ollama", "create", name, "-f", str(mf)], check=True, cwd=dest)
        print(f'\nready:  ollama run {name} "বাংলাদেশ"')
    else:
        print(f"\nnext:   cd {dest}; ollama create {name} -f Modelfile")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
