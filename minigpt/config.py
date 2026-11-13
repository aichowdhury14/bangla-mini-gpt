"""Model sizes. Pick one with --size when training.

The shape is Llama-style (RMSNorm, RoPE, SwiGLU, no biases, tied embeddings)
so the trained model converts to GGUF and runs in Ollama.
"""
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int = 16000
    block_size: int = 512        # context length (tokens the model can see)
    n_layer: int = 8             # transformer blocks stacked
    n_head: int = 8              # attention heads per block
    n_embd: int = 512            # width of every token vector
    ffn_hidden: int = 1408       # SwiGLU hidden width (~2.75 x n_embd)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)


SIZES = {
    # ~2.9M params (2M of it is the 16k-token embedding) - laptop CPU, ~15 min. Pipeline check only.
    "tiny": dict(block_size=256, n_layer=4, n_head=4, n_embd=128, ffn_hidden=352),
    # ~17M params - ~30-40 min on a free Colab T4.
    "small": dict(block_size=512, n_layer=6, n_head=6, n_embd=384, ffn_hidden=1024),
    # ~49M params - the real target. ~2-3 h on a T4, spread over sessions with --resume.
    "base": dict(block_size=512, n_layer=10, n_head=9, n_embd=576, ffn_hidden=1536),
}


def make_config(size: str, vocab_size: int) -> GPTConfig:
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; choose from {list(SIZES)}")
    return GPTConfig(vocab_size=vocab_size, **SIZES[size])
