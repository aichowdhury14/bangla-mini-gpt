"""A Llama-style GPT written from scratch.

    token ids ──► embed ──► [ block × n_layer ] ──► RMSNorm ──► lm_head ──► next-token scores

    block:  x = x + Attention(RMSNorm(x))      # tokens look at earlier tokens
            x = x + SwiGLU-MLP(RMSNorm(x))     # each token "thinks" on its own

Module names match Hugging Face's LlamaForCausalLM on purpose
(model.layers.0.self_attn.q_proj ...). That makes export a plain copy
of the weights, and lets tests prove our maths equals Llama's.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class RMSNorm(nn.Module):
    """Scale each vector to unit root-mean-square, then by a learned weight."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def rope_tables(head_dim: int, length: int, theta: float, device=None):
    """cos/sin tables for Rotary Position Embedding.

    Each pair of channels in q and k is rotated by an angle that grows with
    the token's position, so the dot product q·k depends on *distance*.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(length, device=device).float()
    freqs = torch.outer(pos, inv_freq)            # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)        # (T, head_dim)  - HF "rotate_half" layout
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        d = cfg.n_embd
        self.q_proj = nn.Linear(d, d, bias=False)   # "what am I looking for?"
        self.k_proj = nn.Linear(d, d, bias=False)   # "what do I contain?"
        self.v_proj = nn.Linear(d, d, bias=False)   # "what do I pass on?"
        self.o_proj = nn.Linear(d, d, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        # (B, T, C) -> (B, heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # softmax(q·kᵀ / √d) · v, with a causal mask: token t only sees tokens ≤ t
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class MLP(nn.Module):
    """SwiGLU: down( silu(gate(x)) * up(x) )."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Decoder(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.model = Decoder(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight   # tie: same matrix reads & writes tokens

        cos, sin = rope_tables(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 trick: shrink the layers that write into the residual stream,
        # so the sum over many blocks starts at a sensible size.
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        cos, sin = self.rope_cos[:T].to(idx.device), self.rope_sin[:T].to(idx.device)

        x = self.model.embed_tokens(idx)
        for block in self.model.layers:
            x = block(x, cos, sin)
        logits = self.lm_head(self.model.norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int | None = 50, eos_id: int | None = None):
        """Sample one token at a time and feed it back in."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits[logits < kth] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and nxt.item() == eos_id:
                break
        return idx
