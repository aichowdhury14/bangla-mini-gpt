import torch

from minigpt import GPTConfig, MiniGPT

CFG = GPTConfig(vocab_size=500, block_size=32, n_layer=2, n_head=4, n_embd=64, ffn_hidden=160)


def test_shapes_and_loss():
    m = MiniGPT(CFG)
    x = torch.randint(0, 500, (3, 32))
    y = torch.randint(0, 500, (3, 32))
    logits, loss = m(x, y)
    assert logits.shape == (3, 32, 500)
    # untrained model ~ uniform guess: loss close to ln(vocab)
    assert abs(loss.item() - torch.log(torch.tensor(500.0)).item()) < 0.5


def test_causal_future_tokens_do_not_leak():
    m = MiniGPT(CFG).eval()
    x = torch.randint(0, 500, (1, 20))
    y = x.clone()
    y[0, 15:] = (y[0, 15:] + 1) % 500        # change only the "future"
    a, b = m(x)[0], m(y)[0]
    assert torch.allclose(a[0, :15], b[0, :15], atol=1e-6)
    assert not torch.allclose(a[0, 15:], b[0, 15:])


def test_embeddings_are_tied():
    m = MiniGPT(CFG)
    assert m.lm_head.weight is m.model.embed_tokens.weight


def test_overfits_one_batch():
    """If it can't memorise 1 batch, something in the maths is broken."""
    torch.manual_seed(0)
    m = MiniGPT(CFG)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 500, (4, 33))
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


def test_matches_huggingface_llama():
    """Same weights in HF's LlamaForCausalLM must give the same logits -
    this is what makes GGUF/Ollama export correct."""
    from export import parity_check, to_llama_config
    from transformers import LlamaForCausalLM

    m = MiniGPT(CFG).eval()
    hf = LlamaForCausalLM(to_llama_config(CFG)).eval()
    hf.load_state_dict(m.state_dict(), strict=False)
    assert parity_check(m, hf) < 1e-4
