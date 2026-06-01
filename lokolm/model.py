import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_self_attention import CausalSelfAttention
from .mlp import MLP


# A single Transformer decoder block: attention + MLP, each with a
# pre-norm residual connection (GPT-2 style).

class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# lokoLM — a decoder-only Transformer language model.

class LokoLM(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=768, n_heads=12, n_layers=12, mlp_ratio=4):
        super().__init__()
        self.block_size = block_size

        # Token and positional embeddings
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mlp_ratio) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # Language modeling head; weight-tied to the token embedding
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"sequence length {T} exceeds block size {self.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)  # (B, T, d_model)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # last step

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    # Tiny smoke test
    model = LokoLM(vocab_size=100, block_size=32, d_model=64, n_heads=4, n_layers=2)
    x = torch.randint(0, 100, (2, 16))
    logits, loss = model(x, targets=x)
    print("logits:", logits.shape, "loss:", loss.item())

    out = model.generate(x[:, :1], max_new_tokens=10)
    print("generated:", out.shape)
