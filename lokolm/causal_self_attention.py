import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-Head Causal Attention

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Key, Query, Value projections combined into one linear layer
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dim (d_model)

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        
        # Reshape to (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal self-attention via PyTorch's fused kernel (FlashAttention on GPU).
        # is_causal=True applies the lower-triangular mask internally; scaling is automatic.
        # (B, n_heads, T, head_dim)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)