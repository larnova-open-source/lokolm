"""
Minimal training loop for lokoLM, the decoder-only Transformer, with CUDA support.

Run:
    python train.py

The script auto-detects CUDA and falls back to CPU. See TRAINING.md for
a full explanation of the GPU options used here.
"""

import os
import math
import torch
import torch.nn.functional as F

from lokolm import LokoLM


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Model
vocab_size  = 256        # byte-level vocab for this demo
block_size   = 128        # context length
d_model      = 384
n_heads      = 6
n_layers     = 6
mlp_ratio    = 4

# Optimization
batch_size   = 32
max_iters    = 5000
eval_interval = 250
eval_iters   = 50
learning_rate = 3e-4
weight_decay = 0.1
grad_clip    = 1.0
warmup_iters = 100

# System
device = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 on Ampere+ (e.g. A100, 30xx/40xx), else fp16, else fp32
if device == "cuda" and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16
elif device == "cuda":
    amp_dtype = torch.float16
else:
    amp_dtype = torch.float32
compile_model = True     # torch.compile (PyTorch 2.x); set False if it errors

print(f"device={device}  amp_dtype={amp_dtype}")


# -----------------------------------------------------------------------------
# Data: a tiny byte-level dataset. Replace with your own corpus.
# -----------------------------------------------------------------------------
DATA_PATH = "input.txt"
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "rb") as f:
        raw = f.read()
else:
    # Fallback so the script runs out-of-the-box.
    raw = (b"hello world. this is a tiny decoder-only transformer demo. " * 2000)

data = torch.tensor(list(raw), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
    # pin + non_blocking lets the H2D copy overlap with compute on CUDA
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
model = LokoLM(vocab_size, block_size, d_model, n_heads, n_layers, mlp_ratio).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

if compile_model and hasattr(torch, "compile"):
    model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95)
)
# GradScaler is only needed for fp16; bf16 and fp32 don't use it.
scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))


def lr_for(it):
    # Linear warmup then cosine decay to 10% of peak.
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ratio)))


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True   # faster fp32 matmuls
    torch.backends.cudnn.allow_tf32 = True

model.train()
for it in range(max_iters):
    for g in optimizer.param_groups:
        g["lr"] = lr_for(it)

    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr_for(it):.2e}")

    x, y = get_batch("train")
    with torch.autocast(device_type=device, dtype=amp_dtype, enabled=device == "cuda"):
        _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

print("done.")
torch.save(model.state_dict(), "ckpt.pt")
print("saved checkpoint to ckpt.pt")
