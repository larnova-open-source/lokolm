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
# Model — GPT-2-small-class (~85M params at this byte-level vocab)
vocab_size  = 256        # byte-level vocab for this demo
block_size   = 512        # context length
d_model      = 768
n_heads      = 12
n_layers     = 12
mlp_ratio    = 4

# Optimization
# At this size + context, batch 32 can OOM on smaller GPUs. Start at 16 and raise
# it if VRAM allows (or use gradient accumulation — see the training docs).
batch_size   = 16
max_iters    = 5000
eval_interval = 250
eval_iters   = 50
learning_rate = 3e-4
weight_decay = 0.1
grad_clip    = 1.0
warmup_iters = 100
# LR schedule, decoupled from max_iters so resuming continues the exact same curve:
# cosine-decay from learning_rate down to min_lr over lr_decay_iters *absolute* steps,
# then hold at min_lr. Set lr_decay_iters to your intended total horizon and keep it fixed.
lr_decay_iters = max_iters
min_lr         = learning_rate * 0.1

# Checkpointing / resume
ckpt_path   = "ckpt.pt"                          # file to read/write the checkpoint
# To continue a previous run, point this at a checkpoint. On Colab you can set it without
# editing the file:  !RESUME_FROM=ckpt.pt python train.py   (None = train from scratch)
resume_from = os.environ.get("RESUME_FROM", None)

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

# Resume from a checkpoint if requested. Load weights into the raw (uncompiled) model,
# stripping any "_orig_mod." prefix left by a previous torch.compile run. The model config
# must match the checkpoint's — load_state_dict will error clearly if it doesn't.
start_iter = 0
ckpt = None
if resume_from and os.path.exists(resume_from):
    ckpt = torch.load(resume_from, map_location=device)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    start_iter = ckpt.get("iter", 0)
    # Lock the LR schedule to the original run so the curve continues exactly. (To
    # deliberately change the schedule on resume, delete this block or edit the checkpoint.)
    if "sched" in ckpt:
        s = ckpt["sched"]
        learning_rate, warmup_iters = s["learning_rate"], s["warmup_iters"]
        lr_decay_iters, min_lr = s["lr_decay_iters"], s["min_lr"]
    print(f"resumed from {resume_from} at iter {start_iter}")

if compile_model and hasattr(torch, "compile"):
    model = torch.compile(model)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95)
)
# GradScaler is only needed for fp16; bf16 and fp32 don't use it.
scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))

# Restore optimizer momentum (and the fp16 scaler) so training continues seamlessly.
if ckpt is not None:
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler.is_enabled() and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])


def lr_for(it):
    # Warmup → cosine decay from learning_rate to min_lr over lr_decay_iters → hold at min_lr.
    # A pure function of the absolute step `it`, so resuming reproduces the same curve and
    # extending the run (raising max_iters) just trains the tail at min_lr.
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it >= lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))   # 1 at decay start → 0 at the end
    return min_lr + coeff * (learning_rate - min_lr)


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


def save_checkpoint(it):
    # `it` is the next iteration to run on resume. Store weights, optimizer/scaler state,
    # and the model config (so sample.py can rebuild the exact model). Write to a temp file
    # first and rename, so a crash mid-write can't corrupt an existing checkpoint.
    tmp = ckpt_path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "iter": it,
        "config": dict(
            vocab_size=vocab_size, block_size=block_size, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, mlp_ratio=mlp_ratio,
        ),
        # LR-schedule params, so a resumed run reproduces the exact same curve.
        "sched": dict(
            learning_rate=learning_rate, warmup_iters=warmup_iters,
            lr_decay_iters=lr_decay_iters, min_lr=min_lr,
        ),
    }, tmp)
    os.replace(tmp, ckpt_path)


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True   # faster fp32 matmuls
    torch.backends.cudnn.allow_tf32 = True

model.train()
for it in range(start_iter, max_iters):
    for g in optimizer.param_groups:
        g["lr"] = lr_for(it)

    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr_for(it):.2e}")
        save_checkpoint(it)   # periodic save — resume-safe, survives Colab disconnects

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
save_checkpoint(max_iters)
print(f"saved checkpoint to {ckpt_path}")
