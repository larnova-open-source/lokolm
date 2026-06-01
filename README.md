# lokoLM — model

The Python implementation of lokoLM and its training loop. Run commands from this `model/`
directory so the `lokolm` package is importable.

```
model/
├── lokolm/
│   ├── causal_self_attention.py   # multi-head causal self-attention (uses scaled_dot_product_attention)
│   ├── mlp.py                      # position-wise feed-forward network
│   ├── model.py                    # Block + LokoLM model
│   └── __init__.py                 # exports: from lokolm import LokoLM
├── train.py                        # training loop with CUDA / AMP / torch.compile
└── sample.py                       # load a checkpoint and generate text (inference)
```

## Usage

```powershell
# Smoke-test that the model wires together
python -m lokolm.model

# Train (auto-detects CUDA, falls back to CPU)
python train.py

# Generate text from the trained checkpoint (ckpt.pt)
python sample.py --prompt "hello" --max-new-tokens 200
```

```python
from lokolm import LokoLM

model = LokoLM(vocab_size=256, block_size=512,
               d_model=768, n_heads=12, n_layers=12)
logits, loss = model(idx, targets)
```

Full architecture and training documentation lives in the
[lokolm-docs](https://github.com/larnova-open-source/lokolm-docs) repository.

## Contributing & license

Contributions focus on **training data** and **training checkpoints** — and all data must be
**unlabelled raw text**. See [CONTRIBUTING.md](CONTRIBUTING.md). Released under the
[MIT License](LICENSE) © 2026 Mahmud Suberu / Larnova.

Maintained by **Mahmud Suberu** — Founder & CEO of Larnova. Contact:
[LinkedIn](https://www.linkedin.com/in/mahmud-adinoyi-684020235/).
