"""
Minimal inference / text generation for lokoLM.

Loads a trained checkpoint (ckpt.pt) and autoregressively generates text from a
prompt, using the model's own `generate()` method (temperature + top-k sampling).

Run:
    python sample.py --prompt "hello" --max-new-tokens 200

The checkpoint stores the model config (written by train.py), so this script rebuilds
the exact trained model without needing to mirror train.py's settings. The model is
byte-level (vocab_size=256), so a prompt is just its UTF-8 bytes and generated tokens
decode straight back to text.
"""

import argparse

import torch

from lokolm import LokoLM


def main():
    p = argparse.ArgumentParser(description="Generate text with a trained lokoLM.")
    p.add_argument("--ckpt", default="ckpt.pt", help="checkpoint path")
    p.add_argument("--prompt", default="\n", help="text prompt to continue")
    p.add_argument("--max-new-tokens", type=int, default=200, help="bytes to generate")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="<1.0 = more confident, >1.0 = more random")
    p.add_argument("--top-k", type=int, default=40,
                   help="sample only from the top-k bytes (0 disables)")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the checkpoint and rebuild the exact model it was trained as.
    ckpt = torch.load(args.ckpt, map_location=device)
    model = LokoLM(**ckpt["config"]).to(device)
    state = ckpt["model"]
    # Weights saved from a torch.compile'd model carry a "_orig_mod." prefix.
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Encode the prompt as raw bytes -> token ids.
    prompt_ids = list(args.prompt.encode("utf-8")) or [ord("\n")]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    top_k = args.top_k if args.top_k > 0 else None
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=top_k)

    # Decode bytes back to text (replace any invalid byte sequences rather than crash).
    text = bytes(out[0].tolist()).decode("utf-8", errors="replace")
    print(text)


if __name__ == "__main__":
    main()
