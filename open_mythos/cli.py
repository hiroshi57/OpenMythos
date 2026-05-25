"""
OpenMythos CLI — minimal command-line interface.

Usage examples:

    # Generate text (random weights — useful for smoke-testing)
    mythos generate --prompt "Once upon a time" --max-tokens 100

    # Generate from a saved checkpoint
    mythos generate --checkpoint path/to/model.pt --prompt "The answer is"

    # Stream output token by token
    mythos generate --prompt "Hello" --stream

    # Show model info
    mythos info --variant 1b
"""

from __future__ import annotations

import argparse
import sys

import torch

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.variants import (
    mythos_nano,
    mythos_1b,
    mythos_3b,
    mythos_7b,
    mythos_10b,
)

_VARIANTS = {
    "nano": mythos_nano,
    "1b": mythos_1b,
    "3b": mythos_3b,
    "7b": mythos_7b,
    "10b": mythos_10b,
}


def _build_model(args) -> tuple[OpenMythos, int]:
    """Load or construct a model from CLI args. Returns (model, vocab_size)."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg: MythosConfig = ckpt["cfg"]
        vocab_size: int = ckpt.get("vocab_size", cfg.vocab_size)
        model = OpenMythos(cfg)
        model.load_state_dict(ckpt["model"])
    else:
        variant_fn = _VARIANTS.get(args.variant or "nano")
        if variant_fn is None:
            print(
                f"Unknown variant '{args.variant}'. Choose from: {list(_VARIANTS)}",
                file=sys.stderr,
            )
            sys.exit(1)
        cfg = variant_fn()
        vocab_size = cfg.vocab_size
        model = OpenMythos(cfg)

    model = model.to(device).eval()
    return model, vocab_size, device


def _tokenize(prompt: str, vocab_size: int) -> list[int]:
    """Tokenize prompt, clipping all IDs to the model's vocab_size."""
    try:
        from open_mythos.tokenizer import MythosTokenizer

        enc = MythosTokenizer()
        ids = enc.encode(prompt)
    except Exception:
        ids = [ord(c) for c in prompt]
    return [min(i, vocab_size - 1) for i in ids]


def _detokenize(ids: list[int]) -> str:
    try:
        from open_mythos.tokenizer import MythosTokenizer

        enc = MythosTokenizer()
        return enc.decode(ids)
    except Exception:
        return bytes(min(i, 255) for i in ids).decode("utf-8", errors="replace")


def _safe_print(text: str, end: str = "\n", flush: bool = False) -> None:
    """Write text to stdout, replacing unencodable chars for the active console."""
    buf = sys.stdout
    if hasattr(buf, "buffer"):
        raw = text.encode("utf-8", errors="replace")
        buf.buffer.write(raw + end.encode())
        if flush:
            buf.buffer.flush()
    else:
        print(text, end=end, flush=flush)


def cmd_generate(args) -> None:
    model, vocab_size, device = _build_model(args)

    prompt_ids = _tokenize(args.prompt, vocab_size)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    if args.stream:
        _safe_print(args.prompt, end="", flush=True)
        for token_ids in model.generate_stream(**kwargs):
            token = _detokenize([token_ids[0, -1].item()])
            _safe_print(token, end="", flush=True)
        _safe_print("")
    else:
        output_ids = model.generate(**kwargs)
        all_ids = output_ids[0].tolist()
        _safe_print(_detokenize(all_ids))


def cmd_info(args) -> None:
    variant_fn = _VARIANTS.get(args.variant or "nano")
    if variant_fn is None:
        print(
            f"Unknown variant '{args.variant}'. Choose from: {list(_VARIANTS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    cfg = variant_fn()
    model = OpenMythos(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Variant      : mythos_{args.variant or 'nano'}")
    print(f"Parameters   : {n_params:,}")
    print(f"Hidden dim   : {cfg.dim}")
    print(f"Attn type    : {cfg.attn_type}")
    print(f"Experts      : {cfg.n_experts} routed + {cfg.n_shared_experts} shared")
    print(f"Loop iters   : {cfg.max_loop_iters}")
    print(f"Max seq len  : {cfg.max_seq_len}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mythos",
        description="OpenMythos CLI — generate text and inspect model configs",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = sub.add_parser("generate", help="Generate text from a prompt")
    gen.add_argument("--prompt", required=True, help="Input prompt string")
    gen.add_argument("--checkpoint", default=None, help="Path to a .pt checkpoint")
    gen.add_argument(
        "--variant",
        default="nano",
        choices=list(_VARIANTS),
        help="Model variant (default: nano)",
    )
    gen.add_argument("--max-tokens", type=int, default=100)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--top-k", type=int, default=50)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument(
        "--stream", action="store_true", help="Stream tokens as they are generated"
    )
    gen.add_argument("--device", default=None, help="Device: cpu / cuda / cuda:0")

    # --- info ---
    info = sub.add_parser("info", help="Print model config and parameter count")
    info.add_argument("--variant", default="nano", choices=list(_VARIANTS))

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
