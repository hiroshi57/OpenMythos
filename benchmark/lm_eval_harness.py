#!/usr/bin/env python3
"""
OpenMythos — lm-evaluation-harness integration.

Registers ``OpenMythos`` as an ``lm_eval`` compatible model so that any task
supported by EleutherAI's lm-evaluation-harness can be run against it.

Usage
-----
# Install harness: pip install lm-eval
# Run HellaSwag on nano variant (0-shot)
python benchmark/lm_eval_harness.py --variant nano --tasks hellaswag --num-fewshot 0

# Run multiple tasks
python benchmark/lm_eval_harness.py --variant 1b --tasks arc_easy,winogrande --device cuda

# From checkpoint
python benchmark/lm_eval_harness.py --checkpoint path/to/model.pt --tasks lambada_openai

Standalone mode (no lm-eval installed)
---------------------------------------
If ``lm_eval`` is not installed, the script falls back to a built-in
log-likelihood scorer that evaluates ``lambada`` via the HuggingFace
``datasets`` library and reports accuracy directly.
"""

from __future__ import annotations

import argparse
import math
from typing import Any, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.variants import (
    mythos_nano, mythos_1b, mythos_3b, mythos_7b, mythos_10b,
)

_VARIANTS = {
    "nano": mythos_nano,
    "1b": mythos_1b,
    "3b": mythos_3b,
    "7b": mythos_7b,
    "10b": mythos_10b,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(args) -> tuple[OpenMythos, str]:
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg: MythosConfig = ckpt["cfg"]
        model = OpenMythos(cfg)
        model.load_state_dict(ckpt["model"])
    else:
        fn = _VARIANTS.get(args.variant or "nano")
        if fn is None:
            raise ValueError(f"Unknown variant '{args.variant}'")
        cfg = fn()
        model = OpenMythos(cfg)
    return model.to(dev).eval(), dev


def _tokenize(text: str, vocab_size: int) -> list[int]:
    try:
        from open_mythos.tokenizer import MythosTokenizer
        enc = MythosTokenizer()
        ids = enc.encode(text)
    except Exception:
        ids = [ord(c) for c in text]
    return [min(i, vocab_size - 1) for i in ids]


# ---------------------------------------------------------------------------
# lm-eval compatible model wrapper
# ---------------------------------------------------------------------------

try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
    _HAS_LM_EVAL = True
except ImportError:
    _HAS_LM_EVAL = False
    LM = object  # type: ignore[assignment,misc]


class MythosLMEvalWrapper(LM):
    """Wrap OpenMythos for lm-evaluation-harness.

    Implements the three core scoring primitives required by the harness:
    * ``loglikelihood``        — P(continuation | context)
    * ``loglikelihood_rolling`` — token-level NLL over a full sequence
    * ``generate_until``       — greedy/sampled generation until stop string
    """

    def __init__(
        self,
        model: OpenMythos,
        device: str,
        batch_size: int = 1,
        max_length: Optional[int] = None,
        n_loops: Optional[int] = None,
    ) -> None:
        if _HAS_LM_EVAL:
            super().__init__()
        self._model = model
        self._device = device
        self._batch_size = batch_size
        self._vocab_size = model.cfg.vocab_size
        self._max_length = max_length or model.cfg.max_seq_len
        self._n_loops = n_loops

    # ------------------------------------------------------------------
    # Properties required by harness
    # ------------------------------------------------------------------

    @property
    def eot_token_id(self) -> int:
        return 0

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Core scoring helpers
    # ------------------------------------------------------------------

    def _forward(self, ids: torch.Tensor) -> torch.Tensor:
        kwargs = {}
        if self._n_loops is not None:
            kwargs["n_loops"] = self._n_loops
        with torch.no_grad():
            return self._model(ids, **kwargs)  # (B, T, V)

    def _score_continuation(
        self, context_ids: list[int], continuation_ids: list[int]
    ) -> tuple[float, bool]:
        """Return (sum_log_prob, is_greedy) for continuation given context."""
        all_ids = (context_ids + continuation_ids)[: self._max_length]
        ids_t = torch.tensor(all_ids, dtype=torch.long, device=self._device).unsqueeze(0)
        logits = self._forward(ids_t)  # (1, T, V)

        ctx_len = min(len(context_ids), len(all_ids) - 1)
        cont_len = len(all_ids) - ctx_len

        lp_start = max(ctx_len - 1, 0)
        log_probs = F.log_softmax(logits[0, lp_start: lp_start + cont_len], dim=-1)
        cont_t = torch.tensor(
            all_ids[ctx_len: ctx_len + cont_len],
            dtype=torch.long,
            device=self._device,
        )
        if cont_t.shape[0] == 0 or log_probs.shape[0] == 0:
            return 0.0, True

        nll = F.nll_loss(log_probs, cont_t, reduction="sum").item()
        greedy = (log_probs.argmax(dim=-1) == cont_t).all().item()
        return -nll, bool(greedy)

    # ------------------------------------------------------------------
    # Required harness interface methods
    # ------------------------------------------------------------------

    def tok_encode(self, string: str) -> list[int]:
        return _tokenize(string, self._vocab_size)

    def tok_decode(self, tokens: list[int]) -> str:
        try:
            from open_mythos.tokenizer import MythosTokenizer
            enc = MythosTokenizer()
            return enc.decode(tokens)
        except Exception:
            return "".join(chr(min(t, 127)) for t in tokens)

    def loglikelihood(
        self, requests: list
    ) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            ctx, cont = req.args
            ctx_ids = _tokenize(ctx, self._vocab_size) if isinstance(ctx, str) else list(ctx)
            cont_ids = _tokenize(cont, self._vocab_size) if isinstance(cont, str) else list(cont)
            results.append(self._score_continuation(ctx_ids, cont_ids))
        return results

    def loglikelihood_rolling(
        self, requests: list
    ) -> list[float]:
        results = []
        for req in requests:
            text = req.args[0]
            ids = _tokenize(text, self._vocab_size)
            if len(ids) < 2:
                results.append(0.0)
                continue
            total_nll = 0.0
            stride = self._max_length // 2
            for begin in range(0, len(ids) - 1, stride):
                end = min(begin + self._max_length, len(ids) - 1)
                chunk = ids[begin: end + 1]
                ids_t = torch.tensor(chunk, dtype=torch.long, device=self._device).unsqueeze(0)
                logits = self._forward(ids_t)
                lp = F.log_softmax(logits[0, :-1], dim=-1)
                tgt = torch.tensor(chunk[1:], dtype=torch.long, device=self._device)
                total_nll += F.nll_loss(lp, tgt, reduction="sum").item()
                if end >= len(ids) - 1:
                    break
            results.append(-total_nll)
        return results

    def generate_until(self, requests: list) -> list[str]:
        results = []
        for req in requests:
            ctx = req.args[0]
            gen_kwargs = req.args[1] if len(req.args) > 1 else {}
            stop_strings: list[str] = gen_kwargs.get("until", [])
            max_new = gen_kwargs.get("max_gen_toks", self.max_gen_toks)

            ids = _tokenize(ctx, self._vocab_size) if isinstance(ctx, str) else list(ctx)
            if not ids:
                ids = [0]
            cur = torch.tensor(ids, dtype=torch.long, device=self._device).unsqueeze(0)
            generated_ids: list[int] = []

            for _ in range(max_new):
                logits = self._forward(cur)
                next_id = logits[0, -1, :].argmax().item()
                generated_ids.append(next_id)
                cur = torch.cat(
                    [cur, torch.tensor([[next_id]], dtype=torch.long, device=self._device)], dim=1
                )
                decoded_so_far = self.tok_decode(generated_ids)
                if any(s in decoded_so_far for s in stop_strings):
                    break

            text = self.tok_decode(generated_ids)
            for s in stop_strings:
                if s in text:
                    text = text[: text.index(s)]
            results.append(text)
        return results


# ---------------------------------------------------------------------------
# Standalone fallback: lambada accuracy
# ---------------------------------------------------------------------------

def _run_lambada_standalone(model: OpenMythos, device: str, n_samples: int = 200) -> None:
    """Fallback evaluation on LAMBADA when lm-eval is not installed."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets to run standalone lambada eval")

    ds = load_dataset("lambada", split="test", trust_remote_code=True)
    wrapper = MythosLMEvalWrapper(model, device)
    correct = 0
    total = 0
    for item in list(ds)[:n_samples]:
        text: str = item["text"]
        words = text.split()
        if len(words) < 2:
            continue
        context = " ".join(words[:-1])
        last_word = words[-1]
        ctx_ids = _tokenize(context, model.cfg.vocab_size)
        last_ids = _tokenize(" " + last_word, model.cfg.vocab_size)
        score, is_greedy = wrapper._score_continuation(ctx_ids, last_ids)
        if is_greedy:
            correct += 1
        total += 1

    acc = correct / max(total, 1)
    print(f"\nLAMBADA (standalone, {total} samples)")
    print(f"  Accuracy: {acc:.4f}  ({correct}/{total})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenMythos lm-evaluation-harness integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--variant", default="nano", choices=list(_VARIANTS))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tasks", default="lambada_openai",
                   help="Comma-separated lm-eval task names")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-loops", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Limit samples per task (for quick smoke tests)")
    p.add_argument("--output-path", default=None,
                   help="Write lm-eval results JSON to this path")
    p.add_argument("--standalone-lambada", action="store_true",
                   help="Run built-in LAMBADA eval without lm-eval installed")
    p.add_argument("--standalone-samples", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model, device = _load_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model  : {args.variant or 'checkpoint'}  ({n_params:,} params)")
    print(f"Device : {device}")

    if args.standalone_lambada or not _HAS_LM_EVAL:
        if not _HAS_LM_EVAL and not args.standalone_lambada:
            print("lm_eval not installed — falling back to standalone LAMBADA eval.")
        _run_lambada_standalone(model, device, n_samples=args.standalone_samples)
        return

    import lm_eval

    wrapper = MythosLMEvalWrapper(
        model, device,
        batch_size=args.batch_size,
        n_loops=args.n_loops,
    )
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"Tasks  : {tasks}")
    print(f"Shots  : {args.num_fewshot}")

    results = lm_eval.simple_evaluate(
        model=wrapper,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=False,
    )

    import json
    summary = results.get("results", {})
    print("\n" + "─" * 60)
    for task, metrics in summary.items():
        print(f"  {task}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    print("─" * 60)

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
