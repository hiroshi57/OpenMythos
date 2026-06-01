#!/usr/bin/env python3
"""
OpenMythos Benchmark Runner — perplexity + lm-eval in one shot.

Runs perplexity evaluation (WikiText-2) and optionally lm-evaluation-harness
tasks (HellaSwag, ARC-Easy, WinoGrande) and saves the results as JSON.
Optionally updates the README.md benchmark table.

Usage
-----
# Quick test (nano, hellaswag only, 10 samples)
python benchmark/run_eval.py --variant nano --tasks hellaswag --limit 10

# Full evaluation from checkpoint
python benchmark/run_eval.py --checkpoint checkpoints/pretrain/ckpt_step10000.pt \
    --tasks hellaswag,arc_easy,winogrande --device cuda

# Skip README update
python benchmark/run_eval.py --variant nano --no-readme-update
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Model loader (shared with perplexity.py / lm_eval_harness.py)
# ---------------------------------------------------------------------------


def _load_model_and_device(
    variant: Optional[str],
    checkpoint: Optional[str],
    device: Optional[str],
):
    from open_mythos.main import OpenMythos
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

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = OpenMythos(ckpt["cfg"])
        model.load_state_dict(ckpt["model"])
    else:
        fn = _VARIANTS.get(variant or "nano")
        if fn is None:
            raise ValueError(f"Unknown variant: {variant}")
        model = OpenMythos(fn())

    return model.to(dev).eval(), dev


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------


def run_perplexity_eval(
    variant: Optional[str],
    checkpoint: Optional[str],
    device: Optional[str],
) -> dict:
    """Run WikiText-2 test-set PPL evaluation. Returns result dict."""
    from benchmark.perplexity import evaluate_perplexity, _load_corpus, _tokenize_corpus

    model, dev = _load_model_and_device(variant, checkpoint, device)

    print("[run_eval] running perplexity (WikiText-2 test)…")
    try:
        text = _load_corpus("wikitext-2-raw-v1", split="test")
        token_ids = _tokenize_corpus(text, model.cfg.vocab_size)
        result = evaluate_perplexity(model, token_ids, dev, seq_len=512, stride=256)
        return {
            "wikitext2_test": {
                "ppl": round(result["ppl"], 4),
                "nll": round(result["nll"], 4),
                "n_tokens": result["n_tokens"],
                "elapsed_sec": round(result.get("elapsed_sec", 0), 2),
            }
        }
    except Exception as e:
        print(f"[run_eval] perplexity skipped: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# lm-eval evaluation
# ---------------------------------------------------------------------------


def run_lm_eval(
    variant: Optional[str],
    checkpoint: Optional[str],
    tasks: list[str],
    device: Optional[str],
    num_fewshot: int = 0,
    limit: Optional[int] = None,
) -> dict:
    """Run lm-evaluation-harness tasks. Returns per-task result dict."""
    from benchmark.lm_eval_harness import MythosLMEvalWrapper, _run_lambada_standalone

    model, dev = _load_model_and_device(variant, checkpoint, device)

    try:
        import lm_eval

        wrapper = MythosLMEvalWrapper(model=model, device=dev)
        print(f"[run_eval] running lm-eval tasks: {tasks}  limit={limit}")
        results = lm_eval.simple_evaluate(
            model=wrapper,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            log_samples=False,
        )
        out = {}
        for task, metrics in results["results"].items():
            out[task] = {
                k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)
            }
        return out

    except ImportError:
        # Fallback: standalone LAMBADA if lm-eval is not installed
        print("[run_eval] lm-eval not installed — running standalone LAMBADA fallback")
        try:
            acc = _run_lambada_standalone(model, dev, n_samples=limit or 100)
            return {"lambada_standalone": {"acc": round(acc, 4)}}
        except Exception as e:
            return {"error": f"lm-eval not installed and fallback failed: {e}"}


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def save_results(results: dict, variant: str, out_dir: Path) -> Path:
    """Save results to benchmark/results/{variant}_{timestamp}.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"{variant}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[run_eval] results saved: {path}")
    return path


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

_TABLE_START = "<!-- BENCHMARK_TABLE_START -->"
_TABLE_END = "<!-- BENCHMARK_TABLE_END -->"


def _build_table(results_dir: Path) -> str:
    """Build a markdown table from all JSON files in results_dir."""
    rows = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        v = data.get("variant", p.stem.split("_")[0])
        ppl = data.get("perplexity", {}).get("wikitext2_test", {}).get("ppl", "—")
        lm = data.get("lm_eval", {})
        hellaswag = lm.get("hellaswag", {}).get(
            "acc_norm", lm.get("hellaswag", {}).get("acc", "—")
        )
        arc = lm.get("arc_easy", {}).get("acc", "—")
        wino = lm.get("winogrande", {}).get("acc", "—")
        ts = data.get("timestamp", "")[:10]
        rows.append(f"| {v} | {ppl} | {hellaswag} | {arc} | {wino} | {ts} |")

    if not rows:
        return ""

    header = (
        "| Variant | PPL (Wiki2) | HellaSwag | ARC-Easy | WinoGrande | Date |\n"
        "|---------|------------|-----------|----------|------------|------|\n"
    )
    return _TABLE_START + "\n" + header + "\n".join(rows) + "\n" + _TABLE_END


def update_readme(results_dir: Path, readme_path: Path) -> None:
    """Update README.md benchmark table between sentinel comments."""
    table = _build_table(results_dir)
    if not table:
        print("[run_eval] no results to add to README")
        return

    content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

    if _TABLE_START in content and _TABLE_END in content:
        start = content.index(_TABLE_START)
        end = content.index(_TABLE_END) + len(_TABLE_END)
        new_content = content[:start] + table + content[end:]
    else:
        # Try to insert after "## Benchmarks" heading
        marker = "\n## Benchmarks"
        if marker in content:
            idx = content.index(marker) + len(marker)
            new_content = content[:idx] + "\n\n" + table + "\n" + content[idx:]
        else:
            new_content = content + "\n\n## Benchmarks\n\n" + table + "\n"

    readme_path.write_text(new_content, encoding="utf-8")
    print(f"[run_eval] README.md updated: {readme_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenMythos benchmark runner")
    p.add_argument(
        "--variant",
        default="nano",
        choices=["nano", "1b", "3b", "7b", "10b"],
        help="Model variant (ignored if --checkpoint is given)",
    )
    p.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint file")
    p.add_argument(
        "--tasks",
        default="hellaswag,arc_easy,winogrande",
        help="Comma-separated lm-eval task names",
    )
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        dest="num_fewshot",
        help="Number of few-shot examples",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit evaluation to this many samples (for quick testing)",
    )
    p.add_argument(
        "--out-dir",
        default="benchmark/results",
        dest="out_dir",
        help="Directory to save result JSON files",
    )
    p.add_argument(
        "--no-perplexity",
        action="store_true",
        dest="no_perplexity",
        help="Skip perplexity evaluation",
    )
    p.add_argument(
        "--no-lm-eval",
        action="store_true",
        dest="no_lm_eval",
        help="Skip lm-eval tasks",
    )
    p.add_argument(
        "--no-readme-update",
        action="store_true",
        dest="no_readme_update",
        help="Do not update README.md",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    variant = args.variant

    # Infer variant label from checkpoint filename if not specified
    if args.checkpoint and args.variant == "nano":
        stem = Path(args.checkpoint).stem
        for v in ["10b", "7b", "3b", "1b", "nano"]:
            if v in stem:
                variant = v
                break

    # Resolve device once
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    results: dict = {
        "variant": variant,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "checkpoint": args.checkpoint,
        "git_sha": _git_sha(),
    }

    # Model params (load once for metadata)
    try:
        _, dev = _load_model_and_device(
            variant if not args.checkpoint else None, args.checkpoint, device
        )
        from open_mythos.main import OpenMythos

        if args.checkpoint:
            import torch as _t

            ckpt = _t.load(args.checkpoint, map_location="cpu", weights_only=False)
            _m = OpenMythos(ckpt["cfg"])
        else:
            from open_mythos.variants import (
                mythos_nano,
                mythos_1b,
                mythos_3b,
                mythos_7b,
                mythos_10b,
            )

            _vfns = {
                "nano": mythos_nano,
                "1b": mythos_1b,
                "3b": mythos_3b,
                "7b": mythos_7b,
                "10b": mythos_10b,
            }
            _m = OpenMythos(_vfns[variant]())
        results["model_params"] = sum(p.numel() for p in _m.parameters())
    except Exception:
        pass

    # Perplexity
    if not args.no_perplexity:
        results["perplexity"] = run_perplexity_eval(
            variant if not args.checkpoint else None,
            args.checkpoint,
            args.device,
        )

    # lm-eval
    if not args.no_lm_eval:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        results["lm_eval"] = run_lm_eval(
            variant if not args.checkpoint else None,
            args.checkpoint,
            tasks,
            args.device,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
        )

    # Save
    save_results(results, variant, out_dir)

    # README
    if not args.no_readme_update:
        readme = Path("README.md")
        if readme.exists():
            update_readme(out_dir, readme)

    # Summary
    print("\n── Benchmark Summary ──────────────────────────────")
    if "perplexity" in results:
        ppl_info = results["perplexity"].get("wikitext2_test", results["perplexity"])
        print(f"  PPL (WikiText-2): {ppl_info.get('ppl', '—')}")
    if "lm_eval" in results:
        for task, metrics in results["lm_eval"].items():
            if isinstance(metrics, dict):
                acc = metrics.get("acc_norm", metrics.get("acc", "—"))
                print(f"  {task}: acc={acc}")
    print("────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
