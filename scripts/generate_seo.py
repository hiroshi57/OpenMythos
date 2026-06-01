#!/usr/bin/env python3
"""
OpenMythos SEO / LLMO 最適化コンテンツ生成パイプライン。

OpenMythosモデルを使ってコンテンツを生成し、LLMOスコアで品質を評価する
エンドツーエンドパイプライン。

スタイル:
    answer_first  -- 冒頭で問いに直接答え、理由・詳細・例を続ける
    faq           -- Q&A 形式。AIサーチで頻出構造
    entity_rich   -- エンティティ(数値・固有名詞・統計)を多用した詳細解説

使い方::

    python scripts/generate_seo.py \\
        --prompt "OpenMythosとは何ですか？" \\
        --style answer_first \\
        --max-tokens 200

    python scripts/generate_seo.py \\
        --prompt "LLMOとSEOの違い" \\
        --style faq \\
        --out-dir results/seo
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Literal

import torch

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_mythos.main import MythosConfig, OpenMythos
from open_mythos.llmo import LLMOScorer

# ---------------------------------------------------------------------------
# 型定義
# ---------------------------------------------------------------------------

SEOStyle = Literal["answer_first", "faq", "entity_rich"]

# ---------------------------------------------------------------------------
# スタイル別システムプレフィックス
# ---------------------------------------------------------------------------

STYLE_PREFIXES: dict[str, str] = {
    "answer_first": (
        "[System]: You are an SEO/LLMO expert writer. "
        "Always start with a direct, concise answer in the first sentence. "
        "Then provide supporting details, examples, and data.\n"
    ),
    "faq": (
        "[System]: You are an FAQ writer optimized for AI search engines. "
        "Format your response as Q&A pairs. "
        "Each answer should be 2-4 sentences, factual, and entity-rich.\n"
    ),
    "entity_rich": (
        "[System]: You are a technical writer. "
        "Include specific numbers, dates, proper nouns, and technical terms. "
        "Support every claim with data or examples.\n"
    ),
}

# ---------------------------------------------------------------------------
# トークナイズ (cli.py と同じシンプル実装)
# ---------------------------------------------------------------------------


def _tokenize(text: str, vocab_size: int) -> list[int]:
    """テキストをトークンID列に変換する (文字単位の簡易実装)。"""
    return [ord(c) % vocab_size for c in text]


def _detokenize(ids: list[int]) -> str:
    """トークンIDを文字に変換する。"""
    chars = []
    for i in ids:
        try:
            c = chr(i)
            if c.isprintable() or c in "\n\t ":
                chars.append(c)
        except (ValueError, OverflowError):
            pass
    return "".join(chars)


# ---------------------------------------------------------------------------
# コンテンツ生成
# ---------------------------------------------------------------------------


def build_prompt(user_prompt: str, style: SEOStyle) -> str:
    """スタイルに応じたプロンプトを構築する。"""
    prefix = STYLE_PREFIXES.get(style, STYLE_PREFIXES["answer_first"])
    return f"{prefix}[User]: {user_prompt}\n[Assistant]:"


def generate_seo_content(
    model: OpenMythos,
    device: str,
    prompt: str,
    style: SEOStyle = "answer_first",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    loops: int = 6,
) -> dict:
    """
    SEO/LLMO最適化コンテンツを生成する。

    Args:
        model          -- OpenMythosモデル
        device         -- torch device文字列
        prompt         -- ユーザープロンプト
        style          -- 生成スタイル
        max_new_tokens -- 最大生成トークン数
        temperature    -- サンプリング温度
        top_p          -- nucleus sampling 閾値
        loops          -- 推論ループ数 (高いほど精度↑ 速度↓)

    Returns:
        {
            "text": str,              # 生成テキスト
            "style": str,
            "llmo_score": LLMOScore,  # LLMOスコア
            "latency_ms": float,
            "tokens_generated": int,
            "loops_used": int,
        }
    """
    scorer = LLMOScorer()
    full_prompt = build_prompt(prompt, style)

    vocab_size = model.cfg.vocab_size
    max_prompt_len = max(1, model.cfg.max_seq_len - max_new_tokens - 4)
    ids = _tokenize(full_prompt, vocab_size)[:max_prompt_len]
    if not ids:
        ids = [0]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    t0 = time.perf_counter()

    # n_loops でテスト時計算量を制御 (OpenMythosの差別化機能)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n_loops=loops,
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    generated_ids = output[0, input_ids.shape[1]:].tolist()
    generated_text = _detokenize(generated_ids)

    llmo_score = scorer.score(generated_text)

    return {
        "prompt": prompt,
        "style": style,
        "text": generated_text,
        "llmo_score": {
            "entity_density": llmo_score.entity_density,
            "answer_directness": llmo_score.answer_directness,
            "citability": llmo_score.citability,
            "llmo_total": llmo_score.llmo_total,
            "entities": llmo_score.entities,
            "word_count": llmo_score.word_count,
        },
        "latency_ms": round(latency_ms, 2),
        "tokens_generated": len(generated_ids),
        "loops_used": loops,
    }


def generate_all_styles(
    model: OpenMythos,
    device: str,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    loops: int = 6,
) -> list[dict]:
    """3スタイル全てでコンテンツを生成し、LLMOスコア比較する。"""
    results = []
    for style in ("answer_first", "faq", "entity_rich"):
        result = generate_seo_content(
            model=model,
            device=device,
            prompt=prompt,
            style=style,  # type: ignore[arg-type]
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            loops=loops,
        )
        results.append(result)
    # LLMOスコア降順でソート
    results.sort(key=lambda r: r["llmo_score"]["llmo_total"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# レポート出力
# ---------------------------------------------------------------------------


def save_results(results: list[dict], out_dir: Path, prompt: str) -> Path:
    """生成結果をJSONLに保存する。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_prompt = re.sub(r"[^\w\s-]", "", prompt[:30]).strip().replace(" ", "_")
    out_path = out_dir / f"seo_gen_{safe_prompt}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def print_results(results: list[dict]) -> None:
    """結果をコンソールに出力する。"""
    print("\n" + "=" * 60)
    print("SEO/LLMO Generation Results")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        score = r["llmo_score"]
        print(f"\n#{i} Style: {r['style'].upper()}")
        print(f"  LLMO Total     : {score['llmo_total']:.4f}")
        print(f"  Entity Density : {score['entity_density']:.4f}")
        print(f"  Ans Directness : {score['answer_directness']:.4f}")
        print(f"  Citability     : {score['citability']:.4f}")
        print(f"  Words          : {score['word_count']}")
        print(f"  Tokens gen     : {r['tokens_generated']}")
        print(f"  Latency        : {r['latency_ms']:.1f}ms")
        print(f"  Loops          : {r['loops_used']}")
        if score["entities"]:
            print(f"  Entities       : {', '.join(score['entities'][:8])}")
        print(f"  Text (preview) : {r['text'][:120].strip()!r}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _build_tiny_model(device: str) -> OpenMythos:
    """テスト用の最小モデルを構築する。"""
    cfg = MythosConfig(
        vocab_size=512,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=256,
        max_loop_iters=8,
        prelude_layers=1,
        coda_layers=1,
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=1,
        expert_dim=32,
    )
    return OpenMythos(cfg).to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenMythos SEO/LLMO コンテンツ生成パイプライン"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="生成プロンプト (例: 'LLMOとは何ですか？')",
    )
    parser.add_argument(
        "--style",
        choices=["answer_first", "faq", "entity_rich", "all"],
        default="all",
        help="生成スタイル。'all' で3スタイル全て比較 (デフォルト: all)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="最大生成トークン数 (デフォルト: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="サンプリング温度 (デフォルト: 0.8)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling 閾値 (デフォルト: 0.9)",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=6,
        help="推論ループ数。高いほど精度↑ 速度↓ (デフォルト: 6)",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="モデルチェックポイントパス (.pt)。省略時はランダム重み",
    )
    parser.add_argument(
        "--out-dir",
        default="results/seo",
        help="結果保存ディレクトリ (デフォルト: results/seo)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="結果をファイル保存しない",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # モデル構築
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg: MythosConfig = ckpt["cfg"]
        model = OpenMythos(cfg)
        model.load_state_dict(ckpt["model"])
        model = model.to(device).eval()
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        model = _build_tiny_model(device)
        print("Using random-weight model (no checkpoint)")

    # 生成
    if args.style == "all":
        results = generate_all_styles(
            model=model,
            device=device,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            loops=args.loops,
        )
    else:
        result = generate_seo_content(
            model=model,
            device=device,
            prompt=args.prompt,
            style=args.style,  # type: ignore[arg-type]
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            loops=args.loops,
        )
        results = [result]

    print_results(results)

    if not args.no_save:
        out_path = save_results(results, Path(args.out_dir), args.prompt)
        print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
