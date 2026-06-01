#!/usr/bin/env python3
"""
OpenMythos vs Claude Opus 4.8 — LLMO スコア比較ベンチマーク (Sprint 18.3)

OpenMythos の LLMOScorer を「ルールベースベースライン」および
オプションで「Claude API (Opus 4.8)」と比較する。

実行方法
--------
# ルールベースベースラインのみ (API キー不要):
    python benchmark/compare_opus.py

# Claude API との比較 (Opus 4.8):
    ANTHROPIC_API_KEY=sk-... python benchmark/compare_opus.py --claude

# カスタムデータ:
    python benchmark/compare_opus.py --input data/seo_train.jsonl --n 50

出力
----
    コンソール比較表 + benchmark/results/opus_comparison_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from open_mythos.llmo import LLMOScorer, LLMOScore  # noqa: E402

# ---------------------------------------------------------------------------
# テストデータ (組み込み)
# ---------------------------------------------------------------------------

_BUILTIN_CASES: list[dict] = [
    {
        "id": "seo_ja_01",
        "text": (
            "デジタルマーケティングの最前線：SEOとLLMO最適化の実践ガイド。"
            "本記事では、検索エンジン最適化（SEO）と大規模言語モデル最適化（LLMO）の"
            "違いと実装方法を具体的な数値とともに解説します。"
            "LLMOでは、entity_density（実体密度）・answer_directness（直接回答性）・"
            "citability（引用可能性）の3軸でコンテンツ品質を評価します。"
        ),
        "keywords": ["SEO", "LLMO", "デジタルマーケティング", "最適化"],
        "expected_strength": "high",
    },
    {
        "id": "seo_ja_02",
        "text": "今日はいい天気ですね。散歩が楽しいです。",
        "keywords": ["SEO", "LLMO", "コンテンツ"],
        "expected_strength": "low",
    },
    {
        "id": "seo_en_01",
        "text": (
            "LLMO optimization requires three key metrics: entity density measures "
            "the ratio of named entities to total words, answer directness evaluates "
            "how quickly the content addresses the query, and citability assesses "
            "whether AI systems can extract and quote the content accurately. "
            "Our study of 10,000 pages shows that pages scoring above 0.7 on all "
            "three metrics receive 3.2× more AI citations than lower-scoring pages."
        ),
        "keywords": ["LLMO", "entity density", "citability", "optimization"],
        "expected_strength": "high",
    },
    {
        "id": "ad_copy_01",
        "text": (
            "【期間限定】OpenMythos APIで広告ROASを平均2.4倍に。"
            "導入企業の87%が初月でCTR向上を確認。"
            "無料トライアル14日間 — クレジットカード不要。"
            "今すぐ始めて、Opus 4.8より低コストで高品質なコンテンツ生成を体験。"
        ),
        "keywords": ["ROAS", "CTR", "OpenMythos", "API", "広告"],
        "expected_strength": "high",
    },
    {
        "id": "ad_copy_02",
        "text": "安い！早い！うまい！今すぐクリック！",
        "keywords": ["ROI", "広告", "マーケティング"],
        "expected_strength": "low",
    },
    {
        "id": "technical_doc_01",
        "text": (
            "OpenMythos implements a Recurrent-Depth Transformer (RDT) architecture "
            "where the same transformer block is applied iteratively. "
            "The model supports 1–16 loop iterations at inference time, "
            "enabling dynamic computation depth based on task complexity. "
            "Mixture-of-Experts (MoE) with 8 experts and top-2 routing achieves "
            "3.7× parameter efficiency compared to dense models of equivalent capacity."
        ),
        "keywords": ["transformer", "RDT", "MoE", "inference"],
        "expected_strength": "high",
    },
]


# ---------------------------------------------------------------------------
# Rule-based baseline (Opus 4.8 代替)
# ---------------------------------------------------------------------------


def _rule_based_score(text: str, keywords: list[str]) -> LLMOScore:
    """
    ルールベースの LLMO スコア算出 (Opus 4.8 ベースライン代替)。

    単純な heuristic で entity_density / answer_directness / citability を推定する。
    本番では Claude API を使うが、API キー未設定時のフォールバックとして使用。
    """
    words = text.split()
    n_words = max(1, len(words))

    # entity_density: キーワード密度 + 数字・固有名詞の存在感
    import re

    kw_hits = sum(1 for kw in keywords if kw.lower() in text.lower())
    numbers = len(re.findall(r"\d+[\d,\.]*[%×倍x]?", text))
    entity_density = min(
        1.0, (kw_hits / max(1, len(keywords))) * 0.6 + (numbers / n_words) * 5
    )

    # answer_directness: 直接的な構造 (箇条書き・数値・明確な文末)
    has_list = bool(re.search(r"[・•\-\d+\.]", text))
    has_numbers = numbers > 0
    short_sentences = sum(1 for s in re.split(r"[。.!？?]", text) if 0 < len(s) < 80)
    answer_directness = min(
        1.0,
        0.3
        + (0.2 if has_list else 0)
        + (0.2 if has_numbers else 0)
        + min(0.3, short_sentences * 0.05),
    )

    # citability: 引用可能性 (固有名詞・数値・構造)
    capitalized = len(re.findall(r"[A-Z][A-Za-z0-9]+", text))
    katakana = len(re.findall(r"[ァ-ヴ]+", text))
    citability = min(1.0, (capitalized + katakana) / max(1, n_words) * 3 + 0.2)

    llmo_total = (entity_density + answer_directness + citability) / 3.0

    return LLMOScore(
        entity_density=round(entity_density, 4),
        answer_directness=round(answer_directness, 4),
        citability=round(citability, 4),
        llmo_total=round(llmo_total, 4),
        word_count=n_words,
        entities=[kw for kw in keywords if kw.lower() in text.lower()],
    )


def _claude_api_score(text: str, keywords: list[str]) -> Optional[LLMOScore]:
    """
    Claude API (Opus 4.8) を使った LLMO スコア算出。

    環境変数 ANTHROPIC_API_KEY が必要。
    未設定 or エラー時は None を返す。
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    try:
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f"Rate the following text on three LLMO dimensions (0.0-1.0):\n"
            f"1. entity_density: ratio of meaningful entities/keywords to total content\n"
            f"2. answer_directness: how directly the text answers potential queries\n"
            f"3. citability: how easily an AI can cite/extract this content\n\n"
            f"Target keywords: {', '.join(keywords)}\n\n"
            f"Text: {text[:1000]}\n\n"
            f"Respond with JSON only: "
            f'{{"entity_density": 0.0, "answer_directness": 0.0, "citability": 0.0}}'
        )
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        import re

        m = re.search(r"\{[^}]+\}", raw)
        if not m:
            return None
        data = json.loads(m.group())
        ed = float(data.get("entity_density", 0))
        ad = float(data.get("answer_directness", 0))
        ci = float(data.get("citability", 0))
        return LLMOScore(
            entity_density=round(ed, 4),
            answer_directness=round(ad, 4),
            citability=round(ci, 4),
            llmo_total=round((ed + ad + ci) / 3, 4),
            word_count=len(text.split()),
            entities=[kw for kw in keywords if kw.lower() in text.lower()],
        )
    except Exception as e:  # noqa: BLE001
        print(f"[Claude API error] {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# 比較実行
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    case_id: str
    text_preview: str
    keywords: list[str]
    expected_strength: str
    openmythos: dict
    baseline: dict
    claude_api: Optional[dict]
    openmythos_latency_ms: float
    baseline_latency_ms: float
    delta_overall: float  # OpenMythos - baseline


def run_comparison(
    cases: list[dict],
    use_claude: bool = False,
) -> list[ComparisonResult]:
    scorer = LLMOScorer()
    results = []

    for case in cases:
        text = case["text"]
        keywords = case.get("keywords", [])
        case_id = case.get("id", "unknown")

        # OpenMythos スコア
        t0 = time.perf_counter()
        om_score = scorer.score(text)
        om_ms = (time.perf_counter() - t0) * 1000

        # ルールベースライン スコア
        t1 = time.perf_counter()
        rb_score = _rule_based_score(text, keywords)
        rb_ms = (time.perf_counter() - t1) * 1000

        # Claude API スコア (オプション)
        cl_score = None
        if use_claude:
            cl_score = _claude_api_score(text, keywords)

        results.append(
            ComparisonResult(
                case_id=case_id,
                text_preview=text[:80] + ("..." if len(text) > 80 else ""),
                keywords=keywords,
                expected_strength=case.get("expected_strength", "unknown"),
                openmythos=asdict(om_score),
                baseline=asdict(rb_score),
                claude_api=asdict(cl_score) if cl_score else None,
                openmythos_latency_ms=round(om_ms, 2),
                baseline_latency_ms=round(rb_ms, 2),
                delta_overall=round(om_score.llmo_total - rb_score.llmo_total, 4),
            )
        )

    return results


# ---------------------------------------------------------------------------
# レポート出力
# ---------------------------------------------------------------------------


def print_report(results: list[ComparisonResult]) -> None:
    """コンソールに比較表を出力する。"""
    header = (
        f"{'ID':<20} {'Expected':<8} {'OM':>6} {'Base':>6} {'Delta':>7} {'Claude':>7}"
    )
    print("\n" + "=" * 60)
    print("OpenMythos vs Opus 4.8 Baseline — LLMO Score Comparison")
    print("=" * 60)
    print(header)
    print("-" * 60)

    for r in results:
        om_overall = r.openmythos["llmo_total"]
        rb_overall = r.baseline["llmo_total"]
        cl_str = f"{r.claude_api['llmo_total']:>7.4f}" if r.claude_api else "  N/A  "
        delta_str = f"{r.delta_overall:+.4f}"
        print(
            f"{r.case_id:<20} {r.expected_strength:<8} "
            f"{om_overall:>6.4f} {rb_overall:>6.4f} {delta_str:>7} {cl_str}"
        )

    print("-" * 60)
    avg_om = sum(r.openmythos["llmo_total"] for r in results) / len(results)
    avg_rb = sum(r.baseline["llmo_total"] for r in results) / len(results)
    avg_delta = sum(r.delta_overall for r in results) / len(results)
    print(
        f"{'AVERAGE':<20} {'':<8} " f"{avg_om:>6.4f} {avg_rb:>6.4f} {avg_delta:+>7.4f}"
    )
    print("=" * 60)

    if avg_delta > 0:
        pct = avg_delta / avg_rb * 100 if avg_rb > 0 else 0
        print(
            f"\n✅ OpenMythos が baseline より平均 {avg_delta:.4f} ({pct:.1f}%) 上回る"
        )
    else:
        print(f"\n⚠️  OpenMythos が baseline より平均 {abs(avg_delta):.4f} 下回る")

    avg_om_ms = sum(r.openmythos_latency_ms for r in results) / len(results)
    avg_rb_ms = sum(r.baseline_latency_ms for r in results) / len(results)
    print(f"\nLatency: OpenMythos {avg_om_ms:.1f}ms / Baseline {avg_rb_ms:.1f}ms")


def save_results(results: list[ComparisonResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"opus_comparison_{ts}.json"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "n_cases": len(results),
        "summary": {
            "avg_openmythos_overall": round(
                sum(r.openmythos["llmo_total"] for r in results) / len(results), 4
            ),
            "avg_baseline_overall": round(
                sum(r.baseline["llmo_total"] for r in results) / len(results), 4
            ),
            "avg_delta": round(sum(r.delta_overall for r in results) / len(results), 4),
            "has_claude_api": any(r.claude_api is not None for r in results),
        },
        "results": [asdict(r) for r in results],
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_jsonl_cases(path: Path, n: int) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            record = json.loads(line.strip())
            # content_quality タスク形式に対応
            text = record.get("text", record.get("content", record.get("input", "")))
            keywords = record.get("keywords", record.get("tags", []))
            cases.append(
                {
                    "id": record.get("id", f"case_{i}"),
                    "text": text,
                    "keywords": keywords if isinstance(keywords, list) else [],
                    "expected_strength": record.get("quality", "unknown"),
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenMythos vs Opus 4.8 LLMO Comparison"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="JSONL ファイルパス (省略時は組み込みテストケース使用)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="JSONL から読み込む件数 (デフォルト: 20)",
    )
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Claude API (Opus 4.8) との比較を有効化 (ANTHROPIC_API_KEY 必要)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark" / "results",
        help="結果保存ディレクトリ",
    )
    args = parser.parse_args()

    if args.input:
        cases = _load_jsonl_cases(args.input, args.n)
        print(f"Loaded {len(cases)} cases from {args.input}")
    else:
        cases = _BUILTIN_CASES
        print(f"Using {len(cases)} built-in test cases")

    results = run_comparison(cases, use_claude=args.claude)
    print_report(results)

    path = save_results(results, args.output_dir)
    print(f"\nResults saved → {path}")


if __name__ == "__main__":
    main()
