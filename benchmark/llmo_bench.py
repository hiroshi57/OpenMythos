"""
OpenMythos LLMO Benchmark — Opus 4.8 対抗ベンチマーク

OpenMythos の LLMOScorer を「ルールベースベースライン」および
オプションで「Claude API」と比較する。

実行方法:
    python benchmark/llmo_bench.py
    ANTHROPIC_API_KEY=sk-... python benchmark/llmo_bench.py  # Claude API 比較を有効化

出力:
    コンソール比較表 + benchmark/results/llmo_bench_YYYYMMDD.json
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# リポジトリルートを path に追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_mythos.llmo import LLMOScorer

# ---------------------------------------------------------------------------
# サンプルデータセット（日本語5件 + 英語5件）
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENTS = [
    # --- 日本語 SEO コンテンツ ---
    {
        "id": "ja_01",
        "lang": "ja",
        "keyword": "デジタルマーケティング",
        "title": "デジタルマーケティングとは？基礎から実践まで",
        "h1": "デジタルマーケティングの基礎知識",
        "body": (
            "デジタルマーケティングとは、Google・Meta・LINE などのデジタルプラットフォームを活用して"
            "顧客を獲得・育成する手法です。2024年の調査によると、日本企業の SEO 投資は前年比 32% 増加し、"
            "平均 ROAS は 3.8x を記録しています。\n\n"
            "主要チャネル:\n"
            "- SEO: 長期的オーガニック流入。CPL は広告の 1/5\n"
            "- リスティング広告: 即効性高。CTR 3.5%、CPC 150〜500円\n"
            "- SNS 広告: エンゲージメント率 2〜8%\n\n"
            "出典: デジタルマーケティング白書 2024（株式会社 DI）"
        ),
    },
    {
        "id": "ja_02",
        "lang": "ja",
        "keyword": "LLMO",
        "title": "LLMOとは？AIサーチ時代のコンテンツ最適化",
        "h1": "LLMO（Large Language Model Optimization）完全ガイド",
        "body": (
            "はい、LLMO は 2024年以降の SEO で最重要指標になっています。"
            "LLMO（Large Language Model Optimization）とは、ChatGPT・Perplexity・Gemini などの"
            "AI 検索エンジンがコンテンツを引用・推薦する可能性を高める最適化手法です。\n\n"
            "OpenMythos の LLMOScorer は entity_density・answer_directness・citability の"
            "3スコアで評価します。スコアが 0.7 以上のコンテンツは AI 引用率が 40% 向上すると"
            "2025年の研究で報告されています。"
        ),
    },
    {
        "id": "ja_03",
        "lang": "ja",
        "keyword": "SEO対策",
        "title": "SEO対策の方法",
        "h1": "SEO対策について",
        "body": (
            "SEO対策はとても大切です。検索エンジンで上位に表示されることが重要です。"
            "良いコンテンツを作ることが必要です。定期的に更新することも大切です。"
            "ユーザーに役立つ情報を提供しましょう。"
        ),
    },
    {
        "id": "ja_04",
        "lang": "ja",
        "keyword": "広告ROI",
        "title": "広告ROIの計算方法と改善戦略",
        "h1": "広告 ROI・ROAS の正しい計算と最適化",
        "body": (
            "広告 ROI（Return on Investment）は「(売上 - 広告費) / 広告費 × 100」で算出します。"
            "ROAS（Return on Ad Spend）= 売上 / 広告費。業界平均は 3.0〜5.0x です。\n\n"
            "改善手順:\n"
            "1. Quality Score (QS) を 7 以上に維持（CPC が最大 50% 削減）\n"
            "2. ランディングページの離脱率を 60% 以下に抑える\n"
            "3. CVR 目標: 業種別平均 2〜5%\n\n"
            "Google Ads データ（2024年 Q3）によると、QS 10 の広告は QS 4 比で"
            "CPC が 67% 安く CTR は 2.3 倍高い。"
        ),
    },
    {
        "id": "ja_05",
        "lang": "ja",
        "keyword": "コンテンツマーケティング",
        "title": "コンテンツマーケティング戦略の立て方",
        "h1": "コンテンツマーケティングの基本と実践",
        "body": (
            "コンテンツマーケティングは、価値あるコンテンツを通じて顧客を獲得する手法です。"
            "E-E-A-T（Experience・Expertise・Authoritativeness・Trustworthiness）を意識した"
            "コンテンツが Google の評価を高めます。HubSpot の調査では、月 16 本以上の記事を"
            "公開する企業はリード獲得数が 4.5 倍になると報告されています（2024年版）。"
        ),
    },
    # --- 英語 SEO コンテンツ ---
    {
        "id": "en_01",
        "lang": "en",
        "keyword": "digital marketing",
        "title": "What Is Digital Marketing? Complete 2024 Guide",
        "h1": "Digital Marketing: Definition, Types, and Strategy",
        "body": (
            "Digital marketing is the promotion of products or services through digital channels "
            "such as Google Search, Meta Ads, and email. In 2024, global digital ad spend reached "
            "$740 billion, with SEO driving 53% of website traffic on average.\n\n"
            "Core channels:\n"
            "- SEO: Long-term organic growth. Average CPL is 5x cheaper than paid ads.\n"
            "- PPC: Immediate traffic. Average CTR 3.5%, CPC $1.20–$4.50\n"
            "- Social media: Engagement rate 2–8% depending on platform\n\n"
            "Source: Digital Marketing Benchmark Report 2024 (HubSpot)"
        ),
    },
    {
        "id": "en_02",
        "lang": "en",
        "keyword": "SEO optimization",
        "title": "SEO Optimization Tips",
        "h1": "How to Optimize Your Website",
        "body": (
            "SEO is important. You should use keywords. Make sure your website is fast. "
            "Write good content. Update regularly. Users will find your site."
        ),
    },
    {
        "id": "en_03",
        "lang": "en",
        "keyword": "LLMO",
        "title": "LLMO: The Future of AI-Search Content Optimization",
        "h1": "LLMO (Large Language Model Optimization) Guide",
        "body": (
            "Yes, LLMO is the next evolution of SEO — optimizing content so AI search engines "
            "like Perplexity, ChatGPT Search, and Google AI Overviews cite your content.\n\n"
            "OpenMythos LLMOScorer measures three dimensions: entity_density (0–1), "
            "answer_directness (0–1), and citability (0–1). Content scoring above 0.70 "
            "shows a 40% higher AI citation rate according to a 2025 study. "
            "Structured data, statistics, and direct answers are key signals."
        ),
    },
    {
        "id": "en_04",
        "lang": "en",
        "keyword": "ad ROI",
        "title": "How to Calculate and Maximize Ad ROI in 2024",
        "h1": "Ad ROI and ROAS: Calculation, Benchmarks, and Optimization",
        "body": (
            "Ad ROI = (Revenue - Ad Spend) / Ad Spend × 100. "
            "ROAS = Revenue / Ad Spend. Industry benchmark: 3.0–5.0x ROAS.\n\n"
            "Optimization checklist:\n"
            "1. Achieve Quality Score ≥ 7 (reduces CPC by up to 50%)\n"
            "2. Landing page bounce rate < 60%\n"
            "3. Target CVR: 2–5% depending on vertical\n\n"
            "Google Ads data (Q3 2024): QS 10 ads are 67% cheaper CPC and 2.3x higher CTR "
            "versus QS 4."
        ),
    },
    {
        "id": "en_05",
        "lang": "en",
        "keyword": "content marketing",
        "title": "Content Marketing Strategy: Build Authority and Drive Traffic",
        "h1": "Content Marketing: How to Create High-Impact Content",
        "body": (
            "Content marketing is a strategy for attracting customers through valuable content. "
            "E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness) signals are "
            "critical for Google rankings in 2024. According to HubSpot Research, companies "
            "publishing 16+ articles per month generate 4.5x more leads (2024 State of Marketing)."
        ),
    },
]


# ---------------------------------------------------------------------------
# ベースライン: 単純キーワード密度のみ
# ---------------------------------------------------------------------------


def _baseline_score(doc: dict) -> float:
    """シンプルなキーワード密度スコア（比較ベースライン）。"""
    body = doc["body"].lower()
    kw = doc["keyword"].lower()
    words = body.split()
    if not words:
        return 0.0
    matches = len(re.findall(re.escape(kw), body))
    density = matches / len(words)
    # 0.01〜0.03 が理想。外れるほど減点
    if 0.005 <= density <= 0.04:
        return min(1.0, density * 50.0)
    return max(0.0, 0.5 - abs(density - 0.02) * 10)


# ---------------------------------------------------------------------------
# Claude API 比較（オプション）
# ---------------------------------------------------------------------------


def _claude_score(doc: dict, api_key: str) -> Optional[float]:
    """Claude API で同じテキストの LLMO 品質スコアを取得する（オプション）。"""
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f"Rate this SEO content for AI-search citation potential (LLMO score).\n"
            f"Keyword: {doc['keyword']}\n"
            f"Title: {doc['title']}\n"
            f"Content: {doc['body'][:500]}\n\n"
            f"Reply with ONLY a decimal number between 0.0 and 1.0 representing quality. "
            f"Higher = more likely to be cited by AI search engines."
        )
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        score = float(re.search(r"\d+\.?\d*", raw).group())
        return min(1.0, max(0.0, score))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# メインベンチマーク
# ---------------------------------------------------------------------------


def run_benchmark(
    docs: Optional[list[dict]] = None,
    use_claude_api: bool = True,
) -> dict:
    """
    LLMO ベンチマークを実行して結果を返す。

    Args:
        docs           -- ドキュメントリスト（None でデフォルトサンプルを使用）
        use_claude_api -- True かつ ANTHROPIC_API_KEY が設定済みなら Claude API と比較

    Returns:
        ベンチマーク結果 dict
    """
    if docs is None:
        docs = SAMPLE_DOCUMENTS

    scorer = LLMOScorer()
    api_key = os.environ.get("ANTHROPIC_API_KEY") if use_claude_api else None

    results = []
    for doc in docs:
        om_result = scorer.score_with_keywords(
            doc["body"],
            title=doc.get("title", ""),
            h1=doc.get("h1", ""),
            target_keyword=doc["keyword"],
        )
        baseline = _baseline_score(doc)
        claude = _claude_score(doc, api_key) if api_key else None

        results.append(
            {
                "id": doc["id"],
                "lang": doc["lang"],
                "keyword": doc["keyword"],
                "openmythos_llmo": om_result.llmo_total,
                "openmythos_entity": om_result.entity_density,
                "openmythos_directness": om_result.answer_directness,
                "openmythos_citability": om_result.citability,
                "openmythos_wkd": om_result.weighted_keyword_density,
                "baseline_keyword_density": round(baseline, 4),
                "claude_api_score": claude,
            }
        )

    # 集計
    om_avg = sum(r["openmythos_llmo"] for r in results) / len(results)
    bl_avg = sum(r["baseline_keyword_density"] for r in results) / len(results)
    claude_scores = [
        r["claude_api_score"] for r in results if r["claude_api_score"] is not None
    ]
    claude_avg = sum(claude_scores) / len(claude_scores) if claude_scores else None

    return {
        "timestamp": datetime.now().isoformat(),
        "n_documents": len(results),
        "results": results,
        "summary": {
            "openmythos_avg": round(om_avg, 4),
            "baseline_avg": round(bl_avg, 4),
            "claude_api_avg": round(claude_avg, 4) if claude_avg is not None else None,
            "openmythos_vs_baseline_delta": round(om_avg - bl_avg, 4),
        },
    }


def _print_table(benchmark: dict) -> None:
    """ベンチマーク結果をコンソールに表示する。"""
    print("\n" + "=" * 72)
    print("  OpenMythos LLMO Benchmark — Opus 4.8 対抗比較")
    print("=" * 72)
    header = (
        f"{'ID':<8} {'Lang':<5} {'OpenMythos':>11} {'Baseline':>10} {'Claude API':>11}"
    )
    print(header)
    print("-" * 72)
    for r in benchmark["results"]:
        claude = (
            f"{r['claude_api_score']:.3f}"
            if r["claude_api_score"] is not None
            else "   N/A   "
        )
        print(
            f"{r['id']:<8} {r['lang']:<5} "
            f"{r['openmythos_llmo']:>11.3f} "
            f"{r['baseline_keyword_density']:>10.3f} "
            f"{claude:>11}"
        )
    print("-" * 72)
    s = benchmark["summary"]
    claude_str = (
        f"{s['claude_api_avg']:.3f}" if s["claude_api_avg"] is not None else "   N/A   "
    )
    print(
        f"{'AVERAGE':<14} {s['openmythos_avg']:>11.3f} {s['baseline_avg']:>10.3f} {claude_str:>11}"
    )
    print(f"\n  OpenMythos vs Baseline: {s['openmythos_vs_baseline_delta']:+.4f}")
    if s["claude_api_avg"] is not None:
        delta_vs_claude = s["openmythos_avg"] - s["claude_api_avg"]
        print(f"  OpenMythos vs Claude API: {delta_vs_claude:+.4f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    print("Running LLMO benchmark...")
    t0 = time.perf_counter()
    result = run_benchmark()
    elapsed = time.perf_counter() - t0

    _print_table(result)
    print(f"Completed in {elapsed:.2f}s")

    # 結果を JSON 保存
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = results_dir / f"llmo_bench_{date_str}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {out_path}")
