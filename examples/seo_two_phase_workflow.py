"""
2-Phase SEO Workflow — Claude Opus 4.8 の「コンテキストドリフト」を構造的に回避する設計

Claude Opus 4.8 の既知の弱点:
  「SEO戦略フェーズ（キーワード調査・構成決定）」と「執筆フェーズ」を同一会話に
  混在させると汎用出力に劣化する（コンテキストドリフト）。

OpenMythos の解法:
  Phase 1 と Phase 2 を完全に分離した ConversationMemory インスタンスを使い、
  drift_score を監視してドリフトリスクを定量化する。

実行:
    python examples/seo_two_phase_workflow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from open_mythos.llmo import LLMOScorer
from open_mythos.conversation import ConversationMemory
from open_mythos.tools_marketing import score_content, fetch_trend

# ドリフトリスクの警告しきい値
DRIFT_WARNING_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Phase 1: 戦略フェーズ（キーワード調査 → 構成生成 → LLMOスコア評価）
# ---------------------------------------------------------------------------


def phase1_strategy(keyword: str) -> dict:
    """
    戦略フェーズ: キーワードトレンド調査 + コンテンツ構成をルールベースで生成。
    専用の ConversationMemory を使い、執筆フェーズと文脈を完全分離する。
    """
    memory = ConversationMemory(
        max_turns=10, max_chars=2000, system_msg="あなたは SEO 戦略の専門家です。"
    )

    print("\n" + "=" * 60)
    print(f"  Phase 1: 戦略フェーズ — キーワード: 「{keyword}」")
    print("=" * 60)

    # Step 1-1: トレンド調査
    trend = fetch_trend(keyword, region="JP")
    memory.add_user(f"キーワード「{keyword}」のトレンドを分析してください。")
    trend_summary = (
        f"トレンドスコア: {trend['trend_score']}/100 "
        f"({'上昇中' if trend['is_rising'] else '横ばい'}), "
        f"月間検索数: {trend['search_volume_est']:,}, "
        f"前年比: {trend['yoy_change_pct']:+.1f}%, "
        f"LLMO人気度: {trend['llmo_popularity']:.3f}"
    )
    memory.add_assistant(trend_summary)
    print(f"\n[トレンド分析]\n  {trend_summary}")

    # ドリフト監視
    ds = memory.drift_score
    print(
        f"  drift_score: {ds:.3f}",
        "⚠️ 高リスク" if ds >= DRIFT_WARNING_THRESHOLD else "✅",
    )

    # Step 1-2: コンテンツ構成案を生成
    memory.add_user(
        f"「{keyword}」をターゲットにした SEO 記事の構成を提案してください。"
    )
    structure = (
        f"## 推奨構成: 「{keyword}完全ガイド」\n"
        f"1. {keyword}とは？（定義・概要）— H2、answer-first 形式\n"
        f"2. {keyword}の最新トレンド {trend['yoy_change_pct']:+.0f}% — H2、統計データ含む\n"
        f"3. 具体的な実践手順 — H2、リスト形式\n"
        f"4. ツール・サービス比較表 — H2、entity-rich\n"
        f"5. FAQ — H2、LLMO 最適化\n"
        f"推奨文字数: 2,000〜3,000字"
    )
    memory.add_assistant(structure)
    print(f"\n[コンテンツ構成案]\n{structure}")

    ds = memory.drift_score
    print(
        f"\n  drift_score: {ds:.3f}",
        "⚠️ 高リスク" if ds >= DRIFT_WARNING_THRESHOLD else "✅",
    )

    # Step 1-3: ターゲットキーワードの LLMO評価基準を設定
    memory.add_user(
        "このキーワードで高 LLMO スコアを出すための重点ポイントを教えてください。"
    )
    llmo_tips = (
        f"高 LLMO スコアのための重点ポイント:\n"
        f"• answer_directness: H1直後に「{keyword}とは〜です」形式で直接定義\n"
        f"• entity_density: 数値・固有名詞・統計データを100語に15個以上含める\n"
        f"• citability: 出典付き統計、年号（{trend['yoy_change_pct']:+.0f}%、2024年）、構造マーカー\n"
        f"• keyword密度: title×3, h1×2, body×1 の重み付けで 1〜2% が最適"
    )
    memory.add_assistant(llmo_tips)
    print(f"\n[LLMO 最適化ポイント]\n{llmo_tips}")

    final_drift = memory.drift_score
    print(
        f"\n  最終 drift_score: {final_drift:.3f} "
        + (
            "⚠️ Phase 2 を別インスタンスで開始してください"
            if final_drift >= DRIFT_WARNING_THRESHOLD
            else "✅"
        )
    )

    return {
        "keyword": keyword,
        "trend": trend,
        "structure": structure,
        "llmo_tips": llmo_tips,
        "phase1_drift_score": final_drift,
        "phase1_memory_stats": memory.stats(),
    }


# ---------------------------------------------------------------------------
# Phase 2: 執筆フェーズ（戦略フェーズと完全分離した新インスタンスで実行）
# ---------------------------------------------------------------------------


def phase2_write(strategy: dict) -> dict:
    """
    執筆フェーズ: Phase 1 の戦略情報のみを初期コンテキストとして渡し、
    新しい ConversationMemory で執筆に集中する。コンテキストドリフト回避。
    """
    keyword = strategy["keyword"]

    # Phase 2 専用の新しいメモリインスタンス（Phase 1 の会話履歴を引き継がない）
    memory = ConversationMemory(
        max_turns=8,
        max_chars=3000,
        system_msg=(
            f"あなたは SEO ライターです。以下の戦略に従い「{keyword}」の記事本文を書いてください。\n"
            f"構成: {strategy['structure'][:200]}\n"
            f"重点: {strategy['llmo_tips'][:200]}"
        ),
    )

    print("\n" + "=" * 60)
    print("  Phase 2: 執筆フェーズ（Phase 1 と完全分離）")
    print("=" * 60)

    # サンプルコンテンツを生成（ルールベース、実際はモデルが生成）
    sample_content = (
        f"{keyword}とは、デジタルプラットフォームを活用して顧客を獲得・育成する手法です。"
        f"2024年の調査では、{keyword}への投資は前年比 {strategy['trend']['yoy_change_pct']:+.0f}% 増加し、"
        f"月間検索数は {strategy['trend']['search_volume_est']:,} 件に達しています。\n\n"
        f"## {keyword}の最新トレンド\n\n"
        f"- トレンドスコア: {strategy['trend']['trend_score']}/100\n"
        f"- LLMO 人気度: {strategy['trend']['llmo_popularity']:.3f}\n"
        f"- 関連キーワード: {', '.join(strategy['trend']['related_keywords'])}\n\n"
        f"出典: OpenMythos Trend API（2024年データ）"
    )

    memory.add_user(f"「{keyword}」の記事の導入部と第2章を書いてください。")
    memory.add_assistant(sample_content)

    # LLMO スコア評価
    scorer = LLMOScorer()
    llmo_result = scorer.score_with_keywords(
        sample_content,
        title=f"{keyword}完全ガイド",
        h1=f"{keyword}とは？",
        target_keyword=keyword,
    )

    seo_report = score_content(sample_content, target_keyword=keyword)

    print(f"\n[生成コンテンツ（抜粋）]\n{sample_content[:300]}...\n")
    print("[LLMO スコア評価]")
    print(f"  llmo_total       : {llmo_result.llmo_total:.3f}")
    print(f"  entity_density   : {llmo_result.entity_density:.3f}")
    print(f"  answer_directness: {llmo_result.answer_directness:.3f}")
    print(f"  citability       : {llmo_result.citability:.3f}")
    print(f"  weighted_kw_density: {llmo_result.weighted_keyword_density:.4f}")
    print(f"  word_count       : {llmo_result.word_count}")
    if llmo_result.ja_tokens:
        print(f"  ja_tokens (先頭10件): {llmo_result.ja_tokens[:10]}")

    final_drift = memory.drift_score
    print(
        f"\n  Phase 2 drift_score: {final_drift:.3f} "
        + ("⚠️ リセット推奨" if final_drift >= DRIFT_WARNING_THRESHOLD else "✅")
    )

    print("\n[改善推奨]")
    for rec in seo_report["recommendations"]:
        print(f"  • {rec}")

    return {
        "content": sample_content,
        "llmo_score": llmo_result,
        "phase2_drift_score": final_drift,
        "recommendations": seo_report["recommendations"],
    }


# ---------------------------------------------------------------------------
# A/B テスト: 複数バリアントを比較
# ---------------------------------------------------------------------------


def phase3_ab_test(keyword: str) -> None:
    """複数のコンテンツバリアントを ab_test() で比較する。"""
    scorer = LLMOScorer()

    variants = [
        # バリアント A: entity-rich
        (
            f"{keyword}の市場規模は2024年に4,500億円（前年比+32%）に達しました。"
            f"主要プレイヤーは Google・Meta・LINE で、CTR 平均3.5%、ROAS 3.8x が業界標準です。"
            f"出典: デジタルマーケティング白書2024"
        ),
        # バリアント B: answer-first
        (
            f"はい、{keyword}は2024年も最重要の集客手段です。"
            f"理由は3つ: AI検索への対応、E-E-A-T強化、LLMO最適化です。"
            f"特に日本語コンテンツは競合が少なくROIが高い傾向にあります。"
        ),
        # バリアント C: 一般的（低品質）
        (
            f"{keyword}はとても重要です。しっかり取り組むことが大切です。"
            f"良いコンテンツを作りましょう。"
        ),
    ]

    print("\n" + "=" * 60)
    print(f"  Phase 3: A/B テスト — {keyword}")
    print("=" * 60)

    ab_result = scorer.ab_test(variants, threshold=0.05)
    labels = ["A: entity-rich", "B: answer-first", "C: 低品質"]

    for i, (label, score, delta) in enumerate(
        zip(labels, ab_result.scores, ab_result.deltas)
    ):
        winner_mark = " 👑 WINNER" if i == ab_result.winner_index else ""
        print(f"  {label}: {score:.3f}  (Δ{delta:+.3f}){winner_mark}")

    print(
        f"\n  有意差あり: {'✅ Yes' if ab_result.significant else '❌ No'} "
        f"(閾値: {ab_result.threshold})"
    )


# ---------------------------------------------------------------------------
# メイン実行
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    keyword = "デジタルマーケティング"

    print("\n🚀 OpenMythos 2-Phase SEO Workflow")
    print("   Claude Opus 4.8 のコンテキストドリフトを構造的に回避する設計\n")

    # Phase 1: 戦略（専用メモリ）
    strategy = phase1_strategy(keyword)

    # Phase 2: 執筆（Phase 1 と完全分離した新メモリ）
    result = phase2_write(strategy)

    # Phase 3: A/B テスト
    phase3_ab_test(keyword)

    print("\n" + "=" * 60)
    print("  完了サマリー")
    print("=" * 60)
    print(f"  Phase 1 drift: {strategy['phase1_drift_score']:.3f}")
    print(f"  Phase 2 drift: {result['phase2_drift_score']:.3f}")
    print(f"  最終 LLMO スコア: {result['llmo_score'].llmo_total:.3f}")
    print("\n  ✅ 戦略・執筆の分離により、コンテキストドリフトを回避しました")
    print("=" * 60)
