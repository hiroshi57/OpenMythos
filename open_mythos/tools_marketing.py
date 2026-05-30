"""
OpenMythos マーケティング特化ツール集。

search_competitor  -- 競合他社の広告費・CTR・SEOスコアを検索
calculate_roi      -- ROI / ROAS / CTR を計算
fetch_trend        -- キーワードトレンド & LLMO 人気度を取得
score_content      -- コンテンツの SEO/LLMO スコアを算出

これらはすべてスタブ実装 (ハードコード or 計算ベース)。
本番では外部 API (SimilarWeb・SEMrush・Google Trends 等) に差し替える。
"""

from __future__ import annotations

import math
import re
import random
from typing import Optional

from open_mythos.tools import ToolRegistry, ParameterSchema, ToolDefinition


# ---------------------------------------------------------------------------
# Tool 実装
# ---------------------------------------------------------------------------


def search_competitor(
    company: str,
    metric: str = "ad_spend",
    period: str = "last_30_days",
) -> dict:
    """
    競合他社のマーケティング指標を検索する。

    Args:
        company -- 競合企業名 (例: "Jasper AI", "Copy.ai")
        metric  -- 取得する指標 ("ad_spend" / "ctr" / "seo_score" / "all")
        period  -- 期間 ("last_7_days" / "last_30_days" / "last_90_days")

    Returns:
        競合指標の辞書
    """
    # スタブ: 企業名のハッシュから決定論的な値を生成
    seed = sum(ord(c) for c in company)
    rng = random.Random(seed)

    base_ad_spend = rng.randint(500_000, 5_000_000)
    base_ctr = round(rng.uniform(0.015, 0.065), 4)
    base_seo = rng.randint(42, 92)

    period_multiplier = {"last_7_days": 0.23, "last_30_days": 1.0, "last_90_days": 3.1}
    mult = period_multiplier.get(period, 1.0)

    result: dict = {
        "company": company,
        "period": period,
        "data_source": "OpenMythos Competitive Intelligence (stub)",
    }

    if metric in ("ad_spend", "all"):
        result["ad_spend_usd"] = int(base_ad_spend * mult)
    if metric in ("ctr", "all"):
        result["avg_ctr"] = base_ctr
    if metric in ("seo_score", "all"):
        result["seo_score"] = base_seo
        result["organic_traffic_est"] = int(rng.randint(10_000, 500_000) * mult)
    if metric == "all":
        result["market_share_pct"] = round(rng.uniform(1.5, 18.0), 2)

    return result


def calculate_roi(
    ad_spend: float,
    revenue: float,
    cogs: float = 0.0,
    clicks: Optional[int] = None,
    impressions: Optional[int] = None,
) -> dict:
    """
    ROI / ROAS / CTR を計算する。

    Args:
        ad_spend    -- 広告費 (USD)
        revenue     -- 売上 (USD)
        cogs        -- 売上原価 (USD, デフォルト 0)
        clicks      -- クリック数 (オプション)
        impressions -- インプレッション数 (オプション)

    Returns:
        {roi_pct, roas, gross_profit, cpc_usd, ctr, ...}
    """
    if ad_spend <= 0:
        return {"error": "ad_spend must be > 0"}

    gross_profit = revenue - cogs
    roi_pct = (gross_profit - ad_spend) / ad_spend * 100.0
    roas = revenue / ad_spend

    result: dict = {
        "ad_spend_usd": round(ad_spend, 2),
        "revenue_usd": round(revenue, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "roi_pct": round(roi_pct, 2),
        "roas": round(roas, 3),
        "profitable": roi_pct > 0,
    }

    if clicks is not None and clicks > 0:
        result["cpc_usd"] = round(ad_spend / clicks, 4)  # Cost Per Click
        result["revenue_per_click"] = round(revenue / clicks, 4)

    if clicks is not None and impressions is not None and impressions > 0:
        result["ctr"] = round(clicks / impressions, 6)
        result["cpm_usd"] = round(ad_spend / impressions * 1000, 4)

    return result


def fetch_trend(
    keyword: str,
    region: str = "JP",
    category: str = "marketing",
) -> dict:
    """
    キーワードのトレンド情報と LLMO 人気度を取得する。

    Args:
        keyword  -- 検索キーワード (例: "LLMO", "SEO最適化", "AI広告")
        region   -- 地域コード ("JP" / "US" / "global")
        category -- カテゴリ ("marketing" / "tech" / "general")

    Returns:
        {trend_score, llmo_popularity, search_volume_est, rising, related_keywords}
    """
    seed = sum(ord(c) for c in keyword + region)
    rng = random.Random(seed)

    trend_score = rng.randint(20, 95)
    llmo_popularity = round(rng.uniform(0.1, 0.9), 3)
    search_vol = rng.randint(1_000, 500_000)

    related = [
        f"{keyword} 使い方",
        f"{keyword} 比較",
        f"AI {keyword}",
        f"{keyword} 2025",
        f"{keyword} ツール",
    ]

    return {
        "keyword": keyword,
        "region": region,
        "category": category,
        "trend_score": trend_score,           # 0–100 (Google Trends 類似)
        "llmo_popularity": llmo_popularity,   # AIサーチでの出現頻度推定 0–1
        "search_volume_est": search_vol,      # 月間検索数推定
        "is_rising": trend_score > 60,
        "yoy_change_pct": round(rng.uniform(-30, 150), 1),
        "related_keywords": related[:3],
        "data_source": "OpenMythos Trend API (stub)",
    }


def score_content(
    text: str,
    target_keyword: str = "",
    style: str = "auto",
) -> dict:
    """
    コンテンツの SEO/LLMO スコアを算出する。

    Args:
        text            -- スコアリング対象テキスト
        target_keyword  -- ターゲットキーワード (オプション)
        style           -- コンテンツスタイル ("answer_first" / "faq" / "entity_rich" / "auto")

    Returns:
        {llmo_total, entity_density, answer_directness, citability, keyword_density, recommendations}
    """
    from open_mythos.llmo import LLMOScorer

    scorer = LLMOScorer()
    llmo = scorer.score(text)

    # キーワード密度
    keyword_density = 0.0
    if target_keyword and text:
        kw_lower = re.escape(target_keyword.lower())
        words = text.lower().split()
        matches = re.findall(r'\b' + kw_lower + r'\b', text.lower())
        keyword_density = len(matches) / len(words) if words else 0.0

    # 改善推奨を生成
    recommendations: list[str] = []
    if llmo.entity_density < 0.4:
        recommendations.append("エンティティ密度を上げる: 数値・固有名詞・専門語を追加してください")
    if llmo.answer_directness < 0.4:
        recommendations.append("冒頭1文で直接答える answer-first 形式に変更してください")
    if llmo.citability < 0.4:
        recommendations.append("統計データや出典を追加して引用されやすさを向上させてください")
    if keyword_density > 0.02:
        recommendations.append(f"キーワード'{target_keyword}'の出現が多すぎます (密度: {keyword_density:.1%}, SEO推奨: 1-2%)")
    if not recommendations:
        recommendations.append("コンテンツ品質は良好です")

    return {
        "llmo_total": llmo.llmo_total,
        "entity_density": llmo.entity_density,
        "answer_directness": llmo.answer_directness,
        "citability": llmo.citability,
        "word_count": llmo.word_count,
        "keyword_density": round(keyword_density, 4),
        "entities_detected": llmo.entities[:10],
        "recommendations": recommendations,
        "style_detected": style if style != "auto" else _detect_style(text),
    }


def _detect_style(text: str) -> str:
    """テキストのスタイルを推定する。"""
    if text.strip().lower().startswith(("yes", "no", "はい", "いいえ")):
        return "answer_first"
    if "Q:" in text or "A:" in text or "？" in text[:50]:
        return "faq"
    import re
    if len(re.findall(r'\d+', text)) > 3:
        return "entity_rich"
    return "general"


# ---------------------------------------------------------------------------
# レジストリへの一括登録
# ---------------------------------------------------------------------------

_MARKETING_TOOLS = [
    ToolDefinition(
        name="search_competitor",
        description="競合他社の広告費・CTR・SEOスコア・市場シェアを検索する。マーケティング戦略立案に使用。",
        parameters={
            "company": ParameterSchema(type="string", description="競合企業名 (例: 'Jasper AI')", required=True),
            "metric": ParameterSchema(
                type="string",
                description="取得する指標",
                enum=["ad_spend", "ctr", "seo_score", "all"],
                required=False,
                default="all",
            ),
            "period": ParameterSchema(
                type="string",
                description="集計期間",
                enum=["last_7_days", "last_30_days", "last_90_days"],
                required=False,
                default="last_30_days",
            ),
        },
        fn=search_competitor,
    ),
    ToolDefinition(
        name="calculate_roi",
        description="広告ROI・ROAS・CPA・CTRを計算する。予算計画とキャンペーン評価に使用。",
        parameters={
            "ad_spend": ParameterSchema(type="number", description="広告費 (USD)", required=True),
            "revenue": ParameterSchema(type="number", description="売上 (USD)", required=True),
            "cogs": ParameterSchema(type="number", description="売上原価 (USD)", required=False, default=0.0),
            "clicks": ParameterSchema(type="integer", description="クリック数 (オプション)", required=False, default=None),
            "impressions": ParameterSchema(type="integer", description="インプレッション数 (オプション)", required=False, default=None),
        },
        fn=calculate_roi,
    ),
    ToolDefinition(
        name="fetch_trend",
        description="キーワードのトレンドスコアとLLMO人気度を取得する。コンテンツ戦略立案に使用。",
        parameters={
            "keyword": ParameterSchema(type="string", description="検索キーワード", required=True),
            "region": ParameterSchema(
                type="string",
                description="地域コード",
                enum=["JP", "US", "global"],
                required=False,
                default="JP",
            ),
            "category": ParameterSchema(
                type="string",
                description="カテゴリ",
                enum=["marketing", "tech", "general"],
                required=False,
                default="marketing",
            ),
        },
        fn=fetch_trend,
    ),
    ToolDefinition(
        name="score_content",
        description="テキストコンテンツのSEO・LLMOスコアを算出し改善推奨を返す。コンテンツ品質改善に使用。",
        parameters={
            "text": ParameterSchema(type="string", description="スコアリング対象テキスト", required=True),
            "target_keyword": ParameterSchema(type="string", description="ターゲットキーワード", required=False, default=""),
            "style": ParameterSchema(
                type="string",
                description="コンテンツスタイル",
                enum=["answer_first", "faq", "entity_rich", "auto"],
                required=False,
                default="auto",
            ),
        },
        fn=score_content,
    ),
]


def register_marketing_tools(registry: ToolRegistry) -> None:
    """マーケ特化4ツールをレジストリに登録する。"""
    for tool_def in _MARKETING_TOOLS:
        registry.register(tool_def)
