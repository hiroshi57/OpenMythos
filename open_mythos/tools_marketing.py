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
        "trend_score": trend_score,  # 0–100 (Google Trends 類似)
        "llmo_popularity": llmo_popularity,  # AIサーチでの出現頻度推定 0–1
        "search_volume_est": search_vol,  # 月間検索数推定
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
        matches = re.findall(r"\b" + kw_lower + r"\b", text.lower())
        keyword_density = len(matches) / len(words) if words else 0.0

    # 改善推奨を生成
    recommendations: list[str] = []
    if llmo.entity_density < 0.4:
        recommendations.append(
            "エンティティ密度を上げる: 数値・固有名詞・専門語を追加してください"
        )
    if llmo.answer_directness < 0.4:
        recommendations.append(
            "冒頭1文で直接答える answer-first 形式に変更してください"
        )
    if llmo.citability < 0.4:
        recommendations.append(
            "統計データや出典を追加して引用されやすさを向上させてください"
        )
    if keyword_density > 0.02:
        recommendations.append(
            f"キーワード'{target_keyword}'の出現が多すぎます (密度: {keyword_density:.1%}, SEO推奨: 1-2%)"
        )
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

    if len(re.findall(r"\d+", text)) > 3:
        return "entity_rich"
    return "general"


def quality_score(
    ad_text: str,
    landing_page_text: str,
    keyword: str,
    historical_ctr: Optional[float] = None,
) -> dict:
    """
    Google Ads Quality Score (1–10) を推定する。

    3つのサブスコアを合成:
        expected_ctr         -- キーワード×広告文の関連性（クリック率期待値）
        ad_relevance         -- 広告文とキーワードの関連度
        landing_page_exp     -- LP とキーワードの関連度・読み込み速度推定

    Args:
        ad_text            -- 広告コピーテキスト
        landing_page_text  -- LP の本文テキスト
        keyword            -- ターゲットキーワード
        historical_ctr     -- 過去の実績 CTR (オプション, 0.0–1.0)

    Returns:
        {
            "quality_score": int (1–10),
            "expected_ctr": str ("below_average" | "average" | "above_average"),
            "ad_relevance": str,
            "landing_page_exp": str,
            "sub_scores": dict,
            "recommendations": list[str],
        }
    """
    kw_lower = keyword.lower()

    # --- expected_ctr ---
    # キーワードが広告文に含まれるか + 過去 CTR
    ad_kw_hit = kw_lower in ad_text.lower()
    if historical_ctr is not None:
        if historical_ctr >= 0.05:
            ctr_score = 3
        elif historical_ctr >= 0.02:
            ctr_score = 2
        else:
            ctr_score = 1
    else:
        ctr_score = 2 if ad_kw_hit else 1

    ctr_label = {3: "above_average", 2: "average", 1: "below_average"}[ctr_score]

    # --- ad_relevance ---
    # 広告文中のキーワード出現 + 文字長チェック (25–90字が最適)
    ad_len = len(ad_text)
    ad_kw_count = ad_text.lower().count(kw_lower)
    rel_score = 0
    if ad_kw_count >= 1:
        rel_score += 1
    if ad_kw_count >= 2:
        rel_score += 1
    if 25 <= ad_len <= 90:
        rel_score += 1
    rel_score = min(3, rel_score)
    rel_label = {3: "above_average", 2: "average", 1: "average", 0: "below_average"}[
        rel_score
    ]

    # --- landing_page_exp ---
    # LP のキーワード出現 + テキスト量
    lp_kw_count = landing_page_text.lower().count(kw_lower)
    lp_len = len(landing_page_text.split())
    lp_score = 0
    if lp_kw_count >= 2:
        lp_score += 1
    if lp_kw_count >= 4:
        lp_score += 1
    if lp_len >= 200:
        lp_score += 1
    lp_score = min(3, lp_score)
    lp_label = {3: "above_average", 2: "average", 1: "average", 0: "below_average"}[
        lp_score
    ]

    # --- 総合 QS (1–10) ---
    # expected_ctr: 40%, ad_relevance: 35%, landing_page: 25%
    raw = ctr_score * 0.40 + rel_score * 0.35 + lp_score * 0.25
    qs = max(1, min(10, round(raw / 3.0 * 9 + 1)))

    # --- 改善推奨 ---
    recommendations: list[str] = []
    if ctr_score < 3:
        recommendations.append(
            f"expected_ctr 改善: 広告文の先頭にキーワード「{keyword}」を含め、"
            f"ベネフィットを明示してください"
        )
    if rel_score < 3:
        recommendations.append(
            f"ad_relevance 改善: 見出し1・見出し2にキーワードを含め、"
            f"文字数を 25〜90字に調整してください（現在: {ad_len}字）"
        )
    if lp_score < 3:
        recommendations.append(
            f"landing_page_exp 改善: LP の冒頭にキーワード「{keyword}」を含む"
            f"明確な価値提案を追加し、本文を 200語以上にしてください"
        )
    if not recommendations:
        recommendations.append(
            f"Quality Score {qs} — 高品質。入札戦略の最適化を検討してください"
        )

    return {
        "quality_score": qs,
        "expected_ctr": ctr_label,
        "ad_relevance": rel_label,
        "landing_page_exp": lp_label,
        "sub_scores": {
            "expected_ctr_raw": ctr_score,
            "ad_relevance_raw": rel_score,
            "landing_page_raw": lp_score,
        },
        "keyword": keyword,
        "recommendations": recommendations,
    }


def generate_ad_variants(
    product: str,
    keyword: str,
    n_variants: int = 5,
    style: str = "mixed",
) -> dict:
    """
    広告コピーのバリアントを複数生成し LLMO スコア付きで返す。

    Claude Opus 4.8 の「1クエリで多様なバリアントを生成できない」弱点に対し、
    ルールベースで確実に n_variants 件を生成する。

    Args:
        product    -- 商品・サービス名
        keyword    -- ターゲットキーワード
        n_variants -- 生成するバリアント数 (デフォルト 5)
        style      -- "benefit" / "urgency" / "social_proof" / "question" / "mixed"

    Returns:
        {
            "variants": list[{"headline", "description", "llmo_score", "qs_estimate"}],
            "best_variant_index": int,
            "keyword": str,
        }
    """
    from open_mythos.llmo import LLMOScorer

    scorer = LLMOScorer()

    templates = [
        # benefit
        (
            f"{keyword}で売上 UP | {product}",
            f"{product}は{keyword}に特化した専門サービスです。"
            f"導入企業の平均 ROAS 3.8x・CTR 32%向上を実現。無料トライアル受付中。",
        ),
        # urgency
        (
            f"【期間限定】{keyword}ツール | {product}",
            f"今だけ初月無料。{keyword}の効率を最大化する{product}。"
            f"解約自由・導入2週間で効果実感。",
        ),
        # social proof
        (
            f"{keyword}で 3,000社以上が選ぶ | {product}",
            f"導入実績 3,000社超。{keyword}の専門家が使う{product}で"
            f"競合に差をつけてください。",
        ),
        # question
        (
            f"{keyword}に悩んでいませんか？ | {product}",
            f"{keyword}の課題を根本解決。{product}なら設定30分・"
            f"翌日から成果が出る仕組みを提供します。",
        ),
        # data-driven
        (
            f"{keyword} ROI +200% の実績 | {product}",
            f"事例: A社は{product}導入後に{keyword}経由売上が前年比+200%。"
            f"業界標準の2倍の精度で {keyword} を最適化。",
        ),
    ]

    variants = []
    for i in range(min(n_variants, len(templates))):
        headline, description = templates[i]
        full_text = f"{headline}\n{description}"
        llmo = scorer.score(full_text)

        # 簡易 QS 推定（広告文とKWの関連度のみ）
        qs_data = quality_score(
            ad_text=headline,
            landing_page_text=description,
            keyword=keyword,
        )

        variants.append(
            {
                "headline": headline,
                "description": description,
                "llmo_score": llmo.llmo_total,
                "qs_estimate": qs_data["quality_score"],
                "style": [
                    "benefit",
                    "urgency",
                    "social_proof",
                    "question",
                    "data_driven",
                ][i],
            }
        )

    # LLMO スコア × QS の複合スコアで最良バリアントを選定
    best_idx = max(
        range(len(variants)),
        key=lambda i: variants[i]["llmo_score"] * 0.6
        + variants[i]["qs_estimate"] / 10 * 0.4,
    )

    return {
        "variants": variants,
        "best_variant_index": best_idx,
        "keyword": keyword,
        "product": product,
    }


# ---------------------------------------------------------------------------
# レジストリへの一括登録
# ---------------------------------------------------------------------------

_MARKETING_TOOLS = [
    ToolDefinition(
        name="search_competitor",
        description="競合他社の広告費・CTR・SEOスコア・市場シェアを検索する。マーケティング戦略立案に使用。",
        parameters={
            "company": ParameterSchema(
                type="string", description="競合企業名 (例: 'Jasper AI')", required=True
            ),
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
            "ad_spend": ParameterSchema(
                type="number", description="広告費 (USD)", required=True
            ),
            "revenue": ParameterSchema(
                type="number", description="売上 (USD)", required=True
            ),
            "cogs": ParameterSchema(
                type="number", description="売上原価 (USD)", required=False, default=0.0
            ),
            "clicks": ParameterSchema(
                type="integer",
                description="クリック数 (オプション)",
                required=False,
                default=None,
            ),
            "impressions": ParameterSchema(
                type="integer",
                description="インプレッション数 (オプション)",
                required=False,
                default=None,
            ),
        },
        fn=calculate_roi,
    ),
    ToolDefinition(
        name="fetch_trend",
        description="キーワードのトレンドスコアとLLMO人気度を取得する。コンテンツ戦略立案に使用。",
        parameters={
            "keyword": ParameterSchema(
                type="string", description="検索キーワード", required=True
            ),
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
            "text": ParameterSchema(
                type="string", description="スコアリング対象テキスト", required=True
            ),
            "target_keyword": ParameterSchema(
                type="string",
                description="ターゲットキーワード",
                required=False,
                default="",
            ),
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


_MARKETING_TOOLS += [
    ToolDefinition(
        name="quality_score",
        description="Google Ads Quality Score (1-10) を推定する。広告最適化・入札戦略に使用。",
        parameters={
            "ad_text": ParameterSchema(
                type="string", description="広告コピーテキスト", required=True
            ),
            "landing_page_text": ParameterSchema(
                type="string", description="LP本文テキスト", required=True
            ),
            "keyword": ParameterSchema(
                type="string", description="ターゲットキーワード", required=True
            ),
            "historical_ctr": ParameterSchema(
                type="number",
                description="過去の実績CTR (0.0-1.0, オプション)",
                required=False,
                default=None,
            ),
        },
        fn=quality_score,
    ),
    ToolDefinition(
        name="generate_ad_variants",
        description="広告コピーのバリアントを複数生成しLLMOスコア・QS付きで返す。A/Bテスト素材生成に使用。",
        parameters={
            "product": ParameterSchema(
                type="string", description="商品・サービス名", required=True
            ),
            "keyword": ParameterSchema(
                type="string", description="ターゲットキーワード", required=True
            ),
            "n_variants": ParameterSchema(
                type="integer",
                description="生成するバリアント数 (デフォルト5)",
                required=False,
                default=5,
            ),
            "style": ParameterSchema(
                type="string",
                description="広告スタイル",
                enum=["benefit", "urgency", "social_proof", "question", "mixed"],
                required=False,
                default="mixed",
            ),
        },
        fn=generate_ad_variants,
    ),
]


def register_marketing_tools(registry: ToolRegistry) -> None:
    """マーケ特化6ツールをレジストリに登録する。"""
    for tool_def in _MARKETING_TOOLS:
        registry.register(tool_def)


# ---------------------------------------------------------------------------
# Sprint 18: ROAS Monte Carlo シミュレーター (18.5)
# ---------------------------------------------------------------------------


def roas_simulate(
    ad_spend: float,
    ctr: float,
    cvr: float,
    aov: float,
    n: int = 1000,
    noise: float = 0.20,
    seed: int | None = None,
    noise_dist: str = "uniform",
) -> dict:
    """
    モンテカルロ法による ROAS 予測（信頼区間付き）。

    各パラメータにノイズを加えて ``n`` 回シミュレーションし、
    ROAS 分布の統計量と 90%/50% 信頼区間を返す。

    モデル::

        clicks_i      = ad_spend × ctr_i          # ctr_i = クリック/USD
        conversions_i = clicks_i × cvr_i           # cvr_i = 成約率
        revenue_i     = conversions_i × aov_i      # aov_i = 平均注文金額 (USD)
        roas_i        = revenue_i / ad_spend = ctr_i × cvr_i × aov_i

    Parameters
    ----------
    ad_spend : float
        広告費 (USD)。
    ctr : float
        期待クリック率 (clicks per USD invested)。
        例: 0.1 → 1ドルあたり 0.1 クリック期待。
    cvr : float
        期待成約率 (0.0–1.0)。例: 0.05 → 5%。
    aov : float
        平均注文金額 (USD)。例: 5000.0 → 5,000 USD。
    n : int
        シミュレーション回数。デフォルト 1,000。
    noise : float
        ノイズ幅。uniform: ±noise、normal: 標準偏差 = noise。デフォルト 0.20。
    seed : int | None
        乱数シード。None の場合は非決定的。
    noise_dist : str
        ノイズ分布。"uniform"（一様、デフォルト）または "normal"（正規分布）。

    Returns
    -------
    dict
        mean_roas / std_dev / ci90 / ci50 / p5〜p95 /
        profitable_probability / expected_revenue_usd 等。
    """
    import math

    if ad_spend <= 0:
        raise ValueError("ad_spend must be > 0")
    if not 0 < ctr:
        raise ValueError("ctr must be > 0")
    if not 0 < cvr <= 1:
        raise ValueError("cvr must be in (0, 1]")
    if aov <= 0:
        raise ValueError("aov must be > 0")
    if n < 1:
        raise ValueError("n must be >= 1")
    if noise_dist not in ("uniform", "normal"):
        raise ValueError("noise_dist must be 'uniform' or 'normal'")

    rng = random.Random(seed)
    samples: list[float] = []

    for _ in range(n):
        if noise_dist == "normal":
            ctr_i = max(1e-9, ctr * (1.0 + rng.gauss(0, noise)))
            cvr_i = min(1.0, max(1e-9, cvr * (1.0 + rng.gauss(0, noise))))
            aov_i = max(1e-9, aov * (1.0 + rng.gauss(0, noise)))
        else:
            ctr_i = max(1e-9, ctr * (1.0 + rng.uniform(-noise, noise)))
            cvr_i = min(1.0, max(1e-9, cvr * (1.0 + rng.uniform(-noise, noise))))
            aov_i = max(1e-9, aov * (1.0 + rng.uniform(-noise, noise)))
        samples.append(ctr_i * cvr_i * aov_i)

    samples.sort()
    mean = sum(samples) / n
    variance = sum((x - mean) ** 2 for x in samples) / n

    def _percentile(p: float) -> float:
        idx = max(0, min(n - 1, int(n * p)))
        return samples[idx]

    return {
        "n_simulations": n,
        "noise_dist": noise_dist,
        "ad_spend_usd": round(ad_spend, 2),
        "mean_roas": round(mean, 4),
        "std_dev": round(math.sqrt(variance), 4),
        "ci90": {
            "lower": round(_percentile(0.05), 4),
            "upper": round(_percentile(0.95), 4),
        },
        "ci50": {
            "lower": round(_percentile(0.25), 4),
            "upper": round(_percentile(0.75), 4),
        },
        "p5_roas": round(_percentile(0.05), 4),
        "p25_roas": round(_percentile(0.25), 4),
        "p50_roas": round(_percentile(0.50), 4),
        "p75_roas": round(_percentile(0.75), 4),
        "p95_roas": round(_percentile(0.95), 4),
        "profitable_probability": round(sum(1 for r in samples if r > 1.0) / n, 4),
        "expected_revenue_usd": round(mean * ad_spend, 2),
        "break_even_roas": 1.0,
        "inputs": {"ctr": ctr, "cvr": cvr, "aov": aov, "noise": noise},
    }


# ---------------------------------------------------------------------------
# Sprint 18: ペルソナ × 広告マッチング (18.6)
# ---------------------------------------------------------------------------


def persona_ad_match(
    persona_doc: str,
    ad_candidates: list[str],
    top_k: int = 3,
) -> dict:
    """
    ペルソナドキュメントと広告コピー候補のマッチングスコアを算出する。

    TF-IDF コサイン類似度ベースで各候補を採点し、上位 ``top_k`` 件を
    返す。外部依存なし (numpy のみ)。

    実装原理
    --------
    1. persona_doc + 全 ad_candidates をコーパスとして TF-IDF 行列を構築
    2. persona_doc ベクトルと各 ad candidate ベクトルのコサイン類似度を計算
    3. スコア降順で上位 top_k 件を返す

    Parameters
    ----------
    persona_doc : str
        ペルソナの説明文（例: 「30代女性・育児中・時短家事に関心」）。
    ad_candidates : list[str]
        広告コピー候補の文字列リスト。
    top_k : int
        返す上位件数。デフォルト 3。

    Returns
    -------
    dict
        ranked: [{rank, ad_text, score}, ...] (top_k件)
        best_match: 最高スコアの広告コピー
        persona_keywords: ペルソナから抽出した上位トークン
    """
    import math

    if not ad_candidates:
        raise ValueError("ad_candidates must not be empty")

    top_k = min(top_k, len(ad_candidates))

    # --- シンプルな前処理 (記号除去・小文字化) ---
    def _tokenize(text: str) -> list[str]:
        # 日本語は文字単位、ASCII は単語単位
        tokens: list[str] = []
        for ch in re.sub(r"[^\w\s]", " ", text):
            if ch.strip():
                tokens.append(ch.lower())
        return tokens

    corpus = [persona_doc] + list(ad_candidates)
    tokenized = [_tokenize(doc) for doc in corpus]

    # --- TF 計算 ---
    def _tf(tokens: list[str]) -> dict[str, float]:
        total = max(1, len(tokens))
        freq: dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        return {t: c / total for t, c in freq.items()}

    tfs = [_tf(t) for t in tokenized]
    vocab = sorted({tok for doc in tokenized for tok in doc})

    # --- IDF 計算 ---
    n_docs = len(corpus)
    idf: dict[str, float] = {}
    for term in vocab:
        df = sum(1 for doc in tokenized if term in doc)
        idf[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

    # --- TF-IDF ベクトル & コサイン類似度 ---
    def _vec(tf: dict[str, float]) -> list[float]:
        return [tf.get(t, 0.0) * idf[t] for t in vocab]

    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na * nb > 0 else 0.0

    persona_vec = _vec(tfs[0])
    scores = [_cosine(persona_vec, _vec(tfs[i + 1])) for i in range(len(ad_candidates))]

    # --- ランキング ---
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked = [
        {
            "rank": rank + 1,
            "ad_text": ad_candidates[idx],
            "score": round(scores[idx], 6),
        }
        for rank, idx in enumerate(ranked_idx[:top_k])
    ]

    # ペルソナの上位キーワード (IDF 重み付き)
    persona_tokens = tokenized[0]
    persona_tf = tfs[0]
    kw_scores = {t: persona_tf.get(t, 0) * idf.get(t, 1) for t in set(persona_tokens)}
    top_keywords = sorted(kw_scores, key=lambda t: kw_scores[t], reverse=True)[:10]

    return {
        "ranked": ranked,
        "best_match": ranked[0]["ad_text"] if ranked else "",
        "best_score": ranked[0]["score"] if ranked else 0.0,
        "persona_keywords": top_keywords,
        "n_candidates": len(ad_candidates),
        "top_k": top_k,
    }
