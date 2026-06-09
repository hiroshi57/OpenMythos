"""
serve/ad_router.py — 広告コピー生成 + LLMO分析エンドポイント

POST /v1/ad/generate   : 3モード対応 広告コピーN案生成 + LLMOスコアリング
POST /v1/ad/refine     : ブラッシュアップ（段階的精査）
POST /v1/llmo/analyze  : LLMO指標分析（言及率・引用率・参照率ポテンシャル）
GET  /ad               : ブラウザUI（3タブ・カテゴリスコア可視化）
"""
from __future__ import annotations
import os, time
from enum import Enum
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

router = APIRouter()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# リクエスト / レスポンス モデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PromptMode(str, Enum):
    CEP     = "cep"      # AIに聞くシチュエーション起点
    KEYWORD = "keyword"  # カテゴリ・キーワード起点（デフォルト）
    BRAND   = "brand"    # ブランド起点

class KeywordVolume(BaseModel):
    keyword: str = Field(..., description="キーワード")
    volume:  int = Field(..., description="月間検索ボリューム（概算）")

class CepInput(BaseModel):
    """CEPモード専用入力: ユーザーがAIに問い合わせるシチュエーション"""
    scenario: str          = Field(..., description="AIに聞くシチュエーション（例: 日焼けが気になって検索する30代女性）")
    category: Optional[str] = Field(None,  description="商品カテゴリ（例: 日焼け止め・スキンケア）")

class BrandInput(BaseModel):
    """ブランドモード専用入力"""
    name:   str                    = Field(..., description="ブランド名")
    tone:   Optional[str]          = Field(None, description="ブランドトーン（例: 親しみやすく自然体）")
    values: Optional[list[str]]    = Field(None, description="ブランド価値観（例: ['自由','自然','信頼']）")
    target: Optional[str]          = Field(None, description="ターゲット（例: アウトドア好きな30代女性）")

class ScoreBreakdown(BaseModel):
    conciseness:      float      = Field(..., description="簡潔さ・読みやすさ")
    emotional_impact: float      = Field(..., description="感情的インパクト")
    target_fit:       float      = Field(..., description="ターゲット適合性")
    memorability:     float      = Field(..., description="記憶に残りやすさ")
    action_power:     float      = Field(..., description="行動促進力")
    search_potential: float      = Field(..., description="検索ポテンシャル")
    total:            float      = Field(..., description="総合スコア")
    matched_keywords: list[str]  = Field(default_factory=list, description="含まれた高ボリュームキーワード")

class AdVariant(BaseModel):
    text:            str
    score:           float
    score_breakdown: ScoreBreakdown
    rank:            int

class AdGenerateResponse(BaseModel):
    variants:     list[AdVariant]
    best:         str
    best_score:   float
    latency_ms:   float
    mode_used:    str = Field("keyword", description="使用したプロンプトモード")

class AdGenerateRequest(BaseModel):
    prompt:          str                           = Field(..., description="広告依頼の自然文")
    mode:            PromptMode                    = Field(PromptMode.KEYWORD, description="プロンプトモード")
    n:               int                           = Field(8, ge=1, le=20)
    temperature:     float                         = Field(0.9, ge=0.0, le=2.0)
    keyword:         Optional[str]                 = Field(None, description="LLMOスコア用キーワード")
    keyword_volumes: Optional[list[KeywordVolume]] = Field(None, description="検索ボリューム付きキーワード")
    cep:             Optional[CepInput]            = Field(None, description="CEPモード用入力")
    brand:           Optional[BrandInput]          = Field(None, description="ブランドモード用入力")

class AdRefineRequest(BaseModel):
    base:            str           = Field(..., description="ベースとなるコピー")
    direction:       str           = Field(..., description="改善方向（例: もっと短く）")
    original_prompt: str           = Field(..., description="元の依頼文")
    n:               int           = Field(5, ge=1, le=10)
    keyword:         Optional[str] = Field(None)

# ── LLMO分析モデル ──────────────────────────────────────────────

class LlmoAnalyzeRequest(BaseModel):
    texts:       list[str]           = Field(..., description="分析するコピーテキスト群")
    brand_name:  Optional[str]       = Field(None, description="ブランド名（言及率計算）")
    cep_prompts: Optional[list[str]] = Field(None, description="想定CEPプロンプト群")
    keywords:    Optional[list[str]] = Field(None, description="ターゲットキーワード群")

class LlmoMetrics(BaseModel):
    mention_potential:    float      = Field(..., description="言及ポテンシャル（0-1）")
    citation_potential:   float      = Field(..., description="引用ポテンシャル（0-1）")
    reference_potential:  float      = Field(..., description="参照ポテンシャル（0-1）")
    mention_rate_estimate: str       = Field(..., description="言及率の概算")
    recommendations:      list[str]  = Field(default_factory=list, description="改善推奨事項")

class LlmoAnalyzeResponse(BaseModel):
    metrics:       LlmoMetrics
    top_copy:      str = Field(..., description="最もLLMO指標が高いコピー")
    analysis_note: str = Field(..., description="分析コメント")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# システムプロンプト（モード別）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYSTEM_KEYWORD = (
    "あなたはプロの広告コピーライターです。"
    "提示されたカテゴリ・キーワードを自然に組み込んだ、"
    "短くて印象的な広告コピーを1案だけ返してください。"
    "説明不要。コピーの文章のみ返してください。"
)

_SYSTEM_CEP = (
    "あなたはプロの広告コピーライターです。"
    "提示されたユーザーシチュエーション（CEP: Customer Entry Point）を深く理解し、"
    "そのユーザーがAIに質問したとき、AIが自然に推薦したくなるような"
    "印象的で記憶に残る広告コピーを1案だけ返してください。"
    "CEPで想定されるユーザーの悩み・欲求・文脈をコピーに反映させてください。"
    "説明不要。コピーの文章のみ返してください。"
)

_SYSTEM_BRAND = (
    "あなたはプロの広告コピーライターです。"
    "提示されたブランドのトーン・価値観・ターゲットを忠実に反映し、"
    "そのブランドらしさが直感的に伝わる広告コピーを1案だけ返してください。"
    "ブランドの世界観を壊さず、かつ記憶に残る表現にしてください。"
    "説明不要。コピーの文章のみ返してください。"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 内部関数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_system_and_user(req: AdGenerateRequest) -> tuple[str, str]:
    """モードに応じた (system_prompt, user_prompt) を返す"""

    if req.mode == PromptMode.CEP and req.cep:
        system = _SYSTEM_CEP
        user   = f"【シチュエーション（CEP）】\n{req.cep.scenario}"
        if req.cep.category:
            user += f"\n【カテゴリ】{req.cep.category}"
        if req.prompt:
            user += f"\n【追加依頼】{req.prompt}"

    elif req.mode == PromptMode.BRAND and req.brand:
        system = _SYSTEM_BRAND
        user   = f"【ブランド名】{req.brand.name}"
        if req.brand.tone:
            user += f"\n【トーン】{req.brand.tone}"
        if req.brand.values:
            user += f"\n【価値観】{', '.join(req.brand.values)}"
        if req.brand.target:
            user += f"\n【ターゲット】{req.brand.target}"
        if req.prompt:
            user += f"\n【依頼内容】{req.prompt}"

    else:  # keyword (default)
        system = _SYSTEM_KEYWORD
        user   = req.prompt

    # 検索ボリューム情報を全モードに付加
    if req.keyword_volumes:
        sorted_kws = sorted(req.keyword_volumes, key=lambda x: x.volume, reverse=True)
        top_kws = [f"{kv.keyword}（月{kv.volume:,}検索）" for kv in sorted_kws[:5]]
        user += (
            "\n\n【検索ボリュームの高いキーワード（自然に含めると効果的）】\n"
            + "\n".join(f"  ・{k}" for k in top_kws)
        )

    return system, user


def _call_claude(system: str, user: str, temperature: float = 0.9) -> str:
    """Claude API を呼び出して1案生成する"""
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY が設定されていません")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=150,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Claude API エラー: {e}")


def _llmo_score(
    text: str,
    keyword: str = "",
    keyword_volumes: list[KeywordVolume] | None = None,
) -> ScoreBreakdown:
    """6カテゴリーLLMOスコアリング"""
    length    = len(text)
    chars     = list(text)
    uniqueness = len(set(chars)) / max(len(chars), 1)

    # 1. 簡潔さ
    if   8  <= length <= 22: conciseness = 1.00
    elif 23 <= length <= 35: conciseness = 0.85
    elif 36 <= length <= 50: conciseness = 0.65
    elif length < 8:         conciseness = 0.40
    else:                    conciseness = 0.45

    # 2. 感情的インパクト
    em = 0.3
    strong_words = ["愛", "夢", "自由", "笑", "輝", "守", "届", "刺さ", "溢", "誇"]
    punctuation  = ["！", "。", "…", "—", "・"]
    contrast     = ["でも", "だから", "こそ", "だって", "のに", "から"]
    if any(w in text for w in strong_words): em += 0.30
    if any(p in text for p in punctuation):  em += 0.20
    if any(c in text for c in contrast):     em += 0.20
    emotional_impact = round(min(em, 1.0), 2)

    # 3. ターゲット適合性
    ts = 0.3
    if keyword:
        kws = keyword.split()
        matched = sum(1 for k in kws if k in text)
        ts += min(matched / max(len(kws), 1) * 0.7, 0.7)
    else:
        ts = 0.5
    target_fit = round(min(ts, 1.0), 2)

    # 4. 記憶に残りやすさ
    ms = 0.2
    if uniqueness > 0.70:      ms += 0.25
    if "、" in text:            ms += 0.15
    if text.count("。") == 1:  ms += 0.20
    endings = ("。", "！", "自由", "選択", "味方", "相棒", "はじまり", "未来")
    if text.endswith(endings): ms += 0.20
    memorability = round(min(ms, 1.0), 2)

    # 5. 行動促進力
    ac = 0.2
    action_words = ["選ぶ", "選んだ", "決める", "始める", "使う", "試す",
                    "守る", "感じる", "楽しむ", "変える", "しよう", "ください",
                    "あなたへ", "必需品", "必携"]
    question     = ["？", "か。", "でしょう"]
    if any(w in text for w in action_words): ac += 0.40
    if any(q in text for q in question):     ac += 0.20
    if "あなた" in text:                     ac += 0.20
    action_power = round(min(ac, 1.0), 2)

    # 6. 検索ポテンシャル
    matched_kws: list[str] = []
    if keyword_volumes:
        max_vol   = max((kv.volume for kv in keyword_volumes), default=1)
        pot_score = 0.0
        for kv in keyword_volumes:
            if kv.keyword in text:
                matched_kws.append(kv.keyword)
                pot_score += (kv.volume / max_vol) * 0.4
        search_potential = round(min(0.2 + pot_score, 1.0), 2)
    else:
        search_potential = target_fit

    # 総合スコア
    has_vol = bool(keyword_volumes)
    total = round(
        conciseness      * 0.15 +
        emotional_impact * 0.20 +
        target_fit       * 0.20 +
        memorability     * 0.15 +
        action_power     * 0.10 +
        search_potential * (0.20 if has_vol else 0.00) +
        target_fit       * (0.00 if has_vol else 0.20),
        2,
    )

    return ScoreBreakdown(
        conciseness=conciseness,
        emotional_impact=emotional_impact,
        target_fit=target_fit,
        memorability=memorability,
        action_power=action_power,
        search_potential=search_potential,
        total=total,
        matched_keywords=matched_kws,
    )


def _calc_llmo_metrics(
    texts:       list[str],
    brand_name:  str | None,
    cep_prompts: list[str] | None,
    keywords:    list[str] | None,
) -> LlmoMetrics:
    """LLMO指標のポテンシャルをシミュレーション計算"""
    if not texts:
        return LlmoMetrics(
            mention_potential=0.0,
            citation_potential=0.0,
            reference_potential=0.0,
            mention_rate_estimate="0%",
            recommendations=["テキストを入力してください"],
        )

    recommendations: list[str] = []

    # ── 1. 言及ポテンシャル ─────────────────────────────────────────
    # AIがそのブランドを言及する可能性: ブランド名含有 + キーワード密度 + 簡潔さ
    mention_scores = []
    for text in texts:
        s = 0.25
        if brand_name and brand_name in text:
            s += 0.40
        if keywords:
            hit = sum(1 for k in keywords if k in text) / len(keywords)
            s += hit * 0.25
        if 8 <= len(text) <= 30:
            s += 0.10  # 短く記憶しやすい
        mention_scores.append(min(s, 1.0))
    mention_potential = round(sum(mention_scores) / len(mention_scores), 2)

    # ── 2. 引用ポテンシャル ─────────────────────────────────────────
    # AIが情報源として引用する可能性: 権威性・具体性・情報量
    citation_scores = []
    authority  = ["確か", "実証", "研究", "効果", "成分", "科学", "認定", "実績",
                  "No.1", "1位", "受賞", "特許", "安全", "推奨"]
    specifics  = ["SPF", "UV", "PA", "%", "mg", "ml", "g", "年", "日", "回"]
    for text in texts:
        s = 0.20
        if any(w in text for w in authority): s += 0.35
        if any(w in text for w in specifics): s += 0.25
        if len(text) >= 15: s += 0.20
        citation_scores.append(min(s, 1.0))
    citation_potential = round(sum(citation_scores) / len(citation_scores), 2)

    # ── 3. 参照ポテンシャル ─────────────────────────────────────────
    # AIクローラーが自社ページを参照する可能性: SEO密度 + CEP対応度 + 構造
    ref_scores = []
    for text in texts:
        s = 0.25
        if keywords:
            kc = sum(text.count(k) for k in keywords)
            if 1 <= kc <= 3: s += 0.25  # 適度なキーワード密度
        if "。" in text or "、" in text: s += 0.15
        if cep_prompts:
            for cep in cep_prompts:
                cep_words  = set(cep.replace("。", " ").replace("、", " ").split())
                text_words = set(text.replace("。", " ").replace("、", " ").split())
                overlap = len(cep_words & text_words) / max(len(cep_words), 1)
                if overlap > 0.1: s += 0.15
        ref_scores.append(min(s, 1.0))
    reference_potential = round(sum(ref_scores) / len(ref_scores), 2)

    # ── 推奨事項 ───────────────────────────────────────────────────
    if mention_potential < 0.5:
        if brand_name:
            recommendations.append(f"ブランド名「{brand_name}」をコピーに自然に組み込むと言及率が向上します")
        else:
            recommendations.append("ブランド名を明示的に含めることで言及率が向上します")
    if citation_potential < 0.4:
        recommendations.append("「実証済み」「成分」「効果」など具体的・権威的なワードを追加すると引用率が上がります")
    if reference_potential < 0.4:
        recommendations.append("CEPシチュエーションと対応するキーワードを含めると参照率が向上します")
    if not recommendations:
        recommendations.append("LLMO指標は良好です。定点観測で継続改善を続けましょう")

    mention_rate_pct      = int(mention_potential * 65)
    mention_rate_estimate = f"約{mention_rate_pct}%"

    return LlmoMetrics(
        mention_potential=mention_potential,
        citation_potential=citation_potential,
        reference_potential=reference_potential,
        mention_rate_estimate=mention_rate_estimate,
        recommendations=recommendations,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.post("/v1/ad/generate", response_model=AdGenerateResponse, tags=["ad"])
def ad_generate(req: AdGenerateRequest):
    """
    3モード対応 広告コピーN案生成 + 6カテゴリーLLMOスコアリング。

    **mode**:
    - `keyword` (デフォルト): カテゴリ・キーワードを起点に生成
    - `cep`: ユーザーがAIに聞くシチュエーション（CEP）を起点に生成
    - `brand`: ブランドトーン・価値観を起点に生成
    """
    t0      = time.perf_counter()
    keyword = req.keyword or ""
    system, user = _build_system_and_user(req)

    variants_raw = []
    for _ in range(req.n):
        text      = _call_claude(system, user, req.temperature)
        breakdown = _llmo_score(text, keyword, req.keyword_volumes)
        variants_raw.append((breakdown.total, text, breakdown))

    variants_raw.sort(reverse=True, key=lambda x: x[0])
    variants = [
        AdVariant(text=t, score=s, score_breakdown=bd, rank=i + 1)
        for i, (s, t, bd) in enumerate(variants_raw)
    ]

    latency_ms = (time.perf_counter() - t0) * 1000
    return AdGenerateResponse(
        variants=variants,
        best=variants[0].text,
        best_score=variants[0].score,
        latency_ms=round(latency_ms),
        mode_used=req.mode.value,
    )


@router.post("/v1/ad/refine", response_model=AdGenerateResponse, tags=["ad"])
def ad_refine(req: AdRefineRequest):
    """既存コピーをベースにブラッシュアップ（段階的精査）"""
    t0      = time.perf_counter()
    keyword = req.keyword or ""

    system = (
        "あなたはプロの広告コピーライターです。"
        "提示された広告コピーをベースに、指示通りにブラッシュアップしてください。"
        "1案だけ返してください。説明不要。コピーの文章のみ返してください。"
    )
    user = (
        f"元の依頼: {req.original_prompt}\n"
        f"ベースのコピー: {req.base}\n"
        f"改善方向: {req.direction}\n"
        "より印象的で品質の高い1案を返してください。"
    )

    variants_raw = []
    for _ in range(req.n):
        text      = _call_claude(system, user, temperature=0.8)
        breakdown = _llmo_score(text, keyword)
        variants_raw.append((breakdown.total, text, breakdown))

    variants_raw.sort(reverse=True, key=lambda x: x[0])
    variants = [
        AdVariant(text=t, score=s, score_breakdown=bd, rank=i + 1)
        for i, (s, t, bd) in enumerate(variants_raw)
    ]

    latency_ms = (time.perf_counter() - t0) * 1000
    return AdGenerateResponse(
        variants=variants,
        best=variants[0].text,
        best_score=variants[0].score,
        latency_ms=round(latency_ms),
        mode_used="refine",
    )


@router.post("/v1/llmo/analyze", response_model=LlmoAnalyzeResponse, tags=["llmo"])
def llmo_analyze(req: LlmoAnalyzeRequest):
    """
    LLMO指標（言及率・引用率・参照率）ポテンシャルを分析する。

    AIサーチで自社ブランドが「言及」「引用」「参照」される可能性を
    広告コピー群から数値化し、改善推奨を返します。
    """
    metrics = _calc_llmo_metrics(
        texts=req.texts,
        brand_name=req.brand_name,
        cep_prompts=req.cep_prompts,
        keywords=req.keywords,
    )

    # 最高スコアのコピーを特定
    best_text = max(
        req.texts,
        key=lambda t: _llmo_score(
            t,
            keyword=" ".join(req.keywords or []),
        ).total,
    ) if req.texts else ""

    avg = (metrics.mention_potential + metrics.citation_potential + metrics.reference_potential) / 3
    if avg >= 0.7:
        note = "LLMO総合評価: 優秀。AIサーチでの言及・引用・参照が期待できます。"
    elif avg >= 0.5:
        note = "LLMO総合評価: 標準。推奨事項を実施することで大きく改善できます。"
    else:
        note = "LLMO総合評価: 改善余地あり。ブランド名・具体的キーワード・CEP対応を強化してください。"

    return LlmoAnalyzeResponse(metrics=metrics, top_copy=best_text, analysis_note=note)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ブラウザ UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/ad", response_class=HTMLResponse, include_in_schema=False)
def ad_ui():
    """広告コピー生成 + LLMO分析 ブラウザUI"""
    return HTMLResponse(content=r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenMythos 広告コピー生成</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Hiragino Sans','Yu Gothic',sans-serif; background:#f5f5f7; color:#1d1d1f; }
.header { background:#1d1d1f; color:#fff; padding:18px 36px; }
.header h1 { font-size:20px; font-weight:600; }
.header p  { font-size:12px; color:#aaa; margin-top:4px; }
.container { max-width:860px; margin:32px auto; padding:0 20px; }

/* ── カード ─────────────────────────────── */
.card { background:#fff; border-radius:16px; padding:24px; margin-bottom:20px;
        box-shadow:0 2px 12px rgba(0,0,0,.08); }

/* ── タブ ───────────────────────────────── */
.tabs { display:flex; gap:6px; margin-bottom:18px; border-bottom:2px solid #eee; padding-bottom:0; }
.tab-btn { background:none; border:none; padding:10px 18px; font-size:13px; font-weight:600;
           color:#888; cursor:pointer; border-radius:8px 8px 0 0; border-bottom:3px solid transparent;
           margin-bottom:-2px; transition:all .15s; }
.tab-btn:hover  { background:#f5f5f7; color:#444; }
.tab-btn.active { color:#0071e3; border-bottom-color:#0071e3; background:#fff; }

/* ── フォーム ───────────────────────────── */
.mode-panel { display:none; }
.mode-panel.active { display:block; }
label { font-size:12px; font-weight:600; color:#666; display:block; margin:14px 0 6px; }
label:first-child { margin-top:0; }
textarea { width:100%; border:1.5px solid #ddd; border-radius:10px; padding:12px;
           font-size:14px; resize:vertical; min-height:72px; font-family:inherit; }
textarea:focus, input[type=text]:focus { outline:none; border-color:#0071e3; }
input[type=text] { width:100%; border:1.5px solid #ddd; border-radius:10px;
                   padding:10px 13px; font-size:13px; font-family:inherit; }
.row { display:flex; gap:12px; align-items:flex-end; }
.row .field { flex:1; }

/* ── キーワードボリューム行 ───────────── */
.kv-row { display:flex; gap:8px; align-items:center; margin-bottom:8px; }
.kv-row input { flex:2; }
.kv-row input.vol { flex:1; }
.kv-del { background:none; border:none; color:#c00; font-size:18px; cursor:pointer;
          padding:0 4px; line-height:1; }
.add-kv-btn { background:none; border:1.5px dashed #0071e3; color:#0071e3;
              border-radius:8px; padding:6px 14px; font-size:12px; cursor:pointer;
              margin-top:4px; }
.add-kv-btn:hover { background:#f0f7ff; }

/* ── ボタン ─────────────────────────────── */
.btn-primary { background:#0071e3; color:#fff; border:none; border-radius:10px;
               padding:13px 24px; font-size:14px; font-weight:600; cursor:pointer;
               width:100%; margin-top:14px; }
.btn-primary:hover    { background:#006ad6; }
.btn-primary:disabled { background:#bbb; cursor:not-allowed; }
.btn-secondary { background:#f5f5f7; color:#1d1d1f; border:1.5px solid #ddd;
                 border-radius:10px; padding:10px 20px; font-size:13px; font-weight:600;
                 cursor:pointer; width:100%; margin-top:12px; }
.btn-secondary:hover { background:#e8e8ea; }

/* ── バリアント ─────────────────────────── */
.variant { border:1.5px solid #eee; border-radius:12px; padding:14px 16px;
           margin-bottom:10px; cursor:pointer; transition:all .15s; }
.variant:hover    { border-color:#0071e3; background:#f7faff; }
.variant.selected { border-color:#0071e3; background:#eef5ff; }
.v-text { font-size:15px; margin-bottom:8px; line-height:1.5; }
.badge { display:inline-block; background:#f5c518; color:#333; font-size:10px;
         font-weight:700; padding:2px 8px; border-radius:20px; margin-bottom:6px; }

/* ── 総合スコアバー ─────────────────────── */
.score-row { display:flex; align-items:center; gap:8px; }
.bar-bg   { flex:1; height:5px; background:#eee; border-radius:3px; }
.bar-fill { height:5px; border-radius:3px; background:#0071e3; transition:width .5s; }
.score-num { font-size:12px; color:#666; width:32px; text-align:right; }

/* ── スコア詳細（折りたたみ） ───────────── */
details { margin-top:8px; }
summary { font-size:11px; color:#0071e3; cursor:pointer; user-select:none; }
summary:hover { text-decoration:underline; }
.sb-grid { display:grid; grid-template-columns:auto 1fr auto; gap:4px 8px;
           align-items:center; margin-top:8px; }
.sb-lbl { font-size:11px; color:#888; white-space:nowrap; }
.sb-bar-bg   { height:4px; background:#f0f0f0; border-radius:2px; }
.sb-bar-fill { height:4px; border-radius:2px; background:#64b5f6; transition:width .5s; }
.sb-bar-fill.hi { background:#0071e3; }
.sb-val { font-size:10px; color:#999; }
.kw-tags { margin-top:6px; }
.kw-tag { display:inline-block; background:#e3f2fd; color:#1565c0; font-size:10px;
          padding:2px 7px; border-radius:10px; margin:2px; }

/* ── ステップラベル ─────────────────────── */
.step-lbl { font-size:12px; color:#0071e3; font-weight:700; margin-bottom:12px; }

/* ── ローディング ───────────────────────── */
.loading { text-align:center; padding:36px; color:#666; }
.spinner { width:28px; height:28px; border:3px solid #eee; border-top-color:#0071e3;
           border-radius:50%; animation:spin .7s linear infinite; margin:0 auto 12px; }
@keyframes spin { to { transform:rotate(360deg); } }

/* ── LLMO分析カード ─────────────────────── */
.gauge-row { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:12px; }
.gauge { text-align:center; background:#f7faff; border-radius:12px; padding:14px 10px; }
.gauge .g-val { font-size:28px; font-weight:700; color:#0071e3; }
.gauge .g-lbl { font-size:11px; color:#666; margin-top:4px; }
.gauge .g-sub { font-size:10px; color:#999; margin-top:2px; }
.rec-list { margin-top:14px; }
.rec-item { font-size:12px; color:#444; padding:7px 12px; background:#fffbea;
            border-left:3px solid #f5c518; border-radius:4px; margin-bottom:6px; }

/* ── レスポンシブ ───────────────────────── */
@media(max-width:600px) {
  .gauge-row { grid-template-columns:1fr; }
  .row { flex-direction:column; }
}
</style>
</head>
<body>

<div class="header">
  <h1>OpenMythos 広告コピー生成</h1>
  <p>Claude AI × LLMO スコアリング — CEP / キーワード / ブランド の3モード対応</p>
</div>

<div class="container">

  <!-- ── 入力カード ─────────────────────────────────── -->
  <div class="card">

    <!-- タブ -->
    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('keyword',this)">🔍 キーワード起点</button>
      <button class="tab-btn"        onclick="switchTab('cep',this)"    >💡 CEP起点</button>
      <button class="tab-btn"        onclick="switchTab('brand',this)"  >🏷️ ブランド起点</button>
    </div>

    <!-- ── キーワードモード ── -->
    <div id="panel-keyword" class="mode-panel active">
      <label>広告の依頼内容</label>
      <textarea id="kw-prompt"
        placeholder="例: 30代女性向けの夏の日焼け止め広告コピーを作って。アウトドア派で自然体な人に刺さる感じで"
      ></textarea>

      <label>キーワード（任意 — スペース区切り）</label>
      <input type="text" id="kw-keyword" placeholder="例: 日焼け止め 夏 アウトドア">
    </div>

    <!-- ── CEPモード ── -->
    <div id="panel-cep" class="mode-panel">
      <label>AIに聞くシチュエーション（CEP）</label>
      <textarea id="cep-scenario"
        placeholder="例: 日焼けが気になって夏前に検索する30代女性が、アウトドア向け日焼け止めをAIに聞いているシーン"
      ></textarea>

      <div class="row">
        <div class="field">
          <label>カテゴリ（任意）</label>
          <input type="text" id="cep-category" placeholder="例: 日焼け止め、スキンケア">
        </div>
        <div class="field">
          <label>追加の依頼内容（任意）</label>
          <input type="text" id="cep-prompt" placeholder="例: 自然体でアクティブな雰囲気で">
        </div>
      </div>
    </div>

    <!-- ── ブランドモード ── -->
    <div id="panel-brand" class="mode-panel">
      <div class="row">
        <div class="field">
          <label>ブランド名</label>
          <input type="text" id="brand-name" placeholder="例: SunGuard、LUNASOLEIL">
        </div>
        <div class="field">
          <label>ブランドトーン（任意）</label>
          <input type="text" id="brand-tone" placeholder="例: 親しみやすく、自然体で前向き">
        </div>
      </div>

      <div class="row">
        <div class="field">
          <label>価値観・キーワード（任意、カンマ区切り）</label>
          <input type="text" id="brand-values" placeholder="例: 自由, 自然, 信頼, アクティブ">
        </div>
        <div class="field">
          <label>ターゲット（任意）</label>
          <input type="text" id="brand-target" placeholder="例: アウトドア好きな30代女性">
        </div>
      </div>

      <label>広告の依頼内容</label>
      <textarea id="brand-prompt"
        placeholder="例: 夏の日焼け止め広告コピーを作って"
      ></textarea>
    </div>

    <!-- ── 検索ボリューム（全モード共通） ── -->
    <label>検索ボリュームキーワード（任意）</label>
    <div id="kv-list"></div>
    <button class="add-kv-btn" onclick="addKvRow()">＋ キーワードを追加</button>

    <!-- ── 生成案数 & ボタン ── -->
    <div class="row" style="margin-top:16px">
      <div class="field">
        <label>生成案数</label>
        <input type="text" id="n-cases" value="8" style="width:70px">
      </div>
      <div class="field" style="flex:3">
        <button class="btn-primary" id="gen-btn" onclick="generate()">生成する</button>
      </div>
    </div>
  </div>

  <!-- ── 結果エリア ─────────────────────────────────── -->
  <div id="results"></div>

  <!-- ── LLMO分析カード ──────────────────────────────── -->
  <div class="card" id="llmo-card" style="display:none">
    <div class="step-lbl">📊 LLMO指標分析 — 言及率・引用率・参照率</div>

    <div class="row">
      <div class="field">
        <label>ブランド名（任意）</label>
        <input type="text" id="an-brand" placeholder="例: SunGuard">
      </div>
      <div class="field">
        <label>CEPプロンプト（任意）</label>
        <input type="text" id="an-cep" placeholder="例: アウトドア向け日焼け止めを探している">
      </div>
      <div class="field">
        <label>キーワード（任意）</label>
        <input type="text" id="an-kw" placeholder="例: 日焼け止め UVケア">
      </div>
    </div>

    <button class="btn-secondary" onclick="analyzeLlmo()">LLMO指標を分析する</button>

    <div id="llmo-results"></div>
  </div>

</div><!-- /container -->

<script>
// ─────────────────────────────────────────────────────────────────
// 状態
// ─────────────────────────────────────────────────────────────────
let currentMode   = "keyword";
let currentBest   = "";
let originalPrompt= "";
let refineStep    = 0;
let allVariants   = [];          // 現在の全バリアントテキスト
const REFINE_STEPS = [8, 5, 3];

// ─────────────────────────────────────────────────────────────────
// タブ切替
// ─────────────────────────────────────────────────────────────────
function switchTab(mode, el) {
  currentMode = mode;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  el.classList.add("active");
  document.querySelectorAll(".mode-panel").forEach(p => p.classList.remove("active"));
  document.getElementById("panel-" + mode).classList.add("active");
}

// ─────────────────────────────────────────────────────────────────
// 検索ボリュームキーワード 追加/削除
// ─────────────────────────────────────────────────────────────────
function addKvRow(kw, vol) {
  const list = document.getElementById("kv-list");
  const row  = document.createElement("div");
  row.className = "kv-row";

  const kwIn  = document.createElement("input");
  kwIn.type        = "text";
  kwIn.placeholder = "キーワード";
  if (kw) kwIn.value = kw;

  const volIn = document.createElement("input");
  volIn.type        = "text";
  volIn.className   = "vol";
  volIn.placeholder = "月間ボリューム（例:5000）";
  if (vol) volIn.value = vol;

  const del = document.createElement("button");
  del.className = "kv-del";
  del.textContent = "×";
  del.onclick = () => list.removeChild(row);

  row.appendChild(kwIn);
  row.appendChild(volIn);
  row.appendChild(del);
  list.appendChild(row);
}

function getKvData() {
  const rows = document.querySelectorAll("#kv-list .kv-row");
  const result = [];
  rows.forEach(row => {
    const inputs = row.querySelectorAll("input");
    const kw  = inputs[0].value.trim();
    const vol = parseInt(inputs[1].value.trim()) || 0;
    if (kw && vol > 0) result.push({keyword: kw, volume: vol});
  });
  return result;
}

// ─────────────────────────────────────────────────────────────────
// リクエスト本体を組み立てる
// ─────────────────────────────────────────────────────────────────
function buildRequestBody(n) {
  const kvs = getKvData();
  const base = {
    mode: currentMode,
    n: n,
    keyword_volumes: kvs.length ? kvs : null,
  };

  if (currentMode === "keyword") {
    const prompt  = document.getElementById("kw-prompt").value.trim();
    const keyword = document.getElementById("kw-keyword").value.trim();
    return Object.assign(base, { prompt, keyword: keyword || null });
  }

  if (currentMode === "cep") {
    const scenario = document.getElementById("cep-scenario").value.trim();
    const category = document.getElementById("cep-category").value.trim();
    const extra    = document.getElementById("cep-prompt").value.trim();
    return Object.assign(base, {
      prompt: extra || " ",
      cep: { scenario, category: category || null },
    });
  }

  if (currentMode === "brand") {
    const name   = document.getElementById("brand-name").value.trim();
    const tone   = document.getElementById("brand-tone").value.trim();
    const valStr = document.getElementById("brand-values").value.trim();
    const target = document.getElementById("brand-target").value.trim();
    const prompt = document.getElementById("brand-prompt").value.trim();
    const values = valStr ? valStr.split(/[,，、]/).map(s=>s.trim()).filter(Boolean) : null;
    return Object.assign(base, {
      prompt: prompt || " ",
      brand: {
        name,
        tone:   tone   || null,
        values: values || null,
        target: target || null,
      },
    });
  }
  return base;
}

function validateInput() {
  if (currentMode === "keyword") {
    if (!document.getElementById("kw-prompt").value.trim())
      return "依頼内容を入力してください";
  } else if (currentMode === "cep") {
    if (!document.getElementById("cep-scenario").value.trim())
      return "シチュエーション（CEP）を入力してください";
  } else if (currentMode === "brand") {
    if (!document.getElementById("brand-name").value.trim())
      return "ブランド名を入力してください";
    if (!document.getElementById("brand-prompt").value.trim())
      return "広告の依頼内容を入力してください";
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────
// 生成
// ─────────────────────────────────────────────────────────────────
async function generate() {
  const err = validateInput();
  if (err) return alert(err);

  const n = parseInt(document.getElementById("n-cases").value) || 8;
  refineStep    = 0;
  originalPrompt = (currentMode === "keyword")
    ? document.getElementById("kw-prompt").value.trim()
    : (currentMode === "cep")
      ? document.getElementById("cep-scenario").value.trim()
      : document.getElementById("brand-prompt").value.trim();

  const results = document.getElementById("results");
  results.innerHTML = "";
  results.appendChild(makeLoading("生成中..."));
  document.getElementById("gen-btn").disabled = true;

  try {
    const body = buildRequestBody(n);
    const res  = await fetch("/v1/ad/generate", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "API エラー");

    currentBest = data.best;
    allVariants = data.variants.map(v => v.text);

    results.innerHTML = "";
    const modeLabel = {keyword:"キーワード起点", cep:"CEP起点", brand:"ブランド起点"}[currentMode] || currentMode;
    showResults(data, `ステップ1 — ${n}案（探索）/ ${modeLabel}`, true);

    // LLMO分析カード表示
    document.getElementById("llmo-card").style.display = "block";
    // ブランド名を自動セット
    if (currentMode === "brand") {
      const bn = document.getElementById("brand-name").value.trim();
      if (bn) document.getElementById("an-brand").value = bn;
    }
  } catch(e) {
    results.innerHTML = "";
    const d = document.createElement("div");
    d.className = "card";
    d.textContent = "エラー: " + e.message;
    results.appendChild(d);
  }
  document.getElementById("gen-btn").disabled = false;
}

// ─────────────────────────────────────────────────────────────────
// ブラッシュアップ
// ─────────────────────────────────────────────────────────────────
async function refine() {
  const dirEl = document.getElementById("direction-input");
  if (!dirEl) return;
  const direction = dirEl.value.trim();
  if (!direction) return alert("改善方向を入力してください");

  refineStep++;
  const n   = REFINE_STEPS[Math.min(refineStep, REFINE_STEPS.length - 1)];
  const kw  = currentMode === "keyword"
    ? document.getElementById("kw-keyword").value.trim() : "";

  const loading = makeLoading(`${n}案にブラッシュアップ中...`);
  document.getElementById("results").appendChild(loading);
  const btn = document.getElementById("refine-btn");
  if (btn) btn.disabled = true;

  try {
    const res = await fetch("/v1/ad/refine", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        base: currentBest, direction, original_prompt: originalPrompt,
        n, keyword: kw || null,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "API エラー");

    currentBest  = data.best;
    allVariants  = data.variants.map(v => v.text);
    loading.remove();
    showResults(data, `ステップ${refineStep+1} — ${n}案（ブラッシュアップ）`, refineStep < 2);
  } catch(e) {
    loading.remove();
    alert("エラー: " + e.message);
  }
}

// ─────────────────────────────────────────────────────────────────
// 結果カードを表示
// ─────────────────────────────────────────────────────────────────
function showResults(data, stepLabel, showRefine) {
  const card = document.createElement("div");
  card.className = "card";

  const lbl = document.createElement("div");
  lbl.className   = "step-lbl";
  lbl.textContent = stepLabel;
  card.appendChild(lbl);

  data.variants.forEach((v, i) => {
    const vEl = document.createElement("div");
    vEl.className   = "variant" + (i === 0 ? " selected" : "");
    vEl.dataset.text = v.text;

    if (i === 0) {
      const badge = document.createElement("span");
      badge.className   = "badge";
      badge.textContent = "🏆 最高案";
      vEl.appendChild(badge);
      const br = document.createElement("br");
      vEl.appendChild(br);
    }

    const textEl = document.createElement("div");
    textEl.className   = "v-text";
    textEl.textContent = "「" + v.text + "」";
    vEl.appendChild(textEl);

    // 総合スコアバー
    const scoreRow = document.createElement("div");
    scoreRow.className = "score-row";
    const bg = document.createElement("div"); bg.className = "bar-bg";
    const fill = document.createElement("div"); fill.className = "bar-fill";
    fill.style.width = (v.score * 100) + "%";
    bg.appendChild(fill);
    const num = document.createElement("div"); num.className = "score-num";
    num.textContent = v.score.toFixed(2);
    scoreRow.appendChild(bg);
    scoreRow.appendChild(num);
    vEl.appendChild(scoreRow);

    // スコア詳細（折りたたみ）
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "▶ スコア詳細";
    details.appendChild(summary);

    const bd = v.score_breakdown;
    const cats = [
      ["簡潔さ",          bd.conciseness],
      ["感情インパクト",   bd.emotional_impact],
      ["ターゲット適合",   bd.target_fit],
      ["記憶しやすさ",     bd.memorability],
      ["行動促進力",       bd.action_power],
      ["検索ポテンシャル", bd.search_potential],
    ];
    const grid = document.createElement("div"); grid.className = "sb-grid";
    cats.forEach(([lbl, val]) => {
      const lblEl = document.createElement("span"); lblEl.className = "sb-lbl"; lblEl.textContent = lbl;
      const barBg = document.createElement("div");  barBg.className = "sb-bar-bg";
      const barFl = document.createElement("div");  barFl.className = "sb-bar-fill" + (val >= 0.7 ? " hi" : "");
      barFl.style.width = (val * 100) + "%";
      barBg.appendChild(barFl);
      const valEl = document.createElement("span"); valEl.className = "sb-val"; valEl.textContent = val.toFixed(2);
      grid.appendChild(lblEl); grid.appendChild(barBg); grid.appendChild(valEl);
    });
    details.appendChild(grid);

    // マッチしたキーワード
    if (bd.matched_keywords && bd.matched_keywords.length > 0) {
      const tags = document.createElement("div"); tags.className = "kw-tags";
      bd.matched_keywords.forEach(kw => {
        const tag = document.createElement("span"); tag.className = "kw-tag";
        tag.textContent = kw;
        tags.appendChild(tag);
      });
      details.appendChild(tags);
    }

    vEl.appendChild(details);
    vEl.addEventListener("click", () => selectVariant(vEl));
    card.appendChild(vEl);
  });

  // ブラッシュアップエリア
  if (showRefine) {
    const nextN = REFINE_STEPS[Math.min(refineStep + 1, REFINE_STEPS.length - 1)];
    const ra = document.createElement("div"); ra.style.marginTop = "14px";

    const lbl2 = document.createElement("label");
    lbl2.textContent = "選んだ案をどう変えたいですか？";
    ra.appendChild(lbl2);

    const inp = document.createElement("input");
    inp.type = "text"; inp.id = "direction-input";
    inp.placeholder = "例: もっと短く、若者向けに、インパクトを強く";
    ra.appendChild(inp);

    const btn = document.createElement("button");
    btn.id = "refine-btn"; btn.className = "btn-primary";
    btn.textContent = `ブラッシュアップする（${nextN}案）`;
    btn.addEventListener("click", refine);
    ra.appendChild(btn);
    card.appendChild(ra);
  } else {
    const done = document.createElement("p");
    done.style.cssText = "text-align:center;color:#666;margin-top:14px;font-size:13px";
    done.textContent = "✅ 最終案の選定が完了しました";
    card.appendChild(done);
  }

  document.getElementById("results").appendChild(card);
  card.scrollIntoView({behavior:"smooth"});
}

// ─────────────────────────────────────────────────────────────────
// LLMO分析
// ─────────────────────────────────────────────────────────────────
async function analyzeLlmo() {
  if (!allVariants.length) return alert("先に広告コピーを生成してください");

  const brandName  = document.getElementById("an-brand").value.trim() || null;
  const cepPrompt  = document.getElementById("an-cep").value.trim();
  const kwStr      = document.getElementById("an-kw").value.trim();
  const keywords   = kwStr ? kwStr.split(/\s+/).filter(Boolean) : null;
  const cepPrompts = cepPrompt ? [cepPrompt] : null;

  const resEl = document.getElementById("llmo-results");
  resEl.innerHTML = "";
  resEl.appendChild(makeLoading("LLMO指標を計算中..."));

  try {
    const res = await fetch("/v1/llmo/analyze", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ texts: allVariants, brand_name: brandName, cep_prompts: cepPrompts, keywords }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "API エラー");

    resEl.innerHTML = "";
    renderLlmoResult(data, resEl);
  } catch(e) {
    resEl.innerHTML = "";
    resEl.textContent = "エラー: " + e.message;
  }
}

function renderLlmoResult(data, container) {
  const m = data.metrics;

  // ゲージ3つ
  const gaugeRow = document.createElement("div");
  gaugeRow.className = "gauge-row";

  const gauges = [
    { val: m.mention_potential,   lbl: "言及ポテンシャル",    sub: m.mention_rate_estimate + " 想定" },
    { val: m.citation_potential,  lbl: "引用ポテンシャル",    sub: "情報源として引用される可能性" },
    { val: m.reference_potential, lbl: "参照ポテンシャル",    sub: "AIクローラーが参照する可能性" },
  ];
  gauges.forEach(g => {
    const el = document.createElement("div"); el.className = "gauge";
    const pct = Math.round(g.val * 100);
    const valEl  = document.createElement("div"); valEl.className = "g-val";
    valEl.textContent = pct + "%";
    if (pct >= 70)      valEl.style.color = "#2e7d32";
    else if (pct >= 45) valEl.style.color = "#e65100";
    else                valEl.style.color = "#c62828";

    const lblEl = document.createElement("div"); lblEl.className = "g-lbl"; lblEl.textContent = g.lbl;
    const subEl = document.createElement("div"); subEl.className = "g-sub"; subEl.textContent = g.sub;
    el.appendChild(valEl); el.appendChild(lblEl); el.appendChild(subEl);
    gaugeRow.appendChild(el);
  });
  container.appendChild(gaugeRow);

  // 分析コメント
  const note = document.createElement("p");
  note.style.cssText = "font-size:13px;color:#444;margin-top:12px;padding:10px 14px;background:#f0f0f0;border-radius:8px";
  note.textContent = data.analysis_note;
  container.appendChild(note);

  // 推奨事項
  if (m.recommendations && m.recommendations.length) {
    const recList = document.createElement("div"); recList.className = "rec-list";
    m.recommendations.forEach(r => {
      const item = document.createElement("div"); item.className = "rec-item";
      item.textContent = "💡 " + r;
      recList.appendChild(item);
    });
    container.appendChild(recList);
  }

  // 最高案コピー
  if (data.top_copy) {
    const best = document.createElement("div");
    best.style.cssText = "margin-top:12px;padding:12px 16px;background:#eef5ff;border-radius:10px;font-size:14px";
    const lbl = document.createElement("span");
    lbl.style.cssText = "font-size:11px;font-weight:700;color:#0071e3;display:block;margin-bottom:4px";
    lbl.textContent = "🏆 LLMO最高スコア案";
    const txt = document.createElement("span"); txt.textContent = "「" + data.top_copy + "」";
    best.appendChild(lbl); best.appendChild(txt);
    container.appendChild(best);
  }
}

// ─────────────────────────────────────────────────────────────────
// ユーティリティ
// ─────────────────────────────────────────────────────────────────
function selectVariant(el) {
  el.closest(".card").querySelectorAll(".variant").forEach(v => v.classList.remove("selected"));
  el.classList.add("selected");
  currentBest = el.dataset.text;
}

function makeLoading(msg) {
  const el = document.createElement("div"); el.className = "card loading";
  const sp = document.createElement("div"); sp.className = "spinner";
  const tx = document.createElement("span"); tx.textContent = msg;
  el.appendChild(sp); el.appendChild(tx);
  return el;
}
</script>
</body>
</html>""")
