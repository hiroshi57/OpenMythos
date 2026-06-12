"""
Sprint 60-A — 広告キャンペーン管理

CEP（Customer Entry Point）→ 広告コピー生成 → 評価 の全フローを
ワークフローとして統合した広告キャンペーン管理フレームワーク。

オブジェクト:
  CampaignStatus   : キャンペーンのライフサイクル状態
  AdFormat         : 広告フォーマット分類
  CopyRequest      : コピー生成リクエスト（ブリーフ＋ターゲット）
  AdCopy           : 生成された広告コピー（ヘッドライン/ボディ/CTA）
  CampaignMetrics  : インプレッション / CTR / CVR / ROAS 等の実績指標
  CampaignEntry    : キャンペーン全体の定義と状態管理
  CampaignStore    : キャンペーン CRUD ストア
  CopyGenerator    : CopyRequest → AdCopy 生成エンジン
  CampaignEvaluator: コピー品質スコアリング（LLMO 軸 + CTR 予測）
  CampaignWorkflow : CEP → Copy → Evaluate の統合オーケストレーター
  CampaignReportEngine: Markdown / JSON レポート生成

設計方針:
  - 外部 LLM 依存なし（テンプレートベース生成）
  - CampaignEvaluator は AdEvaluator-互換のスコアリングを独自実装
  - 全データはメモリ内保持（永続化は外部 DB に委ねる設計）
"""
from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum 層
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CampaignStatus(str, Enum):
    DRAFT     = "draft"
    ACTIVE    = "active"
    PAUSED    = "paused"
    COMPLETED = "completed"
    ARCHIVED  = "archived"


class AdFormat(str, Enum):
    BANNER  = "banner"
    TEXT    = "text"
    VIDEO   = "video"
    SOCIAL  = "social"
    EMAIL   = "email"
    SEARCH  = "search"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# コピー生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CopyRequest:
    """広告コピー生成リクエスト"""
    campaign_id:  str
    brief:        str                    # コピーのブリーフ（訴求ポイント）
    format:       AdFormat              = AdFormat.TEXT
    target:       Optional[str]         = None   # ターゲットペルソナ
    keywords:     List[str]             = field(default_factory=list)
    tone:         str                   = "friendly"  # friendly/formal/urgent/inspiring
    max_chars:    int                   = 120

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "brief":       self.brief,
            "format":      self.format.value,
            "target":      self.target,
            "keywords":    self.keywords,
            "tone":        self.tone,
            "max_chars":   self.max_chars,
        }


@dataclass
class AdCopy:
    """生成された広告コピー"""
    id:          str
    request_id:  str
    headline:    str
    body:        str
    cta:         str                    # Call-to-action テキスト
    format:      AdFormat
    score:       float                  = 0.0
    created_at:  int                    = field(default_factory=lambda: int(time.time()))

    def full_text(self) -> str:
        return f"{self.headline} {self.body} {self.cta}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":         self.id,
            "request_id": self.request_id,
            "headline":   self.headline,
            "body":       self.body,
            "cta":        self.cta,
            "format":     self.format.value,
            "score":      self.score,
            "created_at": self.created_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# キャンペーン指標 / エントリー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CampaignMetrics:
    """広告キャンペーン実績指標"""
    impressions: int   = 0
    clicks:      int   = 0
    conversions: int   = 0
    spend:       float = 0.0   # 広告費 (円/ドル)
    revenue:     float = 0.0   # 売上

    @property
    def ctr(self) -> float:
        """Click-Through Rate"""
        return self.clicks / self.impressions if self.impressions > 0 else 0.0

    @property
    def cvr(self) -> float:
        """Conversion Rate (clicks → conversions)"""
        return self.conversions / self.clicks if self.clicks > 0 else 0.0

    @property
    def roas(self) -> float:
        """Return on Ad Spend"""
        return self.revenue / self.spend if self.spend > 0 else 0.0

    @property
    def cpa(self) -> float:
        """Cost Per Acquisition"""
        return self.spend / self.conversions if self.conversions > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "impressions": self.impressions,
            "clicks":      self.clicks,
            "conversions": self.conversions,
            "spend":       self.spend,
            "revenue":     self.revenue,
            "ctr":         round(self.ctr, 4),
            "cvr":         round(self.cvr, 4),
            "roas":        round(self.roas, 4),
            "cpa":         round(self.cpa, 2),
        }

    def update(self, impressions: int = 0, clicks: int = 0,
               conversions: int = 0, spend: float = 0.0, revenue: float = 0.0) -> None:
        self.impressions += impressions
        self.clicks      += clicks
        self.conversions += conversions
        self.spend       += spend
        self.revenue     += revenue


@dataclass
class CampaignEntry:
    """広告キャンペーン定義"""
    id:          str
    name:        str
    status:      CampaignStatus           = CampaignStatus.DRAFT
    cep_ids:     List[str]                = field(default_factory=list)
    copies:      List[AdCopy]             = field(default_factory=list)
    metrics:     CampaignMetrics          = field(default_factory=CampaignMetrics)
    budget:      float                    = 0.0
    description: str                      = ""
    created_at:  int                      = field(default_factory=lambda: int(time.time()))
    updated_at:  int                      = field(default_factory=lambda: int(time.time()))

    def best_copy(self) -> Optional[AdCopy]:
        """最高スコアのコピーを返す"""
        return max(self.copies, key=lambda c: c.score) if self.copies else None

    def activate(self) -> None:
        self.status = CampaignStatus.ACTIVE
        self.updated_at = int(time.time())

    def pause(self) -> None:
        self.status = CampaignStatus.PAUSED
        self.updated_at = int(time.time())

    def complete(self) -> None:
        self.status = CampaignStatus.COMPLETED
        self.updated_at = int(time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":          self.id,
            "name":        self.name,
            "status":      self.status.value,
            "cep_ids":     self.cep_ids,
            "copies":      [c.to_dict() for c in self.copies],
            "metrics":     self.metrics.to_dict(),
            "budget":      self.budget,
            "description": self.description,
            "created_at":  self.created_at,
            "updated_at":  self.updated_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ストア
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CampaignStore:
    """キャンペーン CRUD ストア（メモリ内）"""

    def __init__(self) -> None:
        self._campaigns: Dict[str, CampaignEntry] = {}

    def create(self, name: str, description: str = "", budget: float = 0.0,
               cep_ids: Optional[List[str]] = None) -> CampaignEntry:
        cid = str(uuid.uuid4())
        entry = CampaignEntry(
            id=cid, name=name, description=description,
            budget=budget, cep_ids=cep_ids or [],
        )
        self._campaigns[cid] = entry
        return entry

    def get(self, campaign_id: str) -> Optional[CampaignEntry]:
        return self._campaigns.get(campaign_id)

    def list_all(self, status: Optional[CampaignStatus] = None) -> List[CampaignEntry]:
        campaigns = list(self._campaigns.values())
        if status is not None:
            campaigns = [c for c in campaigns if c.status == status]
        return campaigns

    def update_status(self, campaign_id: str, status: CampaignStatus) -> Optional[CampaignEntry]:
        entry = self._campaigns.get(campaign_id)
        if entry is None:
            return None
        entry.status = status
        entry.updated_at = int(time.time())
        return entry

    def delete(self, campaign_id: str) -> bool:
        if campaign_id in self._campaigns:
            del self._campaigns[campaign_id]
            return True
        return False

    def record_metrics(self, campaign_id: str, impressions: int = 0, clicks: int = 0,
                       conversions: int = 0, spend: float = 0.0, revenue: float = 0.0) -> Optional[CampaignMetrics]:
        entry = self._campaigns.get(campaign_id)
        if entry is None:
            return None
        entry.metrics.update(impressions=impressions, clicks=clicks,
                             conversions=conversions, spend=spend, revenue=revenue)
        return entry.metrics

    def __len__(self) -> int:
        return len(self._campaigns)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# コピー生成エンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_TONE_PREFIXES: Dict[str, List[str]] = {
    "friendly":   ["一緒に", "あなたに", "みんなの", "楽しく"],
    "formal":     ["ご提案", "サービス", "信頼の", "実績の"],
    "urgent":     ["今すぐ", "期間限定", "残りわずか", "今だけ"],
    "inspiring":  ["夢を叶える", "可能性を広げる", "未来へ", "挑戦する"],
}

_FORMAT_CTAS: Dict[str, List[str]] = {
    AdFormat.BANNER.value: ["今すぐ確認", "詳しくはこちら", "クリックして詳細を見る"],
    AdFormat.TEXT.value:   ["詳細を見る", "お問い合わせ", "無料で試す"],
    AdFormat.VIDEO.value:  ["動画を見る", "再生する", "続きを見る"],
    AdFormat.SOCIAL.value: ["シェアする", "フォローする", "いいね！"],
    AdFormat.EMAIL.value:  ["登録する", "購読する", "詳細をメールで受け取る"],
    AdFormat.SEARCH.value: ["公式サイトへ", "今すぐ検索", "詳しくはこちら"],
}


class CopyGenerator:
    """CopyRequest → AdCopy 生成エンジン（テンプレートベース）"""

    def generate(self, req: CopyRequest, n: int = 1) -> List[AdCopy]:
        """n 件のコピーを生成して返す"""
        results: List[AdCopy] = []
        prefixes = _TONE_PREFIXES.get(req.tone, _TONE_PREFIXES["friendly"])
        ctas = _FORMAT_CTAS.get(req.format.value, _FORMAT_CTAS[AdFormat.TEXT.value])

        for i in range(n):
            prefix = prefixes[i % len(prefixes)]
            cta    = ctas[i % len(ctas)]
            headline = self._make_headline(prefix, req.brief, req.keywords)
            body     = self._make_body(req.brief, req.target, req.keywords)
            copy = AdCopy(
                id=str(uuid.uuid4()),
                request_id=req.campaign_id,
                headline=headline[:req.max_chars],
                body=body[:req.max_chars * 2],
                cta=cta,
                format=req.format,
            )
            results.append(copy)
        return results

    def _make_headline(self, prefix: str, brief: str, keywords: List[str]) -> str:
        kw_part = keywords[0] if keywords else ""
        brief_short = brief[:30] if len(brief) > 30 else brief
        return f"{prefix}{brief_short}{' — ' + kw_part if kw_part else ''}"

    def _make_body(self, brief: str, target: Optional[str], keywords: List[str]) -> str:
        kw_str = "、".join(keywords[:3]) if keywords else ""
        target_str = f"{target}向け" if target else ""
        kw_clause = f"キーワード: {kw_str}。" if kw_str else ""
        return f"{target_str}{brief} {kw_clause}".strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# コピー評価エンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CopyScore:
    """コピー評価スコア"""
    copy_id:        str
    clarity:        float = 0.0   # 明確さ (0〜1)
    relevance:      float = 0.0   # キーワード関連度 (0〜1)
    cta_strength:   float = 0.0   # CTA の強さ (0〜1)
    length_penalty: float = 0.0   # 長さペナルティ (0〜1, 高いほど良い)
    overall:        float = 0.0   # 総合スコア (0〜1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "copy_id":        self.copy_id,
            "clarity":        round(self.clarity, 4),
            "relevance":      round(self.relevance, 4),
            "cta_strength":   round(self.cta_strength, 4),
            "length_penalty": round(self.length_penalty, 4),
            "overall":        round(self.overall, 4),
        }


class CampaignEvaluator:
    """コピー品質スコアリング（LLMO 軸 + CTR 予測）"""

    # CTA 強度マップ（高いほどアクション喚起力が強い）
    _STRONG_CTA_WORDS = {"今すぐ", "無料", "期間限定", "限定", "試す", "始める", "今だけ"}
    _MEDIUM_CTA_WORDS = {"詳細", "確認", "見る", "登録", "申込", "お問い合わせ"}

    def score(self, copy: AdCopy, keywords: Optional[List[str]] = None) -> CopyScore:
        text = copy.full_text()
        keywords = keywords or []

        clarity        = self._clarity(copy.headline, copy.body)
        relevance      = self._relevance(text, keywords)
        cta_strength   = self._cta_strength(copy.cta)
        length_penalty = self._length_penalty(copy.headline, copy.body)

        overall = (
            clarity        * 0.30 +
            relevance      * 0.25 +
            cta_strength   * 0.30 +
            length_penalty * 0.15
        )
        return CopyScore(
            copy_id=copy.id,
            clarity=clarity,
            relevance=relevance,
            cta_strength=cta_strength,
            length_penalty=length_penalty,
            overall=min(overall, 1.0),
        )

    def score_batch(self, copies: List[AdCopy],
                    keywords: Optional[List[str]] = None) -> List[CopyScore]:
        return [self.score(c, keywords) for c in copies]

    def predict_ctr(self, copy: AdCopy) -> float:
        """CTR 予測値を返す (0〜1の疑似スコア)"""
        base = 0.02
        cta_boost = 0.01 if any(w in copy.cta for w in self._STRONG_CTA_WORDS) else 0.0
        len_boost  = 0.005 if 10 <= len(copy.headline) <= 40 else 0.0
        return min(base + cta_boost + len_boost, 0.15)

    # ── 内部ヘルパー ────────────────────────────────────────────────

    def _clarity(self, headline: str, body: str) -> float:
        """文章の明確さ: 短い文・句読点の適切な使用を評価"""
        score = 0.5
        # 句読点があると読みやすい
        if any(p in body for p in ["。", "、", "！", "？"]):
            score += 0.2
        # ヘッドラインが短すぎず長すぎない
        hlen = len(headline)
        if 5 <= hlen <= 30:
            score += 0.2
        elif hlen > 60:
            score -= 0.1
        # 同じ単語の繰り返しが少ない
        words = re.findall(r'[^\s]+', headline + " " + body)
        if words:
            unique_ratio = len(set(words)) / len(words)
            score += 0.1 * unique_ratio
        return max(0.0, min(score, 1.0))

    def _relevance(self, text: str, keywords: List[str]) -> float:
        """キーワード関連度: テキスト中のキーワードヒット率"""
        if not keywords:
            return 0.5
        hits = sum(1 for kw in keywords if kw in text)
        return hits / len(keywords)

    def _cta_strength(self, cta: str) -> float:
        """CTA の強さ: アクション喚起ワードの有無"""
        if any(w in cta for w in self._STRONG_CTA_WORDS):
            return 0.9
        if any(w in cta for w in self._MEDIUM_CTA_WORDS):
            return 0.6
        return 0.3

    def _length_penalty(self, headline: str, body: str) -> float:
        """長さペナルティ: 適切な文字数かを評価"""
        h_ok = 5  <= len(headline) <= 40
        b_ok = 10 <= len(body)     <= 200
        if h_ok and b_ok:
            return 1.0
        elif h_ok or b_ok:
            return 0.6
        return 0.3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ワークフロー (CEP → Copy → Evaluate)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class WorkflowResult:
    """CampaignWorkflow の実行結果"""
    campaign_id:  str
    copies:       List[AdCopy]
    scores:       List[CopyScore]
    best_copy_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id":  self.campaign_id,
            "copies":       [c.to_dict() for c in self.copies],
            "scores":       [s.to_dict() for s in self.scores],
            "best_copy_id": self.best_copy_id,
        }


class CampaignWorkflow:
    """CEP → Copy → Evaluate の統合オーケストレーター"""

    def __init__(self,
                 store:     Optional[CampaignStore]    = None,
                 generator: Optional[CopyGenerator]    = None,
                 evaluator: Optional[CampaignEvaluator] = None) -> None:
        self.store     = store     if store     is not None else CampaignStore()
        self.generator = generator if generator is not None else CopyGenerator()
        self.evaluator = evaluator if evaluator is not None else CampaignEvaluator()

    def run(self, campaign_id: str, brief: str,
            format: AdFormat = AdFormat.TEXT,
            target: Optional[str] = None,
            keywords: Optional[List[str]] = None,
            tone: str = "friendly",
            n_copies: int = 3) -> WorkflowResult:
        """ワークフロー実行: コピー生成 → 評価 → 最良コピー選択"""
        keywords = keywords or []
        req = CopyRequest(
            campaign_id=campaign_id,
            brief=brief,
            format=format,
            target=target,
            keywords=keywords,
            tone=tone,
        )
        copies = self.generator.generate(req, n=n_copies)
        scores = self.evaluator.score_batch(copies, keywords=keywords)

        # スコアをコピーに反映
        score_map = {s.copy_id: s.overall for s in scores}
        for copy in copies:
            copy.score = score_map.get(copy.id, 0.0)

        # キャンペーンにコピーを追加
        entry = self.store.get(campaign_id)
        if entry is not None:
            entry.copies.extend(copies)
            entry.updated_at = int(time.time())

        best = max(copies, key=lambda c: c.score) if copies else None
        return WorkflowResult(
            campaign_id=campaign_id,
            copies=copies,
            scores=scores,
            best_copy_id=best.id if best else None,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# レポートエンジン
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CampaignReportEngine:
    """キャンペーンレポート生成（Markdown / JSON）"""

    def to_markdown(self, entry: CampaignEntry) -> str:
        m = entry.metrics
        best = entry.best_copy()
        lines = [
            f"# キャンペーンレポート: {entry.name}",
            f"",
            f"**ステータス**: {entry.status.value}  |  **予算**: ¥{entry.budget:,.0f}",
            f"",
            f"## 実績指標",
            f"| 指標 | 値 |",
            f"|------|-----|",
            f"| インプレッション | {m.impressions:,} |",
            f"| クリック | {m.clicks:,} |",
            f"| コンバージョン | {m.conversions:,} |",
            f"| 広告費 | ¥{m.spend:,.0f} |",
            f"| 売上 | ¥{m.revenue:,.0f} |",
            f"| CTR | {m.ctr:.2%} |",
            f"| CVR | {m.cvr:.2%} |",
            f"| ROAS | {m.roas:.2f}x |",
            f"| CPA | ¥{m.cpa:,.0f} |",
            f"",
        ]
        if best:
            lines += [
                f"## 最優秀コピー (score: {best.score:.3f})",
                f"**ヘッドライン**: {best.headline}",
                f"**ボディ**: {best.body}",
                f"**CTA**: {best.cta}",
                f"",
            ]
        lines += [
            f"## 全コピー一覧",
            f"| # | ヘッドライン | スコア |",
            f"|---|------------|--------|",
        ]
        for i, c in enumerate(entry.copies, 1):
            lines.append(f"| {i} | {c.headline[:30]} | {c.score:.3f} |")
        return "\n".join(lines)

    def to_json(self, entry: CampaignEntry) -> Dict[str, Any]:
        return {
            "report_type": "campaign",
            "campaign":    entry.to_dict(),
            "summary": {
                "total_copies":    len(entry.copies),
                "best_score":      max((c.score for c in entry.copies), default=0.0),
                "avg_score":       (sum(c.score for c in entry.copies) / len(entry.copies))
                                    if entry.copies else 0.0,
                "budget_utilized": (entry.metrics.spend / entry.budget)
                                    if entry.budget > 0 else 0.0,
            },
        }
