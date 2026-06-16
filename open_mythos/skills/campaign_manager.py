"""
Sprint 60 — 広告キャンペーン管理

CEP（Customer Entry Point）→ 広告コピー生成 → 評価 → キャンペーン管理の
全フローをワークフロー化した統合モジュール。

オブジェクト:
  CampaignStatus   : キャンペーン状態 (Draft/Active/Paused/Completed/Archived)
  AdChannel        : 広告チャネル (Search/Social/Display/Email/Video)
  AdObjective      : 広告目標 (Awareness/Consideration/Conversion/Retention)
  AdCopy           : 広告コピー 1 件 (headline/body/cta/channel)
  CampaignBudget   : 予算設定 (total/daily/currency)
  Campaign         : キャンペーン本体 (cep_ids / copies / budget / status)
  CampaignStore    : キャンペーン CRUD ストア
  CopyGenerator    : CEP シナリオ → 広告コピー生成（ルールベース）
  CampaignEvaluator: コピー品質スコアリング (長さ / CTA / CEP 整合)
  CampaignWorkflow : CEP → コピー生成 → 評価 → キャンペーン登録フロー
  CampaignReportEngine: レポート生成 (Markdown / JSON)

設計方針:
  - LLM 非依存のルールベース生成（外部 API 差し替え可能な抽象 I/F）
  - 全データはメモリ内保持（永続化は外部 DB に委ねる設計）
  - CEP との連携: llmo_dashboard.CepEntry の scenario 文を入力として使用
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum 層
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CampaignStatus(str, Enum):
    DRAFT      = "draft"
    ACTIVE     = "active"
    PAUSED     = "paused"
    COMPLETED  = "completed"
    ARCHIVED   = "archived"


class AdChannel(str, Enum):
    SEARCH   = "search"    # Google/Bing 検索広告
    SOCIAL   = "social"    # SNS 広告 (Twitter/Instagram 等)
    DISPLAY  = "display"   # ディスプレイ広告
    EMAIL    = "email"     # メール広告
    VIDEO    = "video"     # 動画広告


class AdObjective(str, Enum):
    AWARENESS      = "awareness"      # 認知拡大
    CONSIDERATION  = "consideration"  # 比較検討
    CONVERSION     = "conversion"     # コンバージョン
    RETENTION      = "retention"      # 顧客維持


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AdCopy:
    """広告コピー 1 件"""
    id:        str
    headline:  str                      # 見出し (最大 30 文字推奨)
    body:      str                      # 本文
    cta:       str                      # Call to Action
    channel:   AdChannel                = AdChannel.SEARCH
    objective: AdObjective              = AdObjective.AWARENESS
    tags:      List[str]                = field(default_factory=list)
    score:     float                    = 0.0   # CampaignEvaluator が設定
    created_at: float                   = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":         self.id,
            "headline":   self.headline,
            "body":       self.body,
            "cta":        self.cta,
            "channel":    self.channel.value,
            "objective":  self.objective.value,
            "tags":       self.tags,
            "score":      round(self.score, 4),
            "created_at": self.created_at,
        }


@dataclass
class CampaignBudget:
    """予算設定"""
    total:    float               # 総予算
    daily:    float               # 日次予算上限
    currency: str = "JPY"         # 通貨 (ISO 4217)
    spent:    float = 0.0         # 消化額（外部連携で更新）

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.spent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total":     self.total,
            "daily":     self.daily,
            "currency":  self.currency,
            "spent":     self.spent,
            "remaining": self.remaining,
        }


@dataclass
class Campaign:
    """広告キャンペーン本体"""
    id:          str
    name:        str
    objective:   AdObjective
    budget:      CampaignBudget
    status:      CampaignStatus          = CampaignStatus.DRAFT
    cep_ids:     List[str]               = field(default_factory=list)
    copies:      List[AdCopy]            = field(default_factory=list)
    channels:    List[AdChannel]         = field(default_factory=list)
    tags:        List[str]               = field(default_factory=list)
    description: str                     = ""
    created_at:  float                   = field(default_factory=time.time)
    updated_at:  float                   = field(default_factory=time.time)

    # ---- 状態遷移 ----

    def activate(self) -> None:
        if self.status not in (CampaignStatus.DRAFT, CampaignStatus.PAUSED):
            raise ValueError(f"Cannot activate from status={self.status.value}")
        self.status = CampaignStatus.ACTIVE
        self.updated_at = time.time()

    def pause(self) -> None:
        if self.status != CampaignStatus.ACTIVE:
            raise ValueError(f"Cannot pause from status={self.status.value}")
        self.status = CampaignStatus.PAUSED
        self.updated_at = time.time()

    def complete(self) -> None:
        if self.status not in (CampaignStatus.ACTIVE, CampaignStatus.PAUSED):
            raise ValueError(f"Cannot complete from status={self.status.value}")
        self.status = CampaignStatus.COMPLETED
        self.updated_at = time.time()

    def archive(self) -> None:
        if self.status == CampaignStatus.ARCHIVED:
            raise ValueError("Already archived")
        self.status = CampaignStatus.ARCHIVED
        self.updated_at = time.time()

    # ---- 集計 ----

    @property
    def best_copy(self) -> Optional[AdCopy]:
        if not self.copies:
            return None
        return max(self.copies, key=lambda c: c.score)

    @property
    def avg_score(self) -> float:
        if not self.copies:
            return 0.0
        return sum(c.score for c in self.copies) / len(self.copies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":          self.id,
            "name":        self.name,
            "objective":   self.objective.value,
            "budget":      self.budget.to_dict(),
            "status":      self.status.value,
            "cep_ids":     self.cep_ids,
            "copies":      [c.to_dict() for c in self.copies],
            "channels":    [ch.value for ch in self.channels],
            "tags":        self.tags,
            "description": self.description,
            "avg_score":   round(self.avg_score, 4),
            "created_at":  self.created_at,
            "updated_at":  self.updated_at,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CampaignStore:
    """キャンペーン CRUD ストア（インメモリ）"""

    def __init__(self) -> None:
        self._campaigns: Dict[str, Campaign] = {}

    # ---- CRUD ----

    def add(self, campaign: Campaign) -> Campaign:
        self._campaigns[campaign.id] = campaign
        return campaign

    def get(self, campaign_id: str) -> Optional[Campaign]:
        return self._campaigns.get(campaign_id)

    def list_all(self) -> List[Campaign]:
        return list(self._campaigns.values())

    def list_by_status(self, status: CampaignStatus) -> List[Campaign]:
        return [c for c in self._campaigns.values() if c.status == status]

    def delete(self, campaign_id: str) -> bool:
        if campaign_id in self._campaigns:
            del self._campaigns[campaign_id]
            return True
        return False

    def count(self) -> int:
        return len(self._campaigns)

    # ---- 検索 ----

    def find_by_objective(self, objective: AdObjective) -> List[Campaign]:
        return [c for c in self._campaigns.values() if c.objective == objective]

    def find_by_cep(self, cep_id: str) -> List[Campaign]:
        return [c for c in self._campaigns.values() if cep_id in c.cep_ids]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CopyGenerator — CEP シナリオ → 広告コピー生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CTA_MAP: Dict[AdObjective, List[str]] = {
    AdObjective.AWARENESS:     ["今すぐチェック", "詳しく見る", "ブランドを知る"],
    AdObjective.CONSIDERATION: ["無料で比較する", "口コミを見る", "デモを申し込む"],
    AdObjective.CONVERSION:    ["今すぐ購入", "無料で試す", "お申し込みはこちら"],
    AdObjective.RETENTION:     ["会員ページへ", "特典を確認する", "続きを読む"],
}

_HEADLINE_TEMPLATES: Dict[AdObjective, List[str]] = {
    AdObjective.AWARENESS:     [
        "{brand}が選ばれる理由",
        "{brand}とは？特徴を解説",
        "{brand}で{benefit}を実現",
    ],
    AdObjective.CONSIDERATION: [
        "{brand}と他社を徹底比較",
        "{brand}の口コミ・評判まとめ",
        "{brand}の導入事例はこちら",
    ],
    AdObjective.CONVERSION: [
        "{brand}を今すぐ無料で試す",
        "【期間限定】{brand}が{discount}OFF",
        "{brand}で{benefit}を今日から",
    ],
    AdObjective.RETENTION: [
        "{brand}会員限定の特典",
        "{brand}から大切なお知らせ",
        "あなたへの{brand}おすすめ情報",
    ],
}


class CopyGenerator:
    """
    CEP シナリオと目標から広告コピーを生成する（ルールベース）。

    外部 LLM API に差し替える場合は `generate_from_scenario` をオーバーライドする。

    Sprint 64B: use_morphology=True で日本語形態素解析 (JaMorphologicalAnalyzer) を
    使った名詞抽出によりタグ精度を向上させる。デフォルトは後方互換のため正規表現抽出。
    """

    def __init__(
        self,
        brand: str = "OpenMythos",
        use_morphology: bool = False,
        analyzer: Optional[Any] = None,
    ) -> None:
        self.brand = brand
        self.use_morphology = use_morphology
        self._analyzer = analyzer
        if use_morphology and self._analyzer is None:
            # 遅延 import（形態素解析は任意機能）
            from open_mythos.tokenizer_ja import JaMorphologicalAnalyzer
            self._analyzer = JaMorphologicalAnalyzer()

    def generate_from_scenario(
        self,
        scenario: str,
        objective: AdObjective = AdObjective.AWARENESS,
        channel: AdChannel = AdChannel.SEARCH,
        extra: Optional[Dict[str, str]] = None,
    ) -> AdCopy:
        """
        シナリオ文から AdCopy を 1 件生成する。

        Args:
            scenario  : CEP シナリオ文 (例: "〇〇が困ったとき")
            objective : 広告目標
            channel   : 配信チャネル
            extra     : テンプレート変数の上書き {"benefit": ..., "discount": ...}
        """
        vars_: Dict[str, str] = {
            "brand":    self.brand,
            "benefit":  self._extract_benefit(scenario),
            "discount": "30%",
        }
        if extra:
            vars_.update(extra)

        templates = _HEADLINE_TEMPLATES.get(objective, _HEADLINE_TEMPLATES[AdObjective.AWARENESS])
        # シナリオ長さに応じてテンプレートを選択（決定論的）
        idx = len(scenario) % len(templates)
        headline_raw = templates[idx]
        headline = self._fill(headline_raw, vars_)[:40]  # 40 文字上限

        body = self._build_body(scenario, objective, vars_)
        cta_list = _CTA_MAP.get(objective, ["詳しく見る"])
        cta = cta_list[len(scenario) % len(cta_list)]

        # タグ: シナリオからキーワード抽出
        tags = self._extract_tags(scenario)

        return AdCopy(
            id=str(uuid.uuid4()),
            headline=headline,
            body=body,
            cta=cta,
            channel=channel,
            objective=objective,
            tags=tags,
        )

    def generate_batch(
        self,
        scenario: str,
        objective: AdObjective = AdObjective.AWARENESS,
        channels: Optional[List[AdChannel]] = None,
        extra: Optional[Dict[str, str]] = None,
    ) -> List[AdCopy]:
        """複数チャネル向けにコピーをバッチ生成する"""
        if channels is None:
            channels = [AdChannel.SEARCH, AdChannel.SOCIAL]
        return [
            self.generate_from_scenario(scenario, objective, ch, extra)
            for ch in channels
        ]

    # ---- 内部ヘルパー ----

    def _extract_benefit(self, scenario: str) -> str:
        """シナリオから便益キーワードを抽出（簡易実装）"""
        # 「〜したい」「〜できる」前後を取る
        m = re.search(r"([^\s]{2,8})(したい|できる|解決|改善)", scenario)
        if m:
            return m.group(1) + "改善"
        # fallback: 最初の名詞相当（2〜6文字）
        m2 = re.search(r"[぀-鿿]{2,6}", scenario)
        return m2.group(0) if m2 else "課題"

    def _fill(self, template: str, vars_: Dict[str, str]) -> str:
        for k, v in vars_.items():
            template = template.replace(f"{{{k}}}", v)
        return template

    def _build_body(
        self,
        scenario: str,
        objective: AdObjective,
        vars_: Dict[str, str],
    ) -> str:
        brand = vars_["brand"]
        benefit = vars_["benefit"]
        if objective == AdObjective.AWARENESS:
            return f"{scenario}に対応するなら {brand} にお任せ。{benefit}をサポートします。"
        if objective == AdObjective.CONSIDERATION:
            return f"{scenario} — {brand} が選ばれる 3 つの理由を今すぐ確認。"
        if objective == AdObjective.CONVERSION:
            return f"今なら {brand} を無料トライアルで体験。{benefit}を実感してください。"
        return f"{brand} 会員の皆さまへ。{scenario} に役立つ情報をお届けします。"

    def _extract_tags(self, scenario: str) -> List[str]:
        """
        シナリオからタグを抽出する。

        use_morphology=True なら形態素解析の名詞抽出を使い、より精度の高い
        キーワードを得る。それ以外は従来の正規表現ベース抽出。
        """
        if self.use_morphology and self._analyzer is not None:
            nouns = self._analyzer.extract_nouns(scenario)
            # 2 文字以上の名詞のみ採用（助詞混入を避ける）
            nouns = [n for n in nouns if len(n) >= 2]
            return list(dict.fromkeys(nouns))[:5]  # 重複除去・上限 5

        tokens = re.findall(r"[぀-鿿]{3,}", scenario)
        return list(dict.fromkeys(tokens))[:5]  # 重複除去・上限 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignEvaluator — コピー品質スコアリング
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EvalResult:
    """評価結果"""
    copy_id:        str
    total_score:    float           # 0.0 〜 1.0
    headline_score: float
    body_score:     float
    cta_score:      float
    alignment_score: float          # CEP シナリオとの整合
    notes:          List[str]       = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "copy_id":         self.copy_id,
            "total_score":     round(self.total_score, 4),
            "headline_score":  round(self.headline_score, 4),
            "body_score":      round(self.body_score, 4),
            "cta_score":       round(self.cta_score, 4),
            "alignment_score": round(self.alignment_score, 4),
            "notes":           self.notes,
        }


class CampaignEvaluator:
    """広告コピーの品質をルールベースでスコアリングする"""

    # 見出し長さ基準 (文字)
    HEADLINE_MIN = 8
    HEADLINE_MAX = 30

    # 本文長さ基準 (文字)
    BODY_MIN = 20
    BODY_MAX = 150

    def evaluate(
        self,
        copy: AdCopy,
        scenario: Optional[str] = None,
    ) -> EvalResult:
        """AdCopy を評価して EvalResult を返す"""
        headline_score, h_notes = self._score_headline(copy.headline)
        body_score, b_notes = self._score_body(copy.body)
        cta_score, c_notes = self._score_cta(copy.cta)
        alignment_score, a_notes = self._score_alignment(copy, scenario)

        total = (
            headline_score  * 0.30
            + body_score    * 0.30
            + cta_score     * 0.20
            + alignment_score * 0.20
        )

        result = EvalResult(
            copy_id=copy.id,
            total_score=total,
            headline_score=headline_score,
            body_score=body_score,
            cta_score=cta_score,
            alignment_score=alignment_score,
            notes=h_notes + b_notes + c_notes + a_notes,
        )
        copy.score = total  # インプレース更新
        return result

    def evaluate_batch(
        self,
        copies: List[AdCopy],
        scenario: Optional[str] = None,
    ) -> List[EvalResult]:
        return [self.evaluate(c, scenario) for c in copies]

    # ---- 内部スコアリング ----

    def _score_headline(self, headline: str) -> Tuple[float, List[str]]:
        notes: List[str] = []
        n = len(headline)
        if n < self.HEADLINE_MIN:
            notes.append(f"見出しが短すぎます ({n}文字 < {self.HEADLINE_MIN})")
            return 0.4, notes
        if n > self.HEADLINE_MAX:
            notes.append(f"見出しが長すぎます ({n}文字 > {self.HEADLINE_MAX})")
            return 0.6, notes
        score = 1.0
        # 数字・感嘆符があるとボーナス
        if re.search(r"[0-9０-９]", headline):
            score = min(1.0, score + 0.05)
        if "！" in headline or "?" in headline or "？" in headline:
            score = min(1.0, score + 0.05)
        return score, notes

    def _score_body(self, body: str) -> Tuple[float, List[str]]:
        notes: List[str] = []
        n = len(body)
        if n < self.BODY_MIN:
            notes.append(f"本文が短すぎます ({n}文字 < {self.BODY_MIN})")
            return 0.3, notes
        if n > self.BODY_MAX:
            notes.append(f"本文が長すぎます ({n}文字 > {self.BODY_MAX})")
            return 0.7, notes
        # 改行があると可読性ボーナス
        score = 0.8
        if "\n" in body or "。" in body:
            score = 0.9
        return score, notes

    def _score_cta(self, cta: str) -> Tuple[float, List[str]]:
        notes: List[str] = []
        if not cta:
            notes.append("CTA が空です")
            return 0.0, notes
        # 動詞・行動喚起語を含むか
        action_words = ["する", "見る", "試す", "申し込む", "チェック", "確認", "購入", "登録"]
        has_action = any(w in cta for w in action_words)
        if has_action:
            return 1.0, notes
        notes.append("CTA に行動喚起語が見当たりません")
        return 0.6, notes

    def _score_alignment(
        self,
        copy: AdCopy,
        scenario: Optional[str],
    ) -> Tuple[float, List[str]]:
        notes: List[str] = []
        if not scenario:
            return 0.8, notes  # シナリオなしは中立スコア
        # コピータグとシナリオの共通トークン数で簡易スコア
        scenario_tokens = set(re.findall(r"[぀-鿿]{2,}", scenario))
        copy_text = copy.headline + copy.body + copy.cta
        copy_tokens = set(re.findall(r"[぀-鿿]{2,}", copy_text))
        overlap = scenario_tokens & copy_tokens
        if not scenario_tokens:
            return 0.8, notes
        ratio = len(overlap) / len(scenario_tokens)
        score = min(1.0, 0.5 + ratio)
        if ratio < 0.2:
            notes.append("コピーとシナリオのキーワード重複が少ないです")
        return score, notes


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignWorkflow — CEP → コピー生成 → 評価 → 登録フロー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class WorkflowResult:
    """ワークフロー実行結果"""
    campaign:     Campaign
    eval_results: List[EvalResult]
    best_copy:    Optional[AdCopy]
    total_copies: int
    avg_score:    float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign":     self.campaign.to_dict(),
            "eval_results": [e.to_dict() for e in self.eval_results],
            "best_copy":    self.best_copy.to_dict() if self.best_copy else None,
            "total_copies": self.total_copies,
            "avg_score":    round(self.avg_score, 4),
        }


class CampaignWorkflow:
    """
    CEP シナリオ → コピー生成 → 評価 → キャンペーン登録の
    フルパイプラインを実行する。

    Usage:
        wf = CampaignWorkflow()
        result = wf.run(
            name="夏キャンペーン",
            scenario="暑い夏に冷たい飲み物を探している",
            objective=AdObjective.CONVERSION,
            budget=CampaignBudget(total=100000, daily=5000),
            channels=[AdChannel.SEARCH, AdChannel.SOCIAL],
        )
    """

    def __init__(
        self,
        store: Optional[CampaignStore] = None,
        generator: Optional[CopyGenerator] = None,
        evaluator: Optional[CampaignEvaluator] = None,
        brand: str = "OpenMythos",
    ) -> None:
        self.store     = store     or CampaignStore()
        self.generator = generator or CopyGenerator(brand=brand)
        self.evaluator = evaluator or CampaignEvaluator()

    def run(
        self,
        name: str,
        scenario: str,
        objective: AdObjective = AdObjective.AWARENESS,
        budget: Optional[CampaignBudget] = None,
        channels: Optional[List[AdChannel]] = None,
        cep_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        extra: Optional[Dict[str, str]] = None,
    ) -> WorkflowResult:
        """
        ワークフローを実行してキャンペーンを生成・ストアに登録する。

        Returns:
            WorkflowResult (campaign / eval_results / best_copy / avg_score)
        """
        if budget is None:
            budget = CampaignBudget(total=100_000, daily=10_000)
        if channels is None:
            channels = [AdChannel.SEARCH, AdChannel.SOCIAL]

        # 1) コピー生成
        copies = self.generator.generate_batch(
            scenario=scenario,
            objective=objective,
            channels=channels,
            extra=extra,
        )

        # 2) 評価
        eval_results = self.evaluator.evaluate_batch(copies, scenario=scenario)

        # 3) キャンペーン組み立て
        campaign = Campaign(
            id=str(uuid.uuid4()),
            name=name,
            objective=objective,
            budget=budget,
            copies=copies,
            channels=channels,
            cep_ids=cep_ids or [],
            tags=tags or [],
            description=description,
        )

        # 4) ストアへ登録
        self.store.add(campaign)

        return WorkflowResult(
            campaign=campaign,
            eval_results=eval_results,
            best_copy=campaign.best_copy,
            total_copies=len(copies),
            avg_score=campaign.avg_score,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignReportEngine — レポート生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CampaignReportEngine:
    """キャンペーン一覧・詳細レポートを生成する"""

    def __init__(self, store: CampaignStore) -> None:
        self._store = store

    def summary_json(self) -> Dict[str, Any]:
        """全キャンペーンのサマリーを JSON 形式で返す"""
        campaigns = self._store.list_all()
        status_counts: Dict[str, int] = {}
        for c in campaigns:
            status_counts[c.status.value] = status_counts.get(c.status.value, 0) + 1

        total_budget = sum(c.budget.total for c in campaigns)
        total_spent  = sum(c.budget.spent for c in campaigns)
        avg_score = (
            sum(c.avg_score for c in campaigns) / len(campaigns)
            if campaigns else 0.0
        )

        return {
            "total_campaigns": len(campaigns),
            "status_counts":   status_counts,
            "total_budget":    total_budget,
            "total_spent":     total_spent,
            "avg_copy_score":  round(avg_score, 4),
            "campaigns":       [c.to_dict() for c in campaigns],
        }

    def campaign_markdown(self, campaign_id: str) -> str:
        """単一キャンペーンの Markdown レポートを生成する"""
        c = self._store.get(campaign_id)
        if c is None:
            return f"# エラー\nキャンペーン ID `{campaign_id}` が見つかりません。"

        lines = [
            f"# キャンペーン: {c.name}",
            "",
            f"**ID**: `{c.id}`  ",
            f"**目標**: {c.objective.value}  ",
            f"**状態**: {c.status.value}  ",
            f"**チャネル**: {', '.join(ch.value for ch in c.channels)}  ",
            "",
            "## 予算",
            f"| 項目 | 金額 ({c.budget.currency}) |",
            "|------|------|",
            f"| 総予算 | {c.budget.total:,.0f} |",
            f"| 日次上限 | {c.budget.daily:,.0f} |",
            f"| 消化額 | {c.budget.spent:,.0f} |",
            f"| 残額 | {c.budget.remaining:,.0f} |",
            "",
            "## 広告コピー",
        ]

        if c.copies:
            lines += [
                f"平均スコア: **{c.avg_score:.4f}**",
                "",
                "| # | 見出し | CTA | チャネル | スコア |",
                "|---|--------|-----|---------|--------|",
            ]
            for i, cp in enumerate(c.copies, 1):
                lines.append(
                    f"| {i} | {cp.headline} | {cp.cta} | {cp.channel.value} | {cp.score:.4f} |"
                )
        else:
            lines.append("*コピーがありません*")

        if c.cep_ids:
            lines += ["", "## 関連 CEP ID", *[f"- `{cid}`" for cid in c.cep_ids]]

        lines += ["", f"*生成日時: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c.created_at))}*"]
        return "\n".join(lines)
