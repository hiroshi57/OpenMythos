"""
Sprint 70C — 自然言語クエリ (NLQ) エージェント

「先週のCTRは？」「来週のclicks予測」などの自然言語を
インテント分類して OpenMythos API 相当の操作を実行する。

オブジェクト:
  NLQIntent   : インテント種別 Enum
  NLQQuery    : パース済みクエリ
  NLQResult   : 実行結果
  NLQParser   : テキスト → NLQQuery (ルールベース、LLM 不要)
  NLQExecutor : NLQQuery → NLQResult

設計方針:
  - 外部 LLM 依存なし (regex キーワードマッチング)
  - 日本語・英語の両方に対応
  - intent ごとに analytics / forecast / alert / budget の各ストアを参照
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from open_mythos.skills.campaign_analytics import CampaignAnalyticsStore, KpiCalculator
from open_mythos.skills.time_series import ForecastStore
from open_mythos.skills.anomaly_detector import AlertStore

_kpi_calculator = KpiCalculator()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NLQIntent(str, Enum):
    CAMPAIGN_KPI = "campaign_kpi"
    FORECAST     = "forecast"
    ANOMALY      = "anomaly"
    BUDGET       = "budget"
    UNKNOWN      = "unknown"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class NLQQuery:
    """パース済み自然言語クエリ"""
    raw:         str
    intent:      NLQIntent
    campaign_id: Optional[str] = None
    metric:      Optional[str] = None
    params:      Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw":         self.raw,
            "intent":      self.intent.value,
            "campaign_id": self.campaign_id,
            "metric":      self.metric,
            "params":      self.params,
        }


@dataclass
class NLQResult:
    """クエリ実行結果"""
    query:   NLQQuery
    intent:  NLQIntent
    data:    Any
    message: str
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query":   self.query.to_dict(),
            "intent":  self.intent.value,
            "data":    self.data,
            "message": self.message,
            "success": self.success,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# パーサー（ルールベース）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# インテント判定キーワード (優先度順で定義)
_INTENT_RULES: List[tuple[NLQIntent, List[str]]] = [
    (NLQIntent.FORECAST, [
        "予測", "predict", "forecast", "来週", "来月", "将来",
        "next week", "next month", "future",
    ]),
    (NLQIntent.ANOMALY, [
        "異常", "アラート", "anomaly", "alert", "spike", "drop",
        "スパイク", "急増", "急減",
    ]),
    (NLQIntent.BUDGET, [
        "最適化", "optimize", "allocation", "配分",
    ]),
    (NLQIntent.CAMPAIGN_KPI, [
        "CTR", "ctr", "CVR", "cvr", "CPC", "cpc", "ROAS", "roas",
        "クリック", "click", "インプレッション", "impression",
        "消費", "spend", "売上", "revenue", "KPI", "kpi",
        "予算", "budget",
        "確認", "教えて", "表示", "show", "get", "data",
    ]),
]

# メトリクス名マッピング
_METRIC_RULES: List[tuple[str, List[str]]] = [
    ("ctr",         ["CTR", "ctr", "クリック率"]),
    ("cvr",         ["CVR", "cvr", "コンバージョン率", "conversion"]),
    ("clicks",      ["クリック", "click", "clicks"]),
    ("impressions", ["インプレッション", "impression", "impressions"]),
    ("spend",       ["消費", "spend", "コスト", "cost"]),
    ("revenue",     ["売上", "revenue", "収益"]),
    ("roas",        ["ROAS", "roas"]),
    ("cpa",         ["CPA", "cpa"]),
]

# campaign_id 抽出パターン: "campaign <id>" or "キャンペーン <id>"
_CAMPAIGN_ID_PATTERN = re.compile(
    r"(?:campaign|キャンペーン)\s+([\w\-]+)",
    re.IGNORECASE,
)


class NLQParser:
    """自然言語テキストを NLQQuery にパースする（ルールベース）"""

    def parse(self, text: str) -> NLQQuery:
        text_lower = text.lower()

        intent = self._detect_intent(text, text_lower)
        metric = self._detect_metric(text, text_lower)
        campaign_id = self._detect_campaign_id(text)

        return NLQQuery(
            raw=text,
            intent=intent,
            campaign_id=campaign_id,
            metric=metric,
        )

    # ── 内部 ───────────────────────────────────────────────────

    @staticmethod
    def _detect_intent(text: str, text_lower: str) -> NLQIntent:
        for intent, keywords in _INTENT_RULES:
            for kw in keywords:
                if kw.lower() in text_lower:
                    return intent
        return NLQIntent.UNKNOWN

    @staticmethod
    def _detect_metric(text: str, text_lower: str) -> Optional[str]:
        for metric_name, keywords in _METRIC_RULES:
            for kw in keywords:
                if kw.lower() in text_lower:
                    return metric_name
        return None

    @staticmethod
    def _detect_campaign_id(text: str) -> Optional[str]:
        m = _CAMPAIGN_ID_PATTERN.search(text)
        return m.group(1) if m else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# エグゼキューター
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class NLQExecutor:
    """NLQQuery を実行して NLQResult を返す"""

    def __init__(
        self,
        analytics_store: CampaignAnalyticsStore,
        forecast_store: ForecastStore,
        alert_store: AlertStore,
    ) -> None:
        self._analytics = analytics_store
        self._forecast  = forecast_store
        self._alerts    = alert_store

    def execute(self, query: NLQQuery) -> NLQResult:
        if query.intent == NLQIntent.CAMPAIGN_KPI:
            return self._exec_kpi(query)
        if query.intent == NLQIntent.FORECAST:
            return self._exec_forecast(query)
        if query.intent == NLQIntent.ANOMALY:
            return self._exec_anomaly(query)
        if query.intent == NLQIntent.BUDGET:
            return self._exec_budget(query)
        # UNKNOWN
        return NLQResult(
            query=query, intent=NLQIntent.UNKNOWN,
            data=None, message="クエリの意図を判定できませんでした。",
            success=False,
        )

    # ── KPI ────────────────────────────────────────────────────

    def _exec_kpi(self, query: NLQQuery) -> NLQResult:
        if query.campaign_id:
            metrics = self._analytics.get(query.campaign_id)
            if metrics is None:
                return NLQResult(
                    query=query, intent=NLQIntent.CAMPAIGN_KPI,
                    data=None,
                    message=f"キャンペーン '{query.campaign_id}' のデータがありません。",
                    success=False,
                )
            kpi = _kpi_calculator.compute(metrics).to_dict()
            if query.metric and query.metric in kpi:
                data: Any = {query.metric: kpi[query.metric]}
                msg = f"{query.campaign_id} の {query.metric}: {kpi[query.metric]:.4f}"
            else:
                data = kpi
                msg = f"{query.campaign_id} の KPI サマリー"
        else:
            # 全キャンペーン サマリー
            all_campaigns = self._analytics.list_campaigns()
            if not all_campaigns:
                data = {}
                msg = "キャンペーンデータがありません。"
            else:
                data = {
                    cid: _kpi_calculator.compute(self._analytics.get(cid)).to_dict()
                    for cid in all_campaigns
                    if self._analytics.get(cid) is not None
                }
                msg = f"{len(data)} キャンペーンの KPI サマリー"

        return NLQResult(
            query=query, intent=NLQIntent.CAMPAIGN_KPI,
            data=data, message=msg, success=True,
        )

    # ── Forecast ───────────────────────────────────────────────

    def _exec_forecast(self, query: NLQQuery) -> NLQResult:
        metric = query.metric or "clicks"
        if query.campaign_id:
            result = self._forecast.latest(query.campaign_id, metric)
            if result is None:
                return NLQResult(
                    query=query, intent=NLQIntent.FORECAST,
                    data=None,
                    message=f"'{query.campaign_id}/{metric}' の予測データがありません。",
                    success=False,
                )
            data = result.to_dict()
            msg = f"{query.campaign_id} の {metric} 予測 (horizon={result.horizon})"
        else:
            # 全キャンペーンの最新予測を返す
            all_campaigns = self._analytics.list_campaigns()
            data = {}
            for cid in all_campaigns:
                r = self._forecast.latest(cid, metric)
                if r:
                    data[cid] = r.to_dict()
            msg = f"{len(data)} キャンペーンの {metric} 予測"

        return NLQResult(
            query=query, intent=NLQIntent.FORECAST,
            data=data, message=msg, success=True,
        )

    # ── Anomaly ────────────────────────────────────────────────

    def _exec_anomaly(self, query: NLQQuery) -> NLQResult:
        if query.campaign_id:
            alerts = self._alerts.list_by_campaign(query.campaign_id)
        else:
            alerts = self._alerts.list_all()
        data = [a.to_dict() for a in alerts]
        msg = f"{len(data)} 件のアラート"
        return NLQResult(
            query=query, intent=NLQIntent.ANOMALY,
            data=data, message=msg, success=True,
        )

    # ── Budget ─────────────────────────────────────────────────

    def _exec_budget(self, query: NLQQuery) -> NLQResult:
        # budget_optimizer の依存を避け、analytics から簡易推奨を生成
        all_campaigns = self._analytics.list_campaigns()
        if not all_campaigns:
            return NLQResult(
                query=query, intent=NLQIntent.BUDGET,
                data=None,
                message="予算最適化に必要なキャンペーンデータがありません。",
                success=True,
            )
        roas_map: Dict[str, float] = {}
        for cid in all_campaigns:
            m = self._analytics.get(cid)
            if m is not None:
                kpi = _kpi_calculator.compute(m).to_dict()
                roas_map[cid] = kpi.get("roas", 0.0)

        total = sum(roas_map.values())
        if total > 0:
            allocation = {cid: round(roas / total, 4) for cid, roas in roas_map.items()}
        else:
            n = len(roas_map)
            allocation = {cid: round(1.0 / n, 4) for cid in roas_map}

        return NLQResult(
            query=query, intent=NLQIntent.BUDGET,
            data={"allocation": allocation, "strategy": "roas_weighted"},
            message=f"{len(allocation)} キャンペーンへの ROAS 比例配分を推奨",
            success=True,
        )
