"""
Sprint 63B — キャンペーン分析ダッシュボード

広告キャンペーンの KPI（インプレッション・クリック・コンバージョン・コスト・
売上）を時系列で蓄積・集計し、派生指標（CTR/CVR/CPC/CPA/ROAS）を算出する
分析ダッシュボード。

オブジェクト:
  MetricType         : 指標種別 (Impressions/Clicks/Conversions/Spend/Revenue)
  MetricPoint        : 時系列データ点 1 件 (timestamp + 各指標値)
  CampaignMetrics    : キャンペーン単位の累積指標 + 派生 KPI
  AnalyticsSnapshot  : 特定時点のスナップショット
  CampaignAnalyticsStore: キャンペーン別メトリクスのストア
  KpiCalculator      : 派生 KPI (CTR/CVR/CPC/CPA/ROAS) の計算
  TrendAnalyzer      : 時系列トレンド分析 (前期比/移動平均)
  CampaignAnalyticsDashboard: 全キャンペーン横断の集計
  AnalyticsReportEngine: レポート生成 (Markdown / JSON)

設計方針:
  - 外部依存なし
  - 派生 KPI はゼロ除算を安全に 0.0 として扱う
  - 時系列は MetricPoint のリストとして保持（永続化は外部 DB に委ねる）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MetricType(str, Enum):
    IMPRESSIONS = "impressions"
    CLICKS      = "clicks"
    CONVERSIONS = "conversions"
    SPEND       = "spend"
    REVENUE     = "revenue"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データモデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class MetricPoint:
    """時系列データ点 1 件"""
    impressions: int   = 0
    clicks:      int   = 0
    conversions: int   = 0
    spend:       float = 0.0
    revenue:     float = 0.0
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "impressions": self.impressions,
            "clicks":      self.clicks,
            "conversions": self.conversions,
            "spend":       self.spend,
            "revenue":     self.revenue,
            "timestamp":   self.timestamp,
        }


def _safe_div(a: float, b: float) -> float:
    """ゼロ除算を 0.0 として扱う安全な除算"""
    return a / b if b else 0.0


@dataclass
class CampaignMetrics:
    """キャンペーン単位の累積指標 + 時系列"""
    campaign_id: str
    points:      List[MetricPoint] = field(default_factory=list)

    def add_point(self, point: MetricPoint) -> None:
        self.points.append(point)

    def record(
        self,
        impressions: int = 0,
        clicks: int = 0,
        conversions: int = 0,
        spend: float = 0.0,
        revenue: float = 0.0,
    ) -> MetricPoint:
        """新しいデータ点を記録する"""
        p = MetricPoint(
            impressions=impressions,
            clicks=clicks,
            conversions=conversions,
            spend=spend,
            revenue=revenue,
        )
        self.points.append(p)
        return p

    # ---- 累積集計 ----

    @property
    def total_impressions(self) -> int:
        return sum(p.impressions for p in self.points)

    @property
    def total_clicks(self) -> int:
        return sum(p.clicks for p in self.points)

    @property
    def total_conversions(self) -> int:
        return sum(p.conversions for p in self.points)

    @property
    def total_spend(self) -> float:
        return sum(p.spend for p in self.points)

    @property
    def total_revenue(self) -> float:
        return sum(p.revenue for p in self.points)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id":       self.campaign_id,
            "point_count":       len(self.points),
            "total_impressions": self.total_impressions,
            "total_clicks":      self.total_clicks,
            "total_conversions": self.total_conversions,
            "total_spend":       round(self.total_spend, 2),
            "total_revenue":     round(self.total_revenue, 2),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KpiCalculator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class KpiSet:
    """派生 KPI セット"""
    ctr:  float   # Click-Through Rate (clicks / impressions)
    cvr:  float   # Conversion Rate (conversions / clicks)
    cpc:  float   # Cost Per Click (spend / clicks)
    cpa:  float   # Cost Per Acquisition (spend / conversions)
    cpm:  float   # Cost Per Mille (spend / impressions * 1000)
    roas: float   # Return On Ad Spend (revenue / spend)
    roi:  float   # Return On Investment ((revenue - spend) / spend)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ctr":  round(self.ctr, 6),
            "cvr":  round(self.cvr, 6),
            "cpc":  round(self.cpc, 4),
            "cpa":  round(self.cpa, 4),
            "cpm":  round(self.cpm, 4),
            "roas": round(self.roas, 4),
            "roi":  round(self.roi, 4),
        }


class KpiCalculator:
    """累積指標から派生 KPI を計算する"""

    def compute(self, metrics: CampaignMetrics) -> KpiSet:
        imp = metrics.total_impressions
        clk = metrics.total_clicks
        cv  = metrics.total_conversions
        spend = metrics.total_spend
        rev = metrics.total_revenue

        return KpiSet(
            ctr=_safe_div(clk, imp),
            cvr=_safe_div(cv, clk),
            cpc=_safe_div(spend, clk),
            cpa=_safe_div(spend, cv),
            cpm=_safe_div(spend, imp) * 1000,
            roas=_safe_div(rev, spend),
            roi=_safe_div(rev - spend, spend),
        )

    def compute_from_point(self, point: MetricPoint) -> KpiSet:
        """単一データ点から KPI を計算する"""
        return KpiSet(
            ctr=_safe_div(point.clicks, point.impressions),
            cvr=_safe_div(point.conversions, point.clicks),
            cpc=_safe_div(point.spend, point.clicks),
            cpa=_safe_div(point.spend, point.conversions),
            cpm=_safe_div(point.spend, point.impressions) * 1000,
            roas=_safe_div(point.revenue, point.spend),
            roi=_safe_div(point.revenue - point.spend, point.spend),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TrendAnalyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrendResult:
    """トレンド分析結果"""
    metric:        str
    current:       float
    previous:      float
    delta:         float       # current - previous
    delta_pct:     float       # (current - previous) / previous
    direction:     str         # "up" / "down" / "flat"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric":    self.metric,
            "current":   round(self.current, 4),
            "previous":  round(self.previous, 4),
            "delta":     round(self.delta, 4),
            "delta_pct": round(self.delta_pct, 4),
            "direction": self.direction,
        }


class TrendAnalyzer:
    """時系列トレンド分析"""

    def _extract(self, point: MetricPoint, metric: str) -> float:
        return float(getattr(point, metric, 0))

    def period_over_period(
        self, metrics: CampaignMetrics, metric: str = "impressions"
    ) -> Optional[TrendResult]:
        """
        直近 2 データ点を比較してトレンドを返す。
        データ点が 2 未満なら None。
        """
        if len(metrics.points) < 2:
            return None
        current = self._extract(metrics.points[-1], metric)
        previous = self._extract(metrics.points[-2], metric)
        delta = current - previous
        delta_pct = _safe_div(delta, previous)

        if delta > 0:
            direction = "up"
        elif delta < 0:
            direction = "down"
        else:
            direction = "flat"

        return TrendResult(
            metric=metric,
            current=current,
            previous=previous,
            delta=delta,
            delta_pct=delta_pct,
            direction=direction,
        )

    def moving_average(
        self, metrics: CampaignMetrics, metric: str = "impressions", window: int = 3
    ) -> List[float]:
        """移動平均を計算する"""
        values = [self._extract(p, metric) for p in metrics.points]
        if window <= 0:
            raise ValueError("window must be positive")
        result: List[float] = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            chunk = values[start:i + 1]
            result.append(sum(chunk) / len(chunk))
        return result

    def total_series(
        self, metrics: CampaignMetrics, metric: str = "impressions"
    ) -> List[float]:
        """指定指標の時系列を返す"""
        return [self._extract(p, metric) for p in metrics.points]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignAnalyticsStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CampaignAnalyticsStore:
    """キャンペーン別メトリクスストア"""

    def __init__(self) -> None:
        self._metrics: Dict[str, CampaignMetrics] = {}

    def get_or_create(self, campaign_id: str) -> CampaignMetrics:
        if campaign_id not in self._metrics:
            self._metrics[campaign_id] = CampaignMetrics(campaign_id=campaign_id)
        return self._metrics[campaign_id]

    def get(self, campaign_id: str) -> Optional[CampaignMetrics]:
        return self._metrics.get(campaign_id)

    def record(
        self,
        campaign_id: str,
        impressions: int = 0,
        clicks: int = 0,
        conversions: int = 0,
        spend: float = 0.0,
        revenue: float = 0.0,
    ) -> MetricPoint:
        """指定キャンペーンにデータ点を記録する"""
        m = self.get_or_create(campaign_id)
        return m.record(
            impressions=impressions,
            clicks=clicks,
            conversions=conversions,
            spend=spend,
            revenue=revenue,
        )

    def list_campaigns(self) -> List[str]:
        return list(self._metrics.keys())

    def count(self) -> int:
        return len(self._metrics)

    def delete(self, campaign_id: str) -> bool:
        if campaign_id in self._metrics:
            del self._metrics[campaign_id]
            return True
        return False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AnalyticsSnapshot
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AnalyticsSnapshot:
    """特定時点の全キャンペーン横断スナップショット"""
    timestamp:         float
    campaign_count:    int
    total_impressions: int
    total_clicks:      int
    total_conversions: int
    total_spend:       float
    total_revenue:     float
    aggregate_kpis:    KpiSet

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp":         self.timestamp,
            "campaign_count":    self.campaign_count,
            "total_impressions": self.total_impressions,
            "total_clicks":      self.total_clicks,
            "total_conversions": self.total_conversions,
            "total_spend":       round(self.total_spend, 2),
            "total_revenue":     round(self.total_revenue, 2),
            "aggregate_kpis":    self.aggregate_kpis.to_dict(),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignAnalyticsDashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CampaignAnalyticsDashboard:
    """全キャンペーン横断の分析ダッシュボード"""

    def __init__(
        self,
        store: Optional[CampaignAnalyticsStore] = None,
        calculator: Optional[KpiCalculator] = None,
    ) -> None:
        self.store = store or CampaignAnalyticsStore()
        self.calculator = calculator or KpiCalculator()

    def campaign_kpis(self, campaign_id: str) -> Optional[KpiSet]:
        """単一キャンペーンの KPI を返す"""
        m = self.store.get(campaign_id)
        if m is None:
            return None
        return self.calculator.compute(m)

    def snapshot(self) -> AnalyticsSnapshot:
        """全キャンペーン横断のスナップショットを生成する"""
        all_metrics = [
            self.store.get(cid) for cid in self.store.list_campaigns()
        ]
        all_metrics = [m for m in all_metrics if m is not None]

        # 横断集計用の仮想 CampaignMetrics を構築
        agg = CampaignMetrics(campaign_id="__aggregate__")
        for m in all_metrics:
            agg.points.extend(m.points)

        return AnalyticsSnapshot(
            timestamp=time.time(),
            campaign_count=len(all_metrics),
            total_impressions=agg.total_impressions,
            total_clicks=agg.total_clicks,
            total_conversions=agg.total_conversions,
            total_spend=agg.total_spend,
            total_revenue=agg.total_revenue,
            aggregate_kpis=self.calculator.compute(agg),
        )

    def rank_by_kpi(self, metric: str = "roas") -> List[Dict[str, Any]]:
        """
        全キャンペーンを指定 KPI でランキングする。

        metric: KpiSet の属性名 ("ctr"/"cvr"/"cpc"/"cpa"/"cpm"/"roas"/"roi")
        """
        rows: List[Dict[str, Any]] = []
        for cid in self.store.list_campaigns():
            m = self.store.get(cid)
            if m is None:
                continue
            kpis = self.calculator.compute(m)
            rows.append({
                "campaign_id": cid,
                "value": getattr(kpis, metric, 0.0),
                "kpis": kpis.to_dict(),
            })
        # CPC/CPA は低いほど良いが、ここでは降順統一（呼び出し側で解釈）
        rows.sort(key=lambda r: r["value"], reverse=True)
        return rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AnalyticsReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnalyticsReportEngine:
    """分析レポート生成"""

    def __init__(self, dashboard: CampaignAnalyticsDashboard) -> None:
        self._dash = dashboard

    def summary_json(self) -> Dict[str, Any]:
        """全キャンペーンサマリーを JSON で返す"""
        snap = self._dash.snapshot()
        return {
            "snapshot": snap.to_dict(),
            "ranking_by_roas": self._dash.rank_by_kpi("roas"),
        }

    def campaign_markdown(self, campaign_id: str) -> str:
        """単一キャンペーンの分析レポートを Markdown で返す"""
        m = self._dash.store.get(campaign_id)
        if m is None:
            return f"# エラー\nキャンペーン `{campaign_id}` のデータがありません。"

        kpis = self._dash.calculator.compute(m)
        lines = [
            f"# キャンペーン分析: {campaign_id}",
            "",
            "## 累積実績",
            "| 指標 | 値 |",
            "|------|-----|",
            f"| インプレッション | {m.total_impressions:,} |",
            f"| クリック | {m.total_clicks:,} |",
            f"| コンバージョン | {m.total_conversions:,} |",
            f"| 広告費 | {m.total_spend:,.2f} |",
            f"| 売上 | {m.total_revenue:,.2f} |",
            "",
            "## KPI",
            "| KPI | 値 |",
            "|-----|-----|",
            f"| CTR | {kpis.ctr:.2%} |",
            f"| CVR | {kpis.cvr:.2%} |",
            f"| CPC | {kpis.cpc:,.2f} |",
            f"| CPA | {kpis.cpa:,.2f} |",
            f"| CPM | {kpis.cpm:,.2f} |",
            f"| ROAS | {kpis.roas:.2f} |",
            f"| ROI | {kpis.roi:.2%} |",
            "",
            f"*データ点数: {len(m.points)}*",
        ]
        return "\n".join(lines)
