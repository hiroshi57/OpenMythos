"""
Sprint 75C — 都市インフラダッシュボード

混雑シミュレーション・アクセシビリティ・地下水位の
3モジュールを束ねて都市全体のインフラ状態を可視化する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from open_mythos.skills.city_map import CityName
from open_mythos.skills.accessibility import AccessibilityAnalyzer
from open_mythos.skills.crowd_simulator import CrowdSimulator
from open_mythos.skills.groundwater import FloodRiskAssessor, FloodRiskLevel, Season


# ─── Enums ────────────────────────────────────────────────────────


class MetricStatus(str, Enum):
    GOOD  = "good"   # 良好
    WARN  = "warn"   # 注意
    ALERT = "alert"  # 危険


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class DashboardMetric:
    """単一指標"""
    name:   str
    value:  float
    unit:   str
    status: MetricStatus
    note:   str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": round(self.value, 2),
            "unit": self.unit,
            "status": self.status.value,
            "note": self.note,
        }


@dataclass
class StationPanel:
    """1駅分のダッシュボードパネル"""
    station_id: str
    metrics: List[DashboardMetric] = field(default_factory=list)

    @property
    def worst_status(self) -> MetricStatus:
        priority = {MetricStatus.ALERT: 2, MetricStatus.WARN: 1, MetricStatus.GOOD: 0}
        if not self.metrics:
            return MetricStatus.GOOD
        return max(self.metrics, key=lambda m: priority[m.status]).status

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "worst_status": self.worst_status.value,
            "metrics": [m.to_dict() for m in self.metrics],
        }


@dataclass
class CityDashboard:
    """都市全体のダッシュボード"""
    city: str
    panels: List[StationPanel]
    generated_at: str  # ISO8601

    def summary(self) -> Dict[str, int]:
        counts = {MetricStatus.ALERT: 0, MetricStatus.WARN: 0, MetricStatus.GOOD: 0}
        for panel in self.panels:
            counts[panel.worst_status] += 1
        return {
            "alert_count": counts[MetricStatus.ALERT],
            "warn_count":  counts[MetricStatus.WARN],
            "ok_count":    counts[MetricStatus.GOOD],
            "total":       len(self.panels),
        }

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "generated_at": self.generated_at,
            "summary": self.summary(),
            "panels": [p.to_dict() for p in self.panels],
        }


# ─── 指標ステータス変換ヘルパー ────────────────────────────────────


def _crowd_status(occupancy_rate: float) -> MetricStatus:
    if occupancy_rate >= 1.5:
        return MetricStatus.ALERT
    elif occupancy_rate >= 1.0:
        return MetricStatus.WARN
    return MetricStatus.GOOD


def _access_status(score: float) -> MetricStatus:
    if score < 30.0:
        return MetricStatus.ALERT
    elif score < 60.0:
        return MetricStatus.WARN
    return MetricStatus.GOOD


def _flood_status(level: FloodRiskLevel) -> MetricStatus:
    if level in (FloodRiskLevel.VERY_HIGH, FloodRiskLevel.HIGH):
        return MetricStatus.ALERT
    elif level == FloodRiskLevel.MODERATE:
        return MetricStatus.WARN
    return MetricStatus.GOOD


# ─── InfraDashboard ───────────────────────────────────────────────


class InfraDashboard:
    """
    3モジュールを統合して都市インフラダッシュボードを生成するクラス。

    - 混雑: CrowdSimulator.snapshot(station_id, hour).occupancy_rate
    - アクセシビリティ: AccessibilityAnalyzer.score(station_id).score
    - 地下水位: FloodRiskAssessor.city_risk(city).level (都市全体値を各駅に適用)
    """

    def __init__(
        self,
        crowd:    CrowdSimulator,
        analyzer: AccessibilityAnalyzer,
        assessor: FloodRiskAssessor,
        season:   Season = Season.SUMMER,
    ) -> None:
        self._crowd    = crowd
        self._analyzer = analyzer
        self._assessor = assessor
        self._season   = season

    def city_panel(self, city: CityName, hour: int) -> CityDashboard:
        """指定都市の全登録駅のパネルを生成する"""
        # アクセシビリティアナライザーから都市の駅IDを取得
        city_profiles = [
            sid for sid in self._analyzer.all_station_ids()
            if self._analyzer.get_profile(sid) is not None
            and self._analyzer.get_profile(sid).city == city
        ]

        # 地下水リスクは都市単位で取得
        flood_result = self._assessor.city_risk(city, self._season)
        flood_level  = flood_result.level if flood_result else FloodRiskLevel.VERY_LOW

        panels: List[StationPanel] = []
        for sid in city_profiles:
            metrics: List[DashboardMetric] = []

            # 1. 混雑指標
            snap = self._crowd.snapshot(sid, hour)
            if snap is not None:
                occ = snap.occupancy_rate
                metrics.append(DashboardMetric(
                    name="crowd_occupancy",
                    value=occ,
                    unit="rate",
                    status=_crowd_status(occ),
                    note=snap.level.value,
                ))

            # 2. アクセシビリティ指標
            acc_score = self._analyzer.score(sid)
            if acc_score is not None:
                metrics.append(DashboardMetric(
                    name="accessibility_score",
                    value=acc_score.score,
                    unit="pt",
                    status=_access_status(acc_score.score),
                    note=acc_score.level.value,
                ))

            # 3. 浸水リスク指標 (都市全体値)
            metrics.append(DashboardMetric(
                name="flood_risk",
                value=flood_result.total_score if flood_result else 0.0,
                unit="score",
                status=_flood_status(flood_level),
                note=flood_level.value,
            ))

            panels.append(StationPanel(station_id=sid, metrics=metrics))

        return CityDashboard(
            city=city.value,
            panels=panels,
            generated_at=datetime.now(tz=timezone.utc).isoformat(),
        )

    def alert_stations(
        self,
        city: CityName,
        hour: int,
        min_status: MetricStatus = MetricStatus.ALERT,
    ) -> List[StationPanel]:
        """ALERT 以上のステータスを持つ駅パネルのみを返す"""
        priority = {MetricStatus.ALERT: 2, MetricStatus.WARN: 1, MetricStatus.GOOD: 0}
        threshold = priority[min_status]
        dashboard = self.city_panel(city, hour)
        return [
            p for p in dashboard.panels
            if priority[p.worst_status] >= threshold
        ]

    def multi_city_summary(
        self,
        cities: List[CityName],
        hour: int,
    ) -> List[Dict]:
        """複数都市のサマリーリストを返す"""
        results = []
        for city in cities:
            db = self.city_panel(city, hour)
            row = {"city": city.value}
            row.update(db.summary())
            results.append(row)
        return results


# ─── DashboardDataset ─────────────────────────────────────────────


class DashboardDataset:
    """InfraDashboard 用データセット (3モジュール統合)"""

    @classmethod
    def build(cls, season: Season = Season.SUMMER) -> "DashboardDataset":
        from open_mythos.skills.crowd_simulator import CrowdDataset
        from open_mythos.skills.accessibility import AccessibilityDataset
        from open_mythos.skills.groundwater import GroundwaterDataset

        crowd    = CrowdDataset.build()
        analyzer = AccessibilityDataset.build()
        assessor = GroundwaterDataset.build()

        return cls(crowd=crowd, analyzer=analyzer, assessor=assessor, season=season)

    def __init__(
        self,
        crowd:    CrowdSimulator,
        analyzer: AccessibilityAnalyzer,
        assessor: FloodRiskAssessor,
        season:   Season = Season.SUMMER,
    ) -> None:
        self.crowd    = crowd
        self.analyzer = analyzer
        self.assessor = assessor
        self.season   = season

    def dashboard(self) -> InfraDashboard:
        return InfraDashboard(
            crowd=self.crowd,
            analyzer=self.analyzer,
            assessor=self.assessor,
            season=self.season,
        )
