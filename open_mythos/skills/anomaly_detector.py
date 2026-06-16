"""
Sprint 66C — KPI 異常検知アラート

キャンペーン分析の時系列メトリクス (campaign_analytics) を監視し、
KPI の急変（スパイク/ドロップ）を統計的に検知してアラートを発する。

オブジェクト:
  AlertSeverity   : アラート深刻度 (Info/Warning/Critical)
  AnomalyType     : 異常種別 (Spike/Drop/Stale)
  Alert           : 検知された 1 件のアラート
  DetectorConfig  : 検知設定 (z 閾値 / 変化率閾値 / 最小サンプル)
  AnomalyDetector : 異常検知エンジン (z-score + 変化率)
  AlertStore      : アラート履歴ストア
  AnomalyReportEngine: アラートレポート生成

設計方針:
  - 外部依存なし (math のみ)
  - z-score 法（移動統計）と 直近変化率法の併用
  - campaign_analytics.CampaignMetrics の時系列を入力
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from open_mythos.skills.campaign_analytics import CampaignMetrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AlertSeverity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    SPIKE = "spike"   # 急増
    DROP  = "drop"    # 急減
    STALE = "stale"   # 更新停滞（変化なし）


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Alert / 設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Alert:
    """検知されたアラート 1 件"""
    id:           str
    campaign_id:  str
    metric:       str
    anomaly_type: AnomalyType
    severity:     AlertSeverity
    current:      float
    baseline:     float
    z_score:      float
    change_pct:   float
    message:      str
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":           self.id,
            "campaign_id":  self.campaign_id,
            "metric":       self.metric,
            "anomaly_type": self.anomaly_type.value,
            "severity":     self.severity.value,
            "current":      round(self.current, 4),
            "baseline":     round(self.baseline, 4),
            "z_score":      round(self.z_score, 4),
            "change_pct":   round(self.change_pct, 4),
            "message":      self.message,
            "timestamp":    self.timestamp,
        }


@dataclass
class DetectorConfig:
    """異常検知設定"""
    z_warning:       float = 2.0    # z-score 警告閾値
    z_critical:      float = 3.0    # z-score 重大閾値
    change_warning:  float = 0.5    # 変化率 警告閾値 (±50%)
    change_critical: float = 1.0    # 変化率 重大閾値 (±100%)
    min_samples:     int   = 3      # 検知に必要な最小データ点数
    stale_threshold: int   = 3      # 同値が続いたら stale とみなす連続数


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AnomalyDetector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: List[float], mean: float) -> float:
    if len(xs) < 2:
        return 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


class AnomalyDetector:
    """
    KPI 異常検知エンジン。

    直近の値を、それ以前の履歴（ベースライン）と比較して
    z-score と変化率から異常を判定する。

    Usage:
        detector = AnomalyDetector()
        alerts = detector.detect(metrics, metric="clicks")
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()

    def detect(
        self, metrics: CampaignMetrics, metric: str = "clicks"
    ) -> List[Alert]:
        """
        単一指標について異常を検知する。検知されなければ空リスト。
        """
        values = [float(getattr(p, metric, 0)) for p in metrics.points]
        if len(values) < self.config.min_samples:
            return []

        current = values[-1]
        baseline_values = values[:-1]
        baseline = _mean(baseline_values)
        std = _stdev(baseline_values, baseline)

        alerts: List[Alert] = []

        # 1) stale 検知（直近 N 点が同値）
        stale_alert = self._check_stale(metrics, metric, values, current)
        if stale_alert:
            alerts.append(stale_alert)
            return alerts  # stale なら他の検知はスキップ

        # 2) z-score 検知
        z = (current - baseline) / std if std > 0 else 0.0
        change_pct = ((current - baseline) / baseline) if baseline else 0.0

        severity = self._severity(abs(z), abs(change_pct))
        if severity is not None:
            anomaly = AnomalyType.SPIKE if current > baseline else AnomalyType.DROP
            alerts.append(Alert(
                id=str(uuid.uuid4()),
                campaign_id=metrics.campaign_id,
                metric=metric,
                anomaly_type=anomaly,
                severity=severity,
                current=current,
                baseline=baseline,
                z_score=z,
                change_pct=change_pct,
                message=self._message(metric, anomaly, current, baseline, change_pct),
            ))
        return alerts

    def detect_multi(
        self, metrics: CampaignMetrics, metric_list: Optional[List[str]] = None
    ) -> List[Alert]:
        """複数指標を一括検知する"""
        if metric_list is None:
            metric_list = ["impressions", "clicks", "conversions", "spend", "revenue"]
        alerts: List[Alert] = []
        for m in metric_list:
            alerts.extend(self.detect(metrics, m))
        return alerts

    # ---- 内部 ----

    def _check_stale(
        self, metrics: CampaignMetrics, metric: str,
        values: List[float], current: float
    ) -> Optional[Alert]:
        n = self.config.stale_threshold
        if len(values) < n:
            return None
        recent = values[-n:]
        if all(v == recent[0] for v in recent) and current != 0:
            return Alert(
                id=str(uuid.uuid4()),
                campaign_id=metrics.campaign_id,
                metric=metric,
                anomaly_type=AnomalyType.STALE,
                severity=AlertSeverity.WARNING,
                current=current,
                baseline=current,
                z_score=0.0,
                change_pct=0.0,
                message=f"{metric} が直近 {n} 点で変化なし（更新停滞の可能性）",
            )
        return None

    def _severity(self, abs_z: float, abs_change: float) -> Optional[AlertSeverity]:
        cfg = self.config
        if abs_z >= cfg.z_critical or abs_change >= cfg.change_critical:
            return AlertSeverity.CRITICAL
        if abs_z >= cfg.z_warning or abs_change >= cfg.change_warning:
            return AlertSeverity.WARNING
        return None

    def _message(
        self, metric: str, anomaly: AnomalyType,
        current: float, baseline: float, change_pct: float
    ) -> str:
        direction = "急増" if anomaly == AnomalyType.SPIKE else "急減"
        return (
            f"{metric} が{direction}: {baseline:.2f} → {current:.2f} "
            f"({change_pct:+.1%})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlertStore
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AlertStore:
    """アラート履歴ストア（インメモリ）"""

    def __init__(self) -> None:
        self._alerts: List[Alert] = []

    def add(self, alert: Alert) -> Alert:
        self._alerts.append(alert)
        return alert

    def add_many(self, alerts: List[Alert]) -> None:
        self._alerts.extend(alerts)

    def list_all(self) -> List[Alert]:
        return list(self._alerts)

    def list_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        return [a for a in self._alerts if a.severity == severity]

    def list_by_campaign(self, campaign_id: str) -> List[Alert]:
        return [a for a in self._alerts if a.campaign_id == campaign_id]

    def critical_count(self) -> int:
        return len(self.list_by_severity(AlertSeverity.CRITICAL))

    def count(self) -> int:
        return len(self._alerts)

    def clear(self) -> None:
        self._alerts.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AnomalyReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnomalyReportEngine:
    """アラートレポート生成"""

    def __init__(self, store: AlertStore) -> None:
        self._store = store

    def summary_json(self) -> Dict[str, Any]:
        alerts = self._store.list_all()
        by_severity: Dict[str, int] = {}
        for a in alerts:
            by_severity[a.severity.value] = by_severity.get(a.severity.value, 0) + 1
        return {
            "total_alerts":   len(alerts),
            "by_severity":    by_severity,
            "critical_count": self._store.critical_count(),
            "alerts":         [a.to_dict() for a in alerts],
        }

    def markdown(self) -> str:
        alerts = self._store.list_all()
        lines = [
            "# KPI 異常検知レポート",
            "",
            f"**総アラート数**: {len(alerts)}  ",
            f"**重大**: {self._store.critical_count()}  ",
            "",
        ]
        if not alerts:
            lines.append("*アラートはありません*")
            return "\n".join(lines)

        lines += [
            "| 深刻度 | キャンペーン | 指標 | 種別 | メッセージ |",
            "|--------|------------|------|------|-----------|",
        ]
        # 重大度順にソート
        order = {AlertSeverity.CRITICAL: 0, AlertSeverity.WARNING: 1, AlertSeverity.INFO: 2}
        for a in sorted(alerts, key=lambda x: order.get(x.severity, 9)):
            lines.append(
                f"| {a.severity.value} | {a.campaign_id} | {a.metric} | "
                f"{a.anomaly_type.value} | {a.message} |"
            )
        return "\n".join(lines)
