"""
Sprint 77A — 災害アラート管理

地震・洪水・火災の警報を管理し、
重大度判定・対象エリアへの通知スケジュールを生成する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ─── Enums ────────────────────────────────────────────────────────


class DisasterType(str, Enum):
    EARTHQUAKE = "earthquake"  # 地震
    FLOOD      = "flood"       # 洪水
    FIRE       = "fire"        # 火災
    TSUNAMI    = "tsunami"     # 津波
    TYPHOON    = "typhoon"     # 台風


class AlertLevel(str, Enum):
    INFO     = "info"      # 情報
    WATCH    = "watch"     # 注意
    WARNING  = "warning"   # 警戒
    CRITICAL = "critical"  # 緊急


class AlertStatus(str, Enum):
    ACTIVE   = "active"    # 発令中
    RESOLVED = "resolved"  # 解除済み
    EXPIRED  = "expired"   # 期限切れ


# ─── 重大度マッピング ──────────────────────────────────────────────

# 災害種別ごとのデフォルト AlertLevel
_DEFAULT_LEVEL: Dict[DisasterType, AlertLevel] = {
    DisasterType.EARTHQUAKE: AlertLevel.WARNING,
    DisasterType.FLOOD:      AlertLevel.WARNING,
    DisasterType.FIRE:       AlertLevel.WATCH,
    DisasterType.TSUNAMI:    AlertLevel.CRITICAL,
    DisasterType.TYPHOON:    AlertLevel.WATCH,
}

# 推奨避難行動
_RECOMMENDED_ACTIONS: Dict[AlertLevel, List[str]] = {
    AlertLevel.INFO:     ["情報を注視してください"],
    AlertLevel.WATCH:    ["避難の準備をしてください", "非常用持ち出し袋を確認してください"],
    AlertLevel.WARNING:  ["速やかに避難してください", "低地・河川沿いは立入禁止"],
    AlertLevel.CRITICAL: ["直ちに避難してください", "生命の危険があります", "高所・堅固な建物へ移動"],
}


def _recommend_actions(level: AlertLevel) -> List[str]:
    return _RECOMMENDED_ACTIONS.get(level, [])


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class DisasterAlert:
    """災害アラート1件。"""
    alert_id: str
    disaster_type: DisasterType
    city: str
    level: AlertLevel
    status: AlertStatus = AlertStatus.ACTIVE
    magnitude: Optional[float] = None   # 地震: マグニチュード / 洪水: 水位 m
    description: str = ""
    affected_areas: List[str] = field(default_factory=list)

    @property
    def recommended_actions(self) -> List[str]:
        return _recommend_actions(self.level)

    @property
    def is_active(self) -> bool:
        return self.status == AlertStatus.ACTIVE

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "disaster_type": self.disaster_type.value,
            "city": self.city,
            "level": self.level.value,
            "status": self.status.value,
            "magnitude": self.magnitude,
            "description": self.description,
            "affected_areas": self.affected_areas,
            "recommended_actions": self.recommended_actions,
            "is_active": self.is_active,
        }


@dataclass
class AlertSummary:
    """都市の災害アラートサマリー。"""
    city: str
    total: int
    active: int
    by_level: Dict[str, int]
    by_type: Dict[str, int]
    highest_level: Optional[str]

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "total": self.total,
            "active": self.active,
            "by_level": self.by_level,
            "by_type": self.by_type,
            "highest_level": self.highest_level,
        }


# ─── Store ────────────────────────────────────────────────────────


class AlertStore:
    """アラートのインメモリ CRUD。"""

    def __init__(self) -> None:
        self._alerts: Dict[str, DisasterAlert] = {}

    def add(self, alert: DisasterAlert) -> DisasterAlert:
        self._alerts[alert.alert_id] = alert
        return alert

    def get(self, alert_id: str) -> Optional[DisasterAlert]:
        return self._alerts.get(alert_id)

    def list_all(self) -> List[DisasterAlert]:
        return list(self._alerts.values())

    def list_by_city(self, city: str) -> List[DisasterAlert]:
        return [a for a in self._alerts.values() if a.city == city]

    def list_active(self, city: Optional[str] = None) -> List[DisasterAlert]:
        alerts = self.list_by_city(city) if city else self.list_all()
        return [a for a in alerts if a.is_active]

    def resolve(self, alert_id: str) -> bool:
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.RESOLVED
            return True
        return False

    def delete(self, alert_id: str) -> bool:
        if alert_id in self._alerts:
            del self._alerts[alert_id]
            return True
        return False

    def count(self) -> int:
        return len(self._alerts)


# ─── Manager ──────────────────────────────────────────────────────

_LEVEL_ORDER = [AlertLevel.INFO, AlertLevel.WATCH, AlertLevel.WARNING, AlertLevel.CRITICAL]


class DisasterAlertManager:
    """災害アラート管理エンジン。"""

    def __init__(self, store: Optional[AlertStore] = None) -> None:
        self.store = store or AlertStore()

    def issue_alert(
        self,
        alert_id: str,
        disaster_type: DisasterType,
        city: str,
        level: Optional[AlertLevel] = None,
        magnitude: Optional[float] = None,
        description: str = "",
        affected_areas: Optional[List[str]] = None,
    ) -> DisasterAlert:
        resolved_level = level or _DEFAULT_LEVEL[disaster_type]
        alert = DisasterAlert(
            alert_id=alert_id,
            disaster_type=disaster_type,
            city=city,
            level=resolved_level,
            magnitude=magnitude,
            description=description,
            affected_areas=affected_areas or [],
        )
        return self.store.add(alert)

    def resolve_alert(self, alert_id: str) -> bool:
        return self.store.resolve(alert_id)

    def get_active_alerts(self, city: Optional[str] = None) -> List[DisasterAlert]:
        alerts = self.store.list_active(city)
        alerts.sort(key=lambda a: _LEVEL_ORDER.index(a.level), reverse=True)
        return alerts

    def city_summary(self, city: str) -> AlertSummary:
        alerts = self.store.list_by_city(city)
        active = [a for a in alerts if a.is_active]
        by_level: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        for a in alerts:
            by_level[a.level.value] = by_level.get(a.level.value, 0) + 1
            by_type[a.disaster_type.value] = by_type.get(a.disaster_type.value, 0) + 1

        highest: Optional[str] = None
        if active:
            best = max(active, key=lambda a: _LEVEL_ORDER.index(a.level))
            highest = best.level.value

        return AlertSummary(
            city=city,
            total=len(alerts),
            active=len(active),
            by_level=by_level,
            by_type=by_type,
            highest_level=highest,
        )
