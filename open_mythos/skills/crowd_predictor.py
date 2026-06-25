"""
Sprint 76C — 群衆予測

スポット単位で人流スナップショットを管理し、
イベント・天候・時間帯から混雑レベルを予測する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ─── Enums ────────────────────────────────────────────────────────


class CrowdLevel(str, Enum):
    SPARSE  = "sparse"   # 閑散
    NORMAL  = "normal"   # 通常
    CROWDED = "crowded"  # 混雑
    PACKED  = "packed"   # 超混雑


class WeatherCondition(str, Enum):
    SUNNY  = "sunny"   # 晴れ
    CLOUDY = "cloudy"  # 曇り
    RAINY  = "rainy"   # 雨
    SNOWY  = "snowy"   # 雪


class EventType(str, Enum):
    NONE       = "none"        # イベントなし
    FESTIVAL   = "festival"    # 祭り・フェスティバル
    CONCERT    = "concert"     # コンサート
    SPORTS     = "sports"      # スポーツ
    HOLIDAY    = "holiday"     # 祝日
    COMMUTE    = "commute"     # 通勤ラッシュ


# ─── スコア係数定義 ────────────────────────────────────────────────

# 基準人流 (人/時): 1.0 = NORMAL の下限
_BASE_CROWD = 500.0

# 時間帯係数
_HOUR_SCALE: Dict[int, float] = {
    **{h: 0.2 for h in range(0, 6)},     # 深夜
    **{h: 1.4 for h in range(6, 10)},    # 朝ラッシュ
    **{h: 1.0 for h in range(10, 16)},   # 日中
    **{h: 1.6 for h in range(16, 20)},   # 夕ラッシュ
    **{h: 0.7 for h in range(20, 24)},   # 夜間
}

# 天候係数
_WEATHER_SCALE: Dict[WeatherCondition, float] = {
    WeatherCondition.SUNNY:  1.2,
    WeatherCondition.CLOUDY: 1.0,
    WeatherCondition.RAINY:  0.6,
    WeatherCondition.SNOWY:  0.4,
}

# イベント追加人数
_EVENT_BOOST: Dict[EventType, float] = {
    EventType.NONE:     0.0,
    EventType.FESTIVAL: 3000.0,
    EventType.CONCERT:  5000.0,
    EventType.SPORTS:   4000.0,
    EventType.HOLIDAY:  1500.0,
    EventType.COMMUTE:  2000.0,
}

# CrowdLevel 分類しきい値（人/時）
_LEVEL_THRESHOLDS: Dict[CrowdLevel, float] = {
    CrowdLevel.PACKED:  3000.0,
    CrowdLevel.CROWDED: 1500.0,
    CrowdLevel.NORMAL:   500.0,
    CrowdLevel.SPARSE:     0.0,
}


def _classify_crowd(count: float) -> CrowdLevel:
    for level, threshold in _LEVEL_THRESHOLDS.items():
        if count >= threshold:
            return level
    return CrowdLevel.SPARSE


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class CrowdSnapshot:
    """スポットの人流スナップショット。"""
    snapshot_id: str
    spot_name: str
    city: str
    count: int           # 観測人数
    hour: int = 12       # 観測時刻
    weather: WeatherCondition = WeatherCondition.SUNNY
    event: EventType = EventType.NONE

    @property
    def level(self) -> CrowdLevel:
        return _classify_crowd(float(self.count))

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "spot_name": self.spot_name,
            "city": self.city,
            "count": self.count,
            "hour": self.hour,
            "weather": self.weather.value,
            "event": self.event.value,
            "level": self.level.value,
        }


@dataclass
class CrowdPrediction:
    """1スポット・1時刻の人流予測結果。"""
    spot_name: str
    city: str
    hour: int
    weather: WeatherCondition
    event: EventType
    predicted_count: float
    predicted_level: CrowdLevel
    hour_scale: float
    weather_scale: float
    event_boost: float

    def to_dict(self) -> dict:
        return {
            "spot_name": self.spot_name,
            "city": self.city,
            "hour": self.hour,
            "weather": self.weather.value,
            "event": self.event.value,
            "predicted_count": round(self.predicted_count),
            "predicted_level": self.predicted_level.value,
            "factors": {
                "hour_scale": self.hour_scale,
                "weather_scale": self.weather_scale,
                "event_boost": self.event_boost,
            },
        }


@dataclass
class HeatmapCell:
    """ヒートマップ1セル（スポット集計）。"""
    spot_name: str
    avg_count: float
    level: CrowdLevel
    snapshot_count: int

    def to_dict(self) -> dict:
        return {
            "spot_name": self.spot_name,
            "avg_count": round(self.avg_count),
            "level": self.level.value,
            "snapshot_count": self.snapshot_count,
        }


# ─── Store ────────────────────────────────────────────────────────


class CrowdStore:
    """人流スナップショットのインメモリ CRUD。"""

    def __init__(self) -> None:
        self._snapshots: Dict[str, CrowdSnapshot] = {}

    def add(self, snap: CrowdSnapshot) -> CrowdSnapshot:
        self._snapshots[snap.snapshot_id] = snap
        return snap

    def get(self, snapshot_id: str) -> Optional[CrowdSnapshot]:
        return self._snapshots.get(snapshot_id)

    def list_all(self) -> List[CrowdSnapshot]:
        return list(self._snapshots.values())

    def list_by_city(self, city: str) -> List[CrowdSnapshot]:
        return [s for s in self._snapshots.values() if s.city == city]

    def list_by_spot(self, spot_name: str, city: str) -> List[CrowdSnapshot]:
        return [s for s in self._snapshots.values()
                if s.spot_name == spot_name and s.city == city]

    def delete(self, snapshot_id: str) -> bool:
        if snapshot_id in self._snapshots:
            del self._snapshots[snapshot_id]
            return True
        return False

    def count(self) -> int:
        return len(self._snapshots)


# ─── Predictor ────────────────────────────────────────────────────


class CrowdPredictor:
    """群衆予測エンジン。"""

    def __init__(self, store: Optional[CrowdStore] = None) -> None:
        self.store = store or CrowdStore()

    def add_snapshot(
        self,
        snapshot_id: str,
        spot_name: str,
        city: str,
        count: int,
        hour: int = 12,
        weather: WeatherCondition = WeatherCondition.SUNNY,
        event: EventType = EventType.NONE,
    ) -> CrowdSnapshot:
        snap = CrowdSnapshot(
            snapshot_id=snapshot_id,
            spot_name=spot_name,
            city=city,
            count=count,
            hour=hour,
            weather=weather,
            event=event,
        )
        return self.store.add(snap)

    def predict(
        self,
        spot_name: str,
        city: str,
        hour: int,
        weather: WeatherCondition = WeatherCondition.SUNNY,
        event: EventType = EventType.NONE,
        base_count: Optional[float] = None,
    ) -> CrowdPrediction:
        """イベント・天候・時間帯から人流を予測する。"""
        # 実績データがあれば基準値に使用
        if base_count is None:
            snaps = self.store.list_by_spot(spot_name, city)
            if snaps:
                base_count = sum(s.count for s in snaps) / len(snaps)
            else:
                base_count = _BASE_CROWD

        h_scale = _HOUR_SCALE.get(hour, 1.0)
        w_scale = _WEATHER_SCALE[weather]
        e_boost = _EVENT_BOOST[event]

        predicted = base_count * h_scale * w_scale + e_boost
        predicted = max(0.0, predicted)

        return CrowdPrediction(
            spot_name=spot_name,
            city=city,
            hour=hour,
            weather=weather,
            event=event,
            predicted_count=predicted,
            predicted_level=_classify_crowd(predicted),
            hour_scale=h_scale,
            weather_scale=w_scale,
            event_boost=e_boost,
        )

    def heatmap(self, city: str) -> List[HeatmapCell]:
        """都市内の全スポットのヒートマップ（混雑度集計）を返す。"""
        snaps = self.store.list_by_city(city)
        if not snaps:
            return []

        spot_map: Dict[str, List[int]] = {}
        for s in snaps:
            spot_map.setdefault(s.spot_name, []).append(s.count)

        cells: List[HeatmapCell] = []
        for spot_name, counts in spot_map.items():
            avg = sum(counts) / len(counts)
            cells.append(HeatmapCell(
                spot_name=spot_name,
                avg_count=avg,
                level=_classify_crowd(avg),
                snapshot_count=len(counts),
            ))

        cells.sort(key=lambda c: c.avg_count, reverse=True)
        return cells

    def daily_forecast(
        self,
        spot_name: str,
        city: str,
        weather: WeatherCondition = WeatherCondition.SUNNY,
        event: EventType = EventType.NONE,
    ) -> List[CrowdPrediction]:
        """24時間分の予測一覧を返す。"""
        return [
            self.predict(spot_name, city, hour=h, weather=weather, event=event)
            for h in range(24)
        ]
