"""
Sprint 76A — 交通量分析

道路セグメント単位で交通量・速度・密度を管理し、
時間帯別の渋滞スコアとホットスポット検出を行う。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ─── Enums ────────────────────────────────────────────────────────


class TrafficLevel(str, Enum):
    CLEAR     = "clear"      # 通常走行
    MODERATE  = "moderate"   # やや混雑
    CONGESTED = "congested"  # 渋滞
    GRIDLOCK  = "gridlock"   # 完全停滞


class TimeOfDay(str, Enum):
    EARLY_MORNING = "early_morning"  # 0–6時
    MORNING_RUSH  = "morning_rush"   # 6–10時
    MIDDAY        = "midday"         # 10–16時
    EVENING_RUSH  = "evening_rush"   # 16–20時
    NIGHT         = "night"          # 20–24時


# ─── 閾値定義（速度ベース km/h） ──────────────────────────────────

_SPEED_THRESHOLDS: Dict[TrafficLevel, float] = {
    TrafficLevel.CLEAR:     60.0,   # 60 km/h 以上
    TrafficLevel.MODERATE:  40.0,   # 40–60 km/h
    TrafficLevel.CONGESTED: 20.0,   # 20–40 km/h
    TrafficLevel.GRIDLOCK:   0.0,   # 20 km/h 未満
}

# 時間帯別渋滞スケール係数（1.0 = 通常）
_TIME_SCALE: Dict[TimeOfDay, float] = {
    TimeOfDay.EARLY_MORNING: 0.3,
    TimeOfDay.MORNING_RUSH:  1.8,
    TimeOfDay.MIDDAY:        1.0,
    TimeOfDay.EVENING_RUSH:  2.0,
    TimeOfDay.NIGHT:         0.5,
}


def _classify_level(speed_kmh: float) -> TrafficLevel:
    if speed_kmh >= _SPEED_THRESHOLDS[TrafficLevel.CLEAR]:
        return TrafficLevel.CLEAR
    elif speed_kmh >= _SPEED_THRESHOLDS[TrafficLevel.MODERATE]:
        return TrafficLevel.MODERATE
    elif speed_kmh >= _SPEED_THRESHOLDS[TrafficLevel.CONGESTED]:
        return TrafficLevel.CONGESTED
    return TrafficLevel.GRIDLOCK


def _get_time_of_day(hour: int) -> TimeOfDay:
    """hour: 0–23"""
    if hour < 6:
        return TimeOfDay.EARLY_MORNING
    elif hour < 10:
        return TimeOfDay.MORNING_RUSH
    elif hour < 16:
        return TimeOfDay.MIDDAY
    elif hour < 20:
        return TimeOfDay.EVENING_RUSH
    return TimeOfDay.NIGHT


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class TrafficSegment:
    """道路セグメント1件分の計測データ。"""
    segment_id: str
    road_name: str
    city: str
    volume: int           # 台数/時間
    speed_kmh: float      # 平均速度 km/h
    density: float        # 密度 (台/km)
    length_km: float = 1.0
    hour: int = 8         # 計測時刻 (0–23)

    @property
    def level(self) -> TrafficLevel:
        return _classify_level(self.speed_kmh)

    @property
    def congestion_score(self) -> float:
        """0.0（通常）〜 1.0（完全停滞）の渋滞スコア。"""
        max_speed = 80.0
        return max(0.0, min(1.0, 1.0 - self.speed_kmh / max_speed))

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "road_name": self.road_name,
            "city": self.city,
            "volume": self.volume,
            "speed_kmh": self.speed_kmh,
            "density": self.density,
            "length_km": self.length_km,
            "hour": self.hour,
            "level": self.level.value,
            "congestion_score": round(self.congestion_score, 3),
            "time_of_day": _get_time_of_day(self.hour).value,
        }


@dataclass
class TrafficHotspot:
    """渋滞ホットスポット（複数セグメントの集約）。"""
    city: str
    road_name: str
    avg_speed_kmh: float
    max_congestion_score: float
    segment_count: int
    level: TrafficLevel

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "road_name": self.road_name,
            "avg_speed_kmh": round(self.avg_speed_kmh, 1),
            "max_congestion_score": round(self.max_congestion_score, 3),
            "segment_count": self.segment_count,
            "level": self.level.value,
        }


@dataclass
class HourlyForecast:
    """1時間分の交通量予測。"""
    hour: int
    time_of_day: TimeOfDay
    predicted_level: TrafficLevel
    scale_factor: float
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "hour": self.hour,
            "time_of_day": self.time_of_day.value,
            "predicted_level": self.predicted_level.value,
            "scale_factor": self.scale_factor,
            "note": self.note,
        }


# ─── Store ────────────────────────────────────────────────────────


class TrafficStore:
    """交通セグメントデータのインメモリ CRUD。"""

    def __init__(self) -> None:
        self._segments: Dict[str, TrafficSegment] = {}

    def add(self, seg: TrafficSegment) -> TrafficSegment:
        self._segments[seg.segment_id] = seg
        return seg

    def get(self, segment_id: str) -> Optional[TrafficSegment]:
        return self._segments.get(segment_id)

    def list_all(self) -> List[TrafficSegment]:
        return list(self._segments.values())

    def list_by_city(self, city: str) -> List[TrafficSegment]:
        return [s for s in self._segments.values() if s.city == city]

    def delete(self, segment_id: str) -> bool:
        if segment_id in self._segments:
            del self._segments[segment_id]
            return True
        return False

    def count(self) -> int:
        return len(self._segments)


# ─── Analyzer ─────────────────────────────────────────────────────


class TrafficAnalyzer:
    """交通量分析エンジン。"""

    def __init__(self, store: Optional[TrafficStore] = None) -> None:
        self.store = store or TrafficStore()

    def add_segment(
        self,
        segment_id: str,
        road_name: str,
        city: str,
        volume: int,
        speed_kmh: float,
        density: float,
        length_km: float = 1.0,
        hour: int = 8,
    ) -> TrafficSegment:
        seg = TrafficSegment(
            segment_id=segment_id,
            road_name=road_name,
            city=city,
            volume=volume,
            speed_kmh=speed_kmh,
            density=density,
            length_km=length_km,
            hour=hour,
        )
        return self.store.add(seg)

    def get_hotspots(self, city: str, top_n: int = 5) -> List[TrafficHotspot]:
        """渋滞スコアが高い上位 N ホットスポットを返す。"""
        segments = self.store.list_by_city(city)
        if not segments:
            return []

        # road_name 別に集約
        road_map: Dict[str, List[TrafficSegment]] = {}
        for seg in segments:
            road_map.setdefault(seg.road_name, []).append(seg)

        hotspots: List[TrafficHotspot] = []
        for road_name, segs in road_map.items():
            avg_speed = sum(s.speed_kmh for s in segs) / len(segs)
            max_score = max(s.congestion_score for s in segs)
            worst_level = _classify_level(min(s.speed_kmh for s in segs))
            hotspots.append(TrafficHotspot(
                city=city,
                road_name=road_name,
                avg_speed_kmh=avg_speed,
                max_congestion_score=max_score,
                segment_count=len(segs),
                level=worst_level,
            ))

        hotspots.sort(key=lambda h: h.max_congestion_score, reverse=True)
        return hotspots[:top_n]

    def predict_by_hour(self, base_speed_kmh: float = 50.0) -> List[HourlyForecast]:
        """24時間の交通量予測を生成する。"""
        forecasts: List[HourlyForecast] = []
        notes = {
            TimeOfDay.MORNING_RUSH: "朝のラッシュ時",
            TimeOfDay.EVENING_RUSH: "夕方のラッシュ時",
            TimeOfDay.EARLY_MORNING: "早朝・交通少",
            TimeOfDay.NIGHT: "夜間・交通少",
            TimeOfDay.MIDDAY: "日中・標準",
        }
        for hour in range(24):
            tod = _get_time_of_day(hour)
            scale = _TIME_SCALE[tod]
            predicted_speed = max(5.0, base_speed_kmh / scale)
            level = _classify_level(predicted_speed)
            forecasts.append(HourlyForecast(
                hour=hour,
                time_of_day=tod,
                predicted_level=level,
                scale_factor=scale,
                note=notes.get(tod, ""),
            ))
        return forecasts

    def city_summary(self, city: str) -> dict:
        """都市全体の交通状況サマリー。"""
        segments = self.store.list_by_city(city)
        if not segments:
            return {"city": city, "segment_count": 0, "avg_speed_kmh": None, "dominant_level": None}

        avg_speed = sum(s.speed_kmh for s in segments) / len(segments)
        level_counts: Dict[str, int] = {}
        for s in segments:
            level_counts[s.level.value] = level_counts.get(s.level.value, 0) + 1
        dominant = max(level_counts, key=lambda k: level_counts[k])

        return {
            "city": city,
            "segment_count": len(segments),
            "avg_speed_kmh": round(avg_speed, 1),
            "dominant_level": dominant,
            "level_breakdown": level_counts,
        }
