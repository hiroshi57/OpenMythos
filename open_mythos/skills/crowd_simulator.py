"""
Sprint 74A — 駅混雑シミュレーション

乗降客数モデル + 時間帯別混雑度を計算する。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from open_mythos.skills.city_map import CityName, Station


# ─── Enums ────────────────────────────────────────────────────────


class CrowdLevel(str, Enum):
    EMPTY = "empty"           # 0〜20%
    QUIET = "quiet"           # 20〜40%
    MODERATE = "moderate"     # 40〜60%
    BUSY = "busy"             # 60〜80%
    CROWDED = "crowded"       # 80〜100%
    OVERCROWDED = "overcrowded"  # 100%超


class TimeSlot(str, Enum):
    EARLY_MORNING = "early_morning"   # 5–7時
    MORNING_RUSH = "morning_rush"     # 7–9時
    MID_MORNING = "mid_morning"       # 9–11時
    MIDDAY = "midday"                 # 11–13時
    AFTERNOON = "afternoon"           # 13–17時
    EVENING_RUSH = "evening_rush"     # 17–20時
    NIGHT = "night"                   # 20–23時
    LATE_NIGHT = "late_night"         # 23–5時


# ─── Time Slot helpers ────────────────────────────────────────────

_SLOT_MAP: Dict[int, TimeSlot] = {}
for _h in range(24):
    if 5 <= _h < 7:
        _SLOT_MAP[_h] = TimeSlot.EARLY_MORNING
    elif 7 <= _h < 9:
        _SLOT_MAP[_h] = TimeSlot.MORNING_RUSH
    elif 9 <= _h < 11:
        _SLOT_MAP[_h] = TimeSlot.MID_MORNING
    elif 11 <= _h < 13:
        _SLOT_MAP[_h] = TimeSlot.MIDDAY
    elif 13 <= _h < 17:
        _SLOT_MAP[_h] = TimeSlot.AFTERNOON
    elif 17 <= _h < 20:
        _SLOT_MAP[_h] = TimeSlot.EVENING_RUSH
    elif 20 <= _h < 23:
        _SLOT_MAP[_h] = TimeSlot.NIGHT
    else:
        _SLOT_MAP[_h] = TimeSlot.LATE_NIGHT


def hour_to_slot(hour: int) -> TimeSlot:
    return _SLOT_MAP[hour % 24]


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class StationProfile:
    """駅の乗降客プロファイル"""
    station_id: str
    station_name: str
    daily_passengers: int        # 1日あたり乗降客数
    platform_count: int = 2
    base_capacity: int = 0       # プラットフォーム容量 (0=自動計算)

    def __post_init__(self) -> None:
        if self.base_capacity == 0:
            self.base_capacity = self.platform_count * 500

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "daily_passengers": self.daily_passengers,
            "platform_count": self.platform_count,
            "base_capacity": self.base_capacity,
        }


@dataclass
class CrowdSnapshot:
    """ある時点の混雑スナップショット"""
    station_id: str
    station_name: str
    hour: int
    slot: TimeSlot
    estimated_passengers: int
    capacity: int
    occupancy_rate: float        # 0.0〜1.5+ (1.0=満員)
    level: CrowdLevel

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "hour": self.hour,
            "slot": self.slot.value,
            "estimated_passengers": self.estimated_passengers,
            "capacity": self.capacity,
            "occupancy_rate": round(self.occupancy_rate, 3),
            "level": self.level.value,
        }


@dataclass
class DailyProfile:
    """1日の時間帯別混雑プロファイル"""
    station_id: str
    station_name: str
    snapshots: List[CrowdSnapshot]

    @property
    def peak_hour(self) -> int:
        return max(self.snapshots, key=lambda s: s.occupancy_rate).hour

    @property
    def peak_level(self) -> CrowdLevel:
        return max(self.snapshots, key=lambda s: s.occupancy_rate).level

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "peak_hour": self.peak_hour,
            "peak_level": self.peak_level.value,
            "snapshots": [s.to_dict() for s in self.snapshots],
        }


# ─── CrowdModel ───────────────────────────────────────────────────

# 時間帯別の乗降客分布係数 (合計≒1.0)
_SLOT_WEIGHTS: Dict[TimeSlot, float] = {
    TimeSlot.EARLY_MORNING:  0.04,
    TimeSlot.MORNING_RUSH:   0.22,
    TimeSlot.MID_MORNING:    0.10,
    TimeSlot.MIDDAY:         0.14,
    TimeSlot.AFTERNOON:      0.16,
    TimeSlot.EVENING_RUSH:   0.24,
    TimeSlot.NIGHT:          0.08,
    TimeSlot.LATE_NIGHT:     0.02,
}

# 各スロット内の各時間の分布 (2〜4時間スロットを均等分)
_SLOT_HOURS: Dict[TimeSlot, List[int]] = {
    TimeSlot.EARLY_MORNING:  [5, 6],
    TimeSlot.MORNING_RUSH:   [7, 8],
    TimeSlot.MID_MORNING:    [9, 10],
    TimeSlot.MIDDAY:         [11, 12],
    TimeSlot.AFTERNOON:      [13, 14, 15, 16],
    TimeSlot.EVENING_RUSH:   [17, 18, 19],
    TimeSlot.NIGHT:          [20, 21, 22],
    TimeSlot.LATE_NIGHT:     [23, 0, 1, 2, 3, 4],
}


def _occupancy_to_level(rate: float) -> CrowdLevel:
    if rate < 0.2:
        return CrowdLevel.EMPTY
    elif rate < 0.4:
        return CrowdLevel.QUIET
    elif rate < 0.6:
        return CrowdLevel.MODERATE
    elif rate < 0.8:
        return CrowdLevel.BUSY
    elif rate <= 1.0:
        return CrowdLevel.CROWDED
    return CrowdLevel.OVERCROWDED


class CrowdSimulator:
    """時間帯別混雑シミュレーター"""

    def __init__(self) -> None:
        self._profiles: Dict[str, StationProfile] = {}

    def register(self, profile: StationProfile) -> None:
        self._profiles[profile.station_id] = profile

    def get_profile(self, station_id: str) -> Optional[StationProfile]:
        return self._profiles.get(station_id)

    def snapshot(self, station_id: str, hour: int) -> Optional[CrowdSnapshot]:
        profile = self._profiles.get(station_id)
        if profile is None:
            return None
        slot = hour_to_slot(hour)
        slot_weight = _SLOT_WEIGHTS[slot]
        slot_hours = _SLOT_HOURS[slot]
        per_hour = (profile.daily_passengers * slot_weight) / len(slot_hours)
        rate = per_hour / profile.base_capacity
        return CrowdSnapshot(
            station_id=station_id,
            station_name=profile.station_name,
            hour=hour,
            slot=slot,
            estimated_passengers=int(per_hour),
            capacity=profile.base_capacity,
            occupancy_rate=rate,
            level=_occupancy_to_level(rate),
        )

    def daily_profile(self, station_id: str) -> Optional[DailyProfile]:
        profile = self._profiles.get(station_id)
        if profile is None:
            return None
        snapshots = [self.snapshot(station_id, h) for h in range(24)]
        return DailyProfile(
            station_id=station_id,
            station_name=profile.station_name,
            snapshots=[s for s in snapshots if s is not None],
        )

    def compare(
        self, station_ids: List[str], hour: int
    ) -> List[CrowdSnapshot]:
        results = []
        for sid in station_ids:
            s = self.snapshot(sid, hour)
            if s is not None:
                results.append(s)
        return sorted(results, key=lambda x: x.occupancy_rate, reverse=True)

    def all_station_ids(self) -> List[str]:
        return list(self._profiles.keys())


# ─── DefaultDataset ───────────────────────────────────────────────


class CrowdDataset:
    """主要駅の乗降客プリセット"""

    _PRESETS = [
        # 東京
        ("tokyo-shinjuku",        "新宿",       3_500_000, 4),
        ("tokyo-tokyo",           "東京",        2_000_000, 2),
        ("tokyo-ginza",           "銀座",          400_000, 2),
        ("tokyo-kasumigaseki",    "霞ケ関",         200_000, 2),
        ("tokyo-otemachi",        "大手町",         800_000, 2),
        ("tokyo-ogikubo",         "荻窪",           300_000, 2),
        ("tokyo-nakano",          "中野",           500_000, 2),
        ("tokyo-yotsuya",         "四ツ谷",         250_000, 2),
        # 大阪
        ("osaka-namba",           "なんば",         800_000, 4),
        ("osaka-shinsaibashi",    "心斎橋",         350_000, 2),
        ("osaka-senri-chuo",      "千里中央",       200_000, 2),
        # 名古屋
        ("nagoya-nagoya",         "名古屋",         900_000, 4),
        ("nagoya-sakae",          "栄",             400_000, 2),
        # 横浜
        ("yokohama-yokohama",     "横浜",           900_000, 4),
        # 福岡
        ("fukuoka-hakata",        "博多",           500_000, 4),
        ("fukuoka-tenjin",        "天神",           450_000, 4),
    ]

    @classmethod
    def build(cls) -> CrowdSimulator:
        sim = CrowdSimulator()
        for sid, name, daily, platforms in cls._PRESETS:
            sim.register(StationProfile(
                station_id=sid,
                station_name=name,
                daily_passengers=daily,
                platform_count=platforms,
            ))
        return sim
