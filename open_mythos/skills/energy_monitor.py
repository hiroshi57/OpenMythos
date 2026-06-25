"""
Sprint 76B — エネルギー消費モニタリング

電力・ガス・水道の消費量を管理し、
集計・異常検出・アラート生成を行う。
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ─── Enums ────────────────────────────────────────────────────────


class EnergyType(str, Enum):
    ELECTRICITY = "electricity"  # 電力 (kWh)
    GAS         = "gas"          # ガス (m³)
    WATER       = "water"        # 水道 (m³)


class AnomalyLevel(str, Enum):
    NORMAL   = "normal"   # 通常範囲
    HIGH     = "high"     # 高使用量
    CRITICAL = "critical" # 異常使用量


# ─── 単位定義 ──────────────────────────────────────────────────────

_UNITS: Dict[EnergyType, str] = {
    EnergyType.ELECTRICITY: "kWh",
    EnergyType.GAS:         "m³",
    EnergyType.WATER:       "m³",
}

# 標準日次消費量の警告倍率 (warning, critical)
_ANOMALY_MULTIPLIERS: Tuple[float, float] = (1.5, 2.5)


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class EnergyReading:
    """エネルギー消費量の1計測値。"""
    reading_id: str
    facility_id: str
    city: str
    energy_type: EnergyType
    value: float           # 消費量
    hour: int = 12         # 計測時刻 (0–23)
    day: int = 1           # 日 (1–31)

    @property
    def unit(self) -> str:
        return _UNITS[self.energy_type]

    def to_dict(self) -> dict:
        return {
            "reading_id": self.reading_id,
            "facility_id": self.facility_id,
            "city": self.city,
            "energy_type": self.energy_type.value,
            "value": self.value,
            "unit": self.unit,
            "hour": self.hour,
            "day": self.day,
        }


@dataclass
class EnergySummary:
    """施設・都市単位のエネルギー消費集計。"""
    scope: str             # facility_id or city
    energy_type: EnergyType
    total: float
    avg_per_reading: float
    max_value: float
    min_value: float
    reading_count: int
    unit: str

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "energy_type": self.energy_type.value,
            "total": round(self.total, 2),
            "avg_per_reading": round(self.avg_per_reading, 2),
            "max_value": round(self.max_value, 2),
            "min_value": round(self.min_value, 2),
            "reading_count": self.reading_count,
            "unit": self.unit,
        }


@dataclass
class EnergyAnomaly:
    """消費量の異常検出結果。"""
    facility_id: str
    city: str
    energy_type: EnergyType
    reading_id: str
    value: float
    baseline: float
    multiplier: float
    level: AnomalyLevel
    unit: str

    def to_dict(self) -> dict:
        return {
            "facility_id": self.facility_id,
            "city": self.city,
            "energy_type": self.energy_type.value,
            "reading_id": self.reading_id,
            "value": round(self.value, 2),
            "baseline": round(self.baseline, 2),
            "multiplier": round(self.multiplier, 2),
            "level": self.level.value,
            "unit": self.unit,
        }


# ─── Store ────────────────────────────────────────────────────────


class EnergyStore:
    """エネルギー計測値のインメモリ CRUD。"""

    def __init__(self) -> None:
        self._readings: Dict[str, EnergyReading] = {}

    def add(self, reading: EnergyReading) -> EnergyReading:
        self._readings[reading.reading_id] = reading
        return reading

    def get(self, reading_id: str) -> Optional[EnergyReading]:
        return self._readings.get(reading_id)

    def list_all(self) -> List[EnergyReading]:
        return list(self._readings.values())

    def list_by_facility(self, facility_id: str) -> List[EnergyReading]:
        return [r for r in self._readings.values() if r.facility_id == facility_id]

    def list_by_city(self, city: str) -> List[EnergyReading]:
        return [r for r in self._readings.values() if r.city == city]

    def list_by_type(self, energy_type: EnergyType) -> List[EnergyReading]:
        return [r for r in self._readings.values() if r.energy_type == energy_type]

    def delete(self, reading_id: str) -> bool:
        if reading_id in self._readings:
            del self._readings[reading_id]
            return True
        return False

    def count(self) -> int:
        return len(self._readings)


# ─── Monitor ──────────────────────────────────────────────────────


class EnergyMonitor:
    """エネルギー消費モニタリングエンジン。"""

    def __init__(self, store: Optional[EnergyStore] = None) -> None:
        self.store = store or EnergyStore()

    def add_reading(
        self,
        reading_id: str,
        facility_id: str,
        city: str,
        energy_type: EnergyType,
        value: float,
        hour: int = 12,
        day: int = 1,
    ) -> EnergyReading:
        reading = EnergyReading(
            reading_id=reading_id,
            facility_id=facility_id,
            city=city,
            energy_type=energy_type,
            value=value,
            hour=hour,
            day=day,
        )
        return self.store.add(reading)

    def summarize_facility(
        self, facility_id: str, energy_type: EnergyType
    ) -> Optional[EnergySummary]:
        readings = [
            r for r in self.store.list_by_facility(facility_id)
            if r.energy_type == energy_type
        ]
        if not readings:
            return None
        values = [r.value for r in readings]
        return EnergySummary(
            scope=facility_id,
            energy_type=energy_type,
            total=sum(values),
            avg_per_reading=statistics.mean(values),
            max_value=max(values),
            min_value=min(values),
            reading_count=len(readings),
            unit=_UNITS[energy_type],
        )

    def summarize_city(
        self, city: str, energy_type: EnergyType
    ) -> Optional[EnergySummary]:
        readings = [
            r for r in self.store.list_by_city(city)
            if r.energy_type == energy_type
        ]
        if not readings:
            return None
        values = [r.value for r in readings]
        return EnergySummary(
            scope=city,
            energy_type=energy_type,
            total=sum(values),
            avg_per_reading=statistics.mean(values),
            max_value=max(values),
            min_value=min(values),
            reading_count=len(readings),
            unit=_UNITS[energy_type],
        )

    def detect_anomalies(
        self, city: str, energy_type: EnergyType
    ) -> List[EnergyAnomaly]:
        """平均値に対して著しく高い消費量を異常として検出する。"""
        readings = [
            r for r in self.store.list_by_city(city)
            if r.energy_type == energy_type
        ]
        if len(readings) < 2:
            return []

        values = [r.value for r in readings]
        baseline = statistics.mean(values)
        if baseline == 0:
            return []

        warn_mult, crit_mult = _ANOMALY_MULTIPLIERS
        anomalies: List[EnergyAnomaly] = []
        for r in readings:
            mult = r.value / baseline
            if mult >= crit_mult:
                level = AnomalyLevel.CRITICAL
            elif mult >= warn_mult:
                level = AnomalyLevel.HIGH
            else:
                continue
            anomalies.append(EnergyAnomaly(
                facility_id=r.facility_id,
                city=r.city,
                energy_type=r.energy_type,
                reading_id=r.reading_id,
                value=r.value,
                baseline=baseline,
                multiplier=mult,
                level=level,
                unit=_UNITS[energy_type],
            ))

        anomalies.sort(key=lambda a: a.multiplier, reverse=True)
        return anomalies

    def hourly_profile(self, city: str, energy_type: EnergyType) -> List[dict]:
        """時間帯別平均消費量プロファイル (hour 0–23)。"""
        readings = [
            r for r in self.store.list_by_city(city)
            if r.energy_type == energy_type
        ]
        hour_buckets: Dict[int, List[float]] = {h: [] for h in range(24)}
        for r in readings:
            if 0 <= r.hour <= 23:
                hour_buckets[r.hour].append(r.value)

        profile = []
        for hour in range(24):
            vals = hour_buckets[hour]
            profile.append({
                "hour": hour,
                "avg_value": round(statistics.mean(vals), 2) if vals else 0.0,
                "reading_count": len(vals),
                "unit": _UNITS[energy_type],
            })
        return profile
