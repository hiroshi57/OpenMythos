"""
Sprint 77B — 水質モニタリング

河川・水道の水質パラメータ（pH / 濁度 / 塩素 / DO）を管理し、
基準値との比較・異常判定・施設別サマリーを行う。
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ─── Enums ────────────────────────────────────────────────────────


class WaterParam(str, Enum):
    PH        = "ph"         # pH 値
    TURBIDITY = "turbidity"  # 濁度 (NTU)
    CHLORINE  = "chlorine"   # 残留塩素 (mg/L)
    DO        = "do"         # 溶存酸素 (mg/L)


class WaterQualityStatus(str, Enum):
    SAFE     = "safe"     # 基準内
    CAUTION  = "caution"  # 注意
    UNSAFE   = "unsafe"   # 基準超過


class SourceType(str, Enum):
    RIVER    = "river"    # 河川
    TAP      = "tap"      # 水道
    WELL     = "well"     # 井戸
    LAKE     = "lake"     # 湖沼


# ─── 基準値定義 ────────────────────────────────────────────────────

# (safe_range_low, safe_range_high, caution_margin)
_STANDARDS: Dict[WaterParam, Tuple[float, float, float]] = {
    WaterParam.PH:        (6.5, 8.5, 0.5),    # WHO 基準: 6.5–8.5
    WaterParam.TURBIDITY: (0.0, 1.0,  2.0),   # 水道水基準: ≤1 NTU
    WaterParam.CHLORINE:  (0.1, 1.0,  0.5),   # 残留塩素: 0.1–1.0 mg/L
    WaterParam.DO:        (6.0, 14.0, 1.0),   # 溶存酸素: ≥6 mg/L が健全
}

_UNITS: Dict[WaterParam, str] = {
    WaterParam.PH:        "pH",
    WaterParam.TURBIDITY: "NTU",
    WaterParam.CHLORINE:  "mg/L",
    WaterParam.DO:        "mg/L",
}


def _assess_status(param: WaterParam, value: float) -> WaterQualityStatus:
    lo, hi, margin = _STANDARDS[param]
    if lo <= value <= hi:
        return WaterQualityStatus.SAFE
    elif (lo - margin) <= value <= (hi + margin):
        return WaterQualityStatus.CAUTION
    return WaterQualityStatus.UNSAFE


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class WaterReading:
    """水質計測値1件。"""
    reading_id: str
    station_id: str
    city: str
    source_type: SourceType
    param: WaterParam
    value: float
    hour: int = 12
    day: int = 1

    @property
    def status(self) -> WaterQualityStatus:
        return _assess_status(self.param, self.value)

    @property
    def unit(self) -> str:
        return _UNITS[self.param]

    def to_dict(self) -> dict:
        return {
            "reading_id": self.reading_id,
            "station_id": self.station_id,
            "city": self.city,
            "source_type": self.source_type.value,
            "param": self.param.value,
            "value": self.value,
            "unit": self.unit,
            "status": self.status.value,
            "hour": self.hour,
            "day": self.day,
        }


@dataclass
class WaterStationSummary:
    """観測所単位の水質サマリー。"""
    station_id: str
    city: str
    param: WaterParam
    avg_value: float
    min_value: float
    max_value: float
    reading_count: int
    unsafe_count: int
    caution_count: int
    overall_status: WaterQualityStatus
    unit: str

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "city": self.city,
            "param": self.param.value,
            "avg_value": round(self.avg_value, 3),
            "min_value": round(self.min_value, 3),
            "max_value": round(self.max_value, 3),
            "reading_count": self.reading_count,
            "unsafe_count": self.unsafe_count,
            "caution_count": self.caution_count,
            "overall_status": self.overall_status.value,
            "unit": self.unit,
        }


# ─── Store ────────────────────────────────────────────────────────


class WaterQualityStore:
    """水質計測値のインメモリ CRUD。"""

    def __init__(self) -> None:
        self._readings: Dict[str, WaterReading] = {}

    def add(self, reading: WaterReading) -> WaterReading:
        self._readings[reading.reading_id] = reading
        return reading

    def get(self, reading_id: str) -> Optional[WaterReading]:
        return self._readings.get(reading_id)

    def list_all(self) -> List[WaterReading]:
        return list(self._readings.values())

    def list_by_city(self, city: str) -> List[WaterReading]:
        return [r for r in self._readings.values() if r.city == city]

    def list_by_station(self, station_id: str) -> List[WaterReading]:
        return [r for r in self._readings.values() if r.station_id == station_id]

    def list_unsafe(self, city: Optional[str] = None) -> List[WaterReading]:
        readings = self.list_by_city(city) if city else self.list_all()
        return [r for r in readings if r.status == WaterQualityStatus.UNSAFE]

    def delete(self, reading_id: str) -> bool:
        if reading_id in self._readings:
            del self._readings[reading_id]
            return True
        return False

    def count(self) -> int:
        return len(self._readings)


# ─── Monitor ──────────────────────────────────────────────────────

_STATUS_ORDER = [WaterQualityStatus.SAFE, WaterQualityStatus.CAUTION, WaterQualityStatus.UNSAFE]


class WaterQualityMonitor:
    """水質モニタリングエンジン。"""

    def __init__(self, store: Optional[WaterQualityStore] = None) -> None:
        self.store = store or WaterQualityStore()

    def add_reading(
        self,
        reading_id: str,
        station_id: str,
        city: str,
        source_type: SourceType,
        param: WaterParam,
        value: float,
        hour: int = 12,
        day: int = 1,
    ) -> WaterReading:
        reading = WaterReading(
            reading_id=reading_id,
            station_id=station_id,
            city=city,
            source_type=source_type,
            param=param,
            value=value,
            hour=hour,
            day=day,
        )
        return self.store.add(reading)

    def station_summary(
        self, station_id: str, param: WaterParam
    ) -> Optional[WaterStationSummary]:
        readings = [r for r in self.store.list_by_station(station_id) if r.param == param]
        if not readings:
            return None
        values = [r.value for r in readings]
        unsafe_n = sum(1 for r in readings if r.status == WaterQualityStatus.UNSAFE)
        caution_n = sum(1 for r in readings if r.status == WaterQualityStatus.CAUTION)
        overall = WaterQualityStatus.SAFE
        if unsafe_n > 0:
            overall = WaterQualityStatus.UNSAFE
        elif caution_n > 0:
            overall = WaterQualityStatus.CAUTION
        return WaterStationSummary(
            station_id=station_id,
            city=readings[0].city,
            param=param,
            avg_value=statistics.mean(values),
            min_value=min(values),
            max_value=max(values),
            reading_count=len(readings),
            unsafe_count=unsafe_n,
            caution_count=caution_n,
            overall_status=overall,
            unit=_UNITS[param],
        )

    def get_unsafe_readings(self, city: Optional[str] = None) -> List[WaterReading]:
        return self.store.list_unsafe(city)

    def city_report(self, city: str) -> dict:
        """都市全体の水質レポート（パラメータ別集計）。"""
        readings = self.store.list_by_city(city)
        if not readings:
            return {"city": city, "reading_count": 0, "params": {}}

        param_data: Dict[str, Dict] = {}
        for param in WaterParam:
            param_readings = [r for r in readings if r.param == param]
            if not param_readings:
                continue
            values = [r.value for r in param_readings]
            unsafe_n = sum(1 for r in param_readings if r.status == WaterQualityStatus.UNSAFE)
            param_data[param.value] = {
                "avg": round(statistics.mean(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "count": len(param_readings),
                "unsafe_count": unsafe_n,
                "unit": _UNITS[param],
            }

        total_unsafe = sum(1 for r in readings if r.status == WaterQualityStatus.UNSAFE)
        return {
            "city": city,
            "reading_count": len(readings),
            "total_unsafe": total_unsafe,
            "params": param_data,
        }
