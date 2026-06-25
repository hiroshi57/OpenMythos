"""
Sprint 77C — 騒音マッピング

地点別の騒音レベル（dB）を管理し、
時間帯規制対比・超過判定・騒音マップ生成を行う。
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ─── Enums ────────────────────────────────────────────────────────


class ZoneType(str, Enum):
    RESIDENTIAL  = "residential"   # 住宅地
    COMMERCIAL   = "commercial"    # 商業地
    INDUSTRIAL   = "industrial"    # 工業地
    QUIET_ZONE   = "quiet_zone"    # 静音区域（病院・学校等）


class TimeSlot(str, Enum):
    DAYTIME  = "daytime"   # 昼間 (6–22時)
    NIGHTTIME = "nighttime" # 夜間 (22–6時)


class NoiseStatus(str, Enum):
    COMPLIANT   = "compliant"    # 規制内
    NEAR_LIMIT  = "near_limit"   # 規制値近傍 (-5dB以内)
    VIOLATION   = "violation"    # 規制超過


# ─── 騒音規制値 (dB) ──────────────────────────────────────────────

# (daytime_limit, nighttime_limit)
_LIMITS: Dict[ZoneType, Tuple[float, float]] = {
    ZoneType.RESIDENTIAL:  (55.0, 45.0),
    ZoneType.COMMERCIAL:   (65.0, 60.0),
    ZoneType.INDUSTRIAL:   (70.0, 65.0),
    ZoneType.QUIET_ZONE:   (45.0, 40.0),
}

_NEAR_LIMIT_MARGIN = 5.0  # 規制値まで 5dB 以内を NEAR_LIMIT とする


def _get_time_slot(hour: int) -> TimeSlot:
    return TimeSlot.DAYTIME if 6 <= hour < 22 else TimeSlot.NIGHTTIME


def _assess_noise(zone: ZoneType, db: float, hour: int) -> NoiseStatus:
    slot = _get_time_slot(hour)
    day_lim, night_lim = _LIMITS[zone]
    limit = day_lim if slot == TimeSlot.DAYTIME else night_lim
    if db > limit:
        return NoiseStatus.VIOLATION
    elif db >= limit - _NEAR_LIMIT_MARGIN:
        return NoiseStatus.NEAR_LIMIT
    return NoiseStatus.COMPLIANT


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class NoiseMeasurement:
    """騒音計測値1件。"""
    measurement_id: str
    location_name: str
    city: str
    zone_type: ZoneType
    db_level: float       # 騒音レベル (dB)
    hour: int = 12

    @property
    def time_slot(self) -> TimeSlot:
        return _get_time_slot(self.hour)

    @property
    def status(self) -> NoiseStatus:
        return _assess_noise(self.zone_type, self.db_level, self.hour)

    @property
    def limit(self) -> float:
        day_lim, night_lim = _LIMITS[self.zone_type]
        return day_lim if self.time_slot == TimeSlot.DAYTIME else night_lim

    @property
    def excess_db(self) -> float:
        return max(0.0, self.db_level - self.limit)

    def to_dict(self) -> dict:
        return {
            "measurement_id": self.measurement_id,
            "location_name": self.location_name,
            "city": self.city,
            "zone_type": self.zone_type.value,
            "db_level": self.db_level,
            "hour": self.hour,
            "time_slot": self.time_slot.value,
            "status": self.status.value,
            "limit_db": self.limit,
            "excess_db": round(self.excess_db, 1),
        }


@dataclass
class NoiseMapCell:
    """騒音マップ1セル（地点集計）。"""
    location_name: str
    city: str
    zone_type: ZoneType
    avg_db: float
    max_db: float
    violation_count: int
    measurement_count: int
    status: NoiseStatus

    def to_dict(self) -> dict:
        return {
            "location_name": self.location_name,
            "city": self.city,
            "zone_type": self.zone_type.value,
            "avg_db": round(self.avg_db, 1),
            "max_db": round(self.max_db, 1),
            "violation_count": self.violation_count,
            "measurement_count": self.measurement_count,
            "status": self.status.value,
        }


@dataclass
class NoiseReport:
    """都市騒音レポート。"""
    city: str
    total_measurements: int
    violations: int
    near_limit: int
    compliant: int
    worst_location: Optional[str]
    avg_db_city: float

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "total_measurements": self.total_measurements,
            "violations": self.violations,
            "near_limit": self.near_limit,
            "compliant": self.compliant,
            "compliance_rate": round(
                self.compliant / self.total_measurements, 3
            ) if self.total_measurements > 0 else 1.0,
            "worst_location": self.worst_location,
            "avg_db_city": round(self.avg_db_city, 1),
        }


# ─── Store ────────────────────────────────────────────────────────


class NoiseMeasurementStore:
    """騒音計測値のインメモリ CRUD。"""

    def __init__(self) -> None:
        self._measurements: Dict[str, NoiseMeasurement] = {}

    def add(self, m: NoiseMeasurement) -> NoiseMeasurement:
        self._measurements[m.measurement_id] = m
        return m

    def get(self, measurement_id: str) -> Optional[NoiseMeasurement]:
        return self._measurements.get(measurement_id)

    def list_all(self) -> List[NoiseMeasurement]:
        return list(self._measurements.values())

    def list_by_city(self, city: str) -> List[NoiseMeasurement]:
        return [m for m in self._measurements.values() if m.city == city]

    def list_violations(self, city: Optional[str] = None) -> List[NoiseMeasurement]:
        ms = self.list_by_city(city) if city else self.list_all()
        return [m for m in ms if m.status == NoiseStatus.VIOLATION]

    def delete(self, measurement_id: str) -> bool:
        if measurement_id in self._measurements:
            del self._measurements[measurement_id]
            return True
        return False

    def count(self) -> int:
        return len(self._measurements)


# ─── Mapper ───────────────────────────────────────────────────────


class NoiseMapper:
    """騒音マッピングエンジン。"""

    def __init__(self, store: Optional[NoiseMeasurementStore] = None) -> None:
        self.store = store or NoiseMeasurementStore()

    def add_measurement(
        self,
        measurement_id: str,
        location_name: str,
        city: str,
        zone_type: ZoneType,
        db_level: float,
        hour: int = 12,
    ) -> NoiseMeasurement:
        m = NoiseMeasurement(
            measurement_id=measurement_id,
            location_name=location_name,
            city=city,
            zone_type=zone_type,
            db_level=db_level,
            hour=hour,
        )
        return self.store.add(m)

    def get_violations(self, city: Optional[str] = None) -> List[NoiseMeasurement]:
        violations = self.store.list_violations(city)
        violations.sort(key=lambda m: m.excess_db, reverse=True)
        return violations

    def generate_map(self, city: str) -> List[NoiseMapCell]:
        """都市内の全地点の騒音マップを生成する。"""
        measurements = self.store.list_by_city(city)
        if not measurements:
            return []

        loc_map: Dict[str, List[NoiseMeasurement]] = {}
        for m in measurements:
            loc_map.setdefault(m.location_name, []).append(m)

        cells: List[NoiseMapCell] = []
        for loc_name, ms in loc_map.items():
            dbs = [m.db_level for m in ms]
            avg_db = statistics.mean(dbs)
            max_db = max(dbs)
            violation_n = sum(1 for m in ms if m.status == NoiseStatus.VIOLATION)
            # 代表ゾーンは最初の計測値のゾーン
            zone = ms[0].zone_type
            # 代表ステータス: 1件でも VIOLATION があれば VIOLATION
            if violation_n > 0:
                status = NoiseStatus.VIOLATION
            elif any(m.status == NoiseStatus.NEAR_LIMIT for m in ms):
                status = NoiseStatus.NEAR_LIMIT
            else:
                status = NoiseStatus.COMPLIANT

            cells.append(NoiseMapCell(
                location_name=loc_name,
                city=city,
                zone_type=zone,
                avg_db=avg_db,
                max_db=max_db,
                violation_count=violation_n,
                measurement_count=len(ms),
                status=status,
            ))

        cells.sort(key=lambda c: c.avg_db, reverse=True)
        return cells

    def city_report(self, city: str) -> NoiseReport:
        """都市全体の騒音レポート。"""
        measurements = self.store.list_by_city(city)
        if not measurements:
            return NoiseReport(
                city=city, total_measurements=0, violations=0,
                near_limit=0, compliant=0, worst_location=None, avg_db_city=0.0,
            )

        violations = [m for m in measurements if m.status == NoiseStatus.VIOLATION]
        near = [m for m in measurements if m.status == NoiseStatus.NEAR_LIMIT]
        compliant = [m for m in measurements if m.status == NoiseStatus.COMPLIANT]
        avg_db = statistics.mean(m.db_level for m in measurements)

        worst: Optional[str] = None
        if violations:
            worst_m = max(violations, key=lambda m: m.excess_db)
            worst = worst_m.location_name

        return NoiseReport(
            city=city,
            total_measurements=len(measurements),
            violations=len(violations),
            near_limit=len(near),
            compliant=len(compliant),
            worst_location=worst,
            avg_db_city=avg_db,
        )
