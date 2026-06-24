"""
Sprint 75A — 駅環境センサー統合

気温・湿度・CO2・騒音センサーデータを管理し、
閾値ベースのステータス判定とアラート抽出を行う。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from open_mythos.skills.city_map import CityName


# ─── Enums ────────────────────────────────────────────────────────


class SensorType(str, Enum):
    TEMPERATURE = "temperature"   # 気温 (°C)
    HUMIDITY    = "humidity"      # 湿度 (%)
    CO2         = "co2"           # CO2濃度 (ppm)
    NOISE       = "noise"         # 騒音レベル (dB)


class SensorStatus(str, Enum):
    NORMAL   = "normal"    # 通常範囲
    WARNING  = "warning"   # 注意
    CRITICAL = "critical"  # 危険


# ─── 閾値定義 ─────────────────────────────────────────────────────

# (warning_threshold, critical_threshold)
_THRESHOLDS: Dict[SensorType, tuple[float, float]] = {
    SensorType.TEMPERATURE: (28.0, 35.0),   # °C
    SensorType.HUMIDITY:    (70.0, 80.0),   # %
    SensorType.CO2:         (1000.0, 2000.0),  # ppm
    SensorType.NOISE:       (75.0, 90.0),   # dB
}

_UNITS: Dict[SensorType, str] = {
    SensorType.TEMPERATURE: "°C",
    SensorType.HUMIDITY:    "%",
    SensorType.CO2:         "ppm",
    SensorType.NOISE:       "dB",
}


def _classify_status(sensor_type: SensorType, value: float) -> SensorStatus:
    warn, crit = _THRESHOLDS[sensor_type]
    if value >= crit:
        return SensorStatus.CRITICAL
    elif value >= warn:
        return SensorStatus.WARNING
    return SensorStatus.NORMAL


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class SensorReading:
    """センサー読み取り値"""
    sensor_type: SensorType
    value: float
    unit: str
    status: SensorStatus

    @classmethod
    def create(cls, sensor_type: SensorType, value: float) -> "SensorReading":
        return cls(
            sensor_type=sensor_type,
            value=value,
            unit=_UNITS[sensor_type],
            status=_classify_status(sensor_type, value),
        )

    def to_dict(self) -> dict:
        return {
            "sensor_type": self.sensor_type.value,
            "value": self.value,
            "unit": self.unit,
            "status": self.status.value,
        }


@dataclass
class StationEnvironment:
    """駅の環境データ"""
    station_id: str
    station_name: str
    city: CityName
    readings: List[SensorReading] = field(default_factory=list)

    def get_reading(self, sensor_type: SensorType) -> Optional[SensorReading]:
        for r in self.readings:
            if r.sensor_type == sensor_type:
                return r
        return None

    def overall_status(self) -> SensorStatus:
        """全センサー中の最悪ステータスを返す"""
        if not self.readings:
            return SensorStatus.NORMAL
        priority = {
            SensorStatus.CRITICAL: 2,
            SensorStatus.WARNING:  1,
            SensorStatus.NORMAL:   0,
        }
        worst = max(self.readings, key=lambda r: priority[r.status])
        return worst.status

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "city": self.city.value,
            "overall_status": self.overall_status().value,
            "readings": [r.to_dict() for r in self.readings],
        }


@dataclass
class EnvComparisonResult:
    """複数駅の環境比較結果"""
    station_ids: List[str]
    environments: List[StationEnvironment]
    worst_station_id: Optional[str]

    def to_dict(self) -> dict:
        return {
            "station_ids": self.station_ids,
            "worst_station_id": self.worst_station_id,
            "environments": [e.to_dict() for e in self.environments],
        }


# ─── EnvSensorAnalyzer ────────────────────────────────────────────

_STATUS_PRIORITY = {
    SensorStatus.CRITICAL: 2,
    SensorStatus.WARNING:  1,
    SensorStatus.NORMAL:   0,
}


class EnvSensorAnalyzer:
    """環境センサーデータを管理・分析するクラス"""

    def __init__(self) -> None:
        self._envs: Dict[str, StationEnvironment] = {}

    def register(self, env: StationEnvironment) -> None:
        self._envs[env.station_id] = env

    def get_environment(self, station_id: str) -> Optional[StationEnvironment]:
        return self._envs.get(station_id)

    def snapshot(self, station_id: str) -> Optional[StationEnvironment]:
        """指定駅の現在の環境スナップショットを返す"""
        return self._envs.get(station_id)

    def compare(self, station_ids: List[str]) -> EnvComparisonResult:
        """複数駅の環境を比較する"""
        envs = [self._envs[sid] for sid in station_ids if sid in self._envs]
        worst_id: Optional[str] = None
        if envs:
            worst = max(
                envs,
                key=lambda e: _STATUS_PRIORITY[e.overall_status()],
            )
            worst_id = worst.station_id
        return EnvComparisonResult(
            station_ids=station_ids,
            environments=envs,
            worst_station_id=worst_id,
        )

    def alert_stations(
        self,
        city: Optional[CityName] = None,
        min_status: SensorStatus = SensorStatus.WARNING,
    ) -> List[StationEnvironment]:
        """WARNING 以上のステータスを持つ駅を返す"""
        threshold = _STATUS_PRIORITY[min_status]
        results = [
            e for e in self._envs.values()
            if (city is None or e.city == city)
            and _STATUS_PRIORITY[e.overall_status()] >= threshold
        ]
        return sorted(
            results,
            key=lambda e: _STATUS_PRIORITY[e.overall_status()],
            reverse=True,
        )

    def all_station_ids(self) -> List[str]:
        return list(self._envs.keys())


# ─── DefaultDataset ───────────────────────────────────────────────


class EnvSensorDataset:
    """主要駅の環境センサープリセット (ラッシュ時データ)"""

    # (station_id, name, city, temp, humidity, co2, noise)
    _PRESETS = [
        # 東京 — ラッシュ時は CO2・騒音が高め
        ("tokyo-shinjuku",     "新宿",    CityName.TOKYO,    32.5, 78.0, 1800.0, 88.0),
        ("tokyo-tokyo",        "東京",    CityName.TOKYO,    31.0, 72.0, 1500.0, 84.0),
        ("tokyo-ginza",        "銀座",    CityName.TOKYO,    29.5, 68.0, 1100.0, 78.0),
        ("tokyo-otemachi",     "大手町",  CityName.TOKYO,    27.0, 65.0,  900.0, 72.0),
        ("tokyo-kasumigaseki", "霞ケ関",  CityName.TOKYO,    25.5, 60.0,  800.0, 70.0),
        ("tokyo-ogikubo",      "荻窪",    CityName.TOKYO,    30.0, 70.0, 1200.0, 80.0),
        ("tokyo-nakano",       "中野",    CityName.TOKYO,    28.5, 66.0,  950.0, 74.0),
        ("tokyo-yotsuya",      "四ツ谷",  CityName.TOKYO,    26.0, 62.0,  820.0, 71.0),
        # 大阪
        ("osaka-namba",        "なんば",  CityName.OSAKA,    33.0, 80.0, 1950.0, 90.0),
        ("osaka-shinsaibashi", "心斎橋",  CityName.OSAKA,    31.5, 75.0, 1400.0, 82.0),
        ("osaka-senri-chuo",   "千里中央",CityName.OSAKA,    27.5, 63.0,  870.0, 73.0),
        # 名古屋
        ("nagoya-nagoya",      "名古屋",  CityName.NAGOYA,   30.5, 71.0, 1300.0, 81.0),
        ("nagoya-sakae",       "栄",      CityName.NAGOYA,   28.0, 64.0,  920.0, 75.0),
        # 横浜
        ("yokohama-yokohama",  "横浜",    CityName.YOKOHAMA, 29.0, 69.0, 1050.0, 77.0),
        # 福岡
        ("fukuoka-hakata",     "博多",    CityName.FUKUOKA,  34.0, 82.0, 2100.0, 92.0),
        ("fukuoka-tenjin",     "天神",    CityName.FUKUOKA,  31.0, 74.0, 1350.0, 83.0),
    ]

    @classmethod
    def build(cls) -> EnvSensorAnalyzer:
        analyzer = EnvSensorAnalyzer()
        for sid, name, city, temp, humidity, co2, noise in cls._PRESETS:
            readings = [
                SensorReading.create(SensorType.TEMPERATURE, temp),
                SensorReading.create(SensorType.HUMIDITY,    humidity),
                SensorReading.create(SensorType.CO2,         co2),
                SensorReading.create(SensorType.NOISE,       noise),
            ]
            analyzer.register(StationEnvironment(
                station_id=sid,
                station_name=name,
                city=city,
                readings=readings,
            ))
        return analyzer
