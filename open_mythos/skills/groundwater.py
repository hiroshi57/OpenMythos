"""
Sprint 74C — 地下水位モニタリング

地質層と水位データを重ね合わせた浸水リスク分析。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from open_mythos.skills.city_map import CityName, GeologyLayerType


# ─── Enums ────────────────────────────────────────────────────────


class WaterLevelZone(str, Enum):
    SURFACE = "surface"          # 地表〜2m
    SHALLOW = "shallow"          # 2〜10m
    INTERMEDIATE = "intermediate"  # 10〜30m
    DEEP = "deep"                # 30m以深


class FloodRiskLevel(str, Enum):
    VERY_HIGH = "very_high"    # スコア 80+
    HIGH = "high"              # 60〜79
    MODERATE = "moderate"      # 40〜59
    LOW = "low"                # 20〜39
    VERY_LOW = "very_low"      # 0〜19


class Season(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class GroundwaterLayer:
    """地下水位層データ"""
    id: str
    city: CityName
    zone: WaterLevelZone
    depth_m: float              # 水位面の深度 (m)
    pressure_kpa: float         # 水圧 (kPa)
    seasonal_variation_m: float = 1.0  # 季節変動幅 (m)
    artesian: bool = False      # 被圧地下水か

    def season_depth(self, season: Season) -> float:
        """季節による深度調整"""
        delta = {
            Season.SPRING: -0.5,   # 融雪で水位上昇 → 深度減
            Season.SUMMER:  0.3,   # 渇水で水位低下 → 深度増
            Season.AUTUMN:  0.0,
            Season.WINTER: -0.2,
        }
        return max(0.1, self.depth_m + delta.get(season, 0.0))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "city": self.city.value,
            "zone": self.zone.value,
            "depth_m": self.depth_m,
            "pressure_kpa": self.pressure_kpa,
            "seasonal_variation_m": self.seasonal_variation_m,
            "artesian": self.artesian,
        }


@dataclass
class FloodRiskFactor:
    """浸水リスク要因"""
    name: str
    score: float          # 0〜100 の部分スコア
    weight: float         # 重み
    description: str

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "weight": self.weight,
            "weighted_score": round(self.weighted_score, 1),
            "description": self.description,
        }


@dataclass
class FloodRiskResult:
    """浸水リスク評価結果"""
    city: str
    station_id: Optional[str]
    station_depth_m: Optional[float]  # 駅の地下深度
    total_score: float
    level: FloodRiskLevel
    factors: List[FloodRiskFactor]
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "station_id": self.station_id,
            "station_depth_m": self.station_depth_m,
            "total_score": round(self.total_score, 1),
            "level": self.level.value,
            "factors": [f.to_dict() for f in self.factors],
            "recommendation": self.recommendation,
        }


@dataclass
class CityWaterProfile:
    """都市の地下水プロファイル"""
    city: CityName
    layers: List[GroundwaterLayer]
    monitoring_stations: int = 0

    def shallowest_layer(self) -> Optional[GroundwaterLayer]:
        if not self.layers:
            return None
        return min(self.layers, key=lambda l: l.depth_m)

    def to_dict(self) -> dict:
        return {
            "city": self.city.value,
            "layers": [l.to_dict() for l in self.layers],
            "monitoring_stations": self.monitoring_stations,
            "shallowest_depth_m": self.shallowest_layer().depth_m if self.layers else None,
        }


# ─── Risk Scoring ─────────────────────────────────────────────────


def _score_to_level(score: float) -> FloodRiskLevel:
    if score >= 80:
        return FloodRiskLevel.VERY_HIGH
    elif score >= 60:
        return FloodRiskLevel.HIGH
    elif score >= 40:
        return FloodRiskLevel.MODERATE
    elif score >= 20:
        return FloodRiskLevel.LOW
    return FloodRiskLevel.VERY_LOW


def _recommendation(level: FloodRiskLevel) -> str:
    return {
        FloodRiskLevel.VERY_HIGH: "緊急対策が必要。防水壁・排水ポンプの点検と補強を推奨。",
        FloodRiskLevel.HIGH:      "定期的な水位モニタリングと防水対策の強化を推奨。",
        FloodRiskLevel.MODERATE:  "標準的な防水基準を維持。雨季前の点検を推奨。",
        FloodRiskLevel.LOW:       "現状維持で問題なし。年1回の定期点検を実施。",
        FloodRiskLevel.VERY_LOW:  "リスクは最小限。通常の維持管理で十分。",
    }[level]


# ─── FloodRiskAssessor ────────────────────────────────────────────


class FloodRiskAssessor:
    """地質層 + 水位データから浸水リスクを評価するクラス"""

    def __init__(self) -> None:
        self._profiles: Dict[str, CityWaterProfile] = {}

    def register(self, profile: CityWaterProfile) -> None:
        self._profiles[profile.city.value] = profile

    def get_profile(self, city: CityName) -> Optional[CityWaterProfile]:
        return self._profiles.get(city.value)

    def city_risk(
        self,
        city: CityName,
        season: Season = Season.SUMMER,
    ) -> Optional[FloodRiskResult]:
        profile = self._profiles.get(city.value)
        if profile is None:
            return None
        return self._assess(city.value, profile, None, None, season)

    def station_risk(
        self,
        city: CityName,
        station_id: str,
        station_depth_m: float,
        season: Season = Season.SUMMER,
    ) -> Optional[FloodRiskResult]:
        profile = self._profiles.get(city.value)
        if profile is None:
            return None
        return self._assess(city.value, profile, station_id, station_depth_m, season)

    def _assess(
        self,
        city_name: str,
        profile: CityWaterProfile,
        station_id: Optional[str],
        station_depth_m: Optional[float],
        season: Season,
    ) -> FloodRiskResult:
        factors: List[FloodRiskFactor] = []

        # 1. 水位の浅さ (最浅層)
        shallowest = profile.shallowest_layer()
        if shallowest:
            adj_depth = shallowest.season_depth(season)
            # 深いほどリスク低 (0m→100, 20m→0)
            depth_score = max(0.0, 100.0 - adj_depth * 5.0)
            factors.append(FloodRiskFactor(
                name="water_depth",
                score=depth_score,
                weight=0.35,
                description=f"最浅水位: {adj_depth:.1f}m ({season.value})",
            ))

        # 2. 被圧地下水の存在
        artesian_count = sum(1 for l in profile.layers if l.artesian)
        artesian_score = min(100.0, artesian_count * 40.0)
        factors.append(FloodRiskFactor(
            name="artesian_pressure",
            score=artesian_score,
            weight=0.25,
            description=f"被圧地下水帯水層: {artesian_count}層",
        ))

        # 3. 駅の深度 vs 水位 (駅固有)
        if station_depth_m is not None and shallowest:
            adj_depth = shallowest.season_depth(season)
            margin = station_depth_m - adj_depth
            # マージンが小さいほどリスク高
            proximity_score = max(0.0, min(100.0, (20.0 - margin) * 5.0))
            factors.append(FloodRiskFactor(
                name="station_water_margin",
                score=proximity_score,
                weight=0.30,
                description=f"駅深度({station_depth_m}m) - 水位({adj_depth:.1f}m) = {margin:.1f}m",
            ))
        else:
            # 都市全体評価: 水位ゾーンによるスコア
            zone_scores = {
                WaterLevelZone.SURFACE:      90.0,
                WaterLevelZone.SHALLOW:      60.0,
                WaterLevelZone.INTERMEDIATE: 30.0,
                WaterLevelZone.DEEP:         10.0,
            }
            zone_score = max(
                (zone_scores.get(l.zone, 0) for l in profile.layers),
                default=0.0,
            )
            factors.append(FloodRiskFactor(
                name="zone_risk",
                score=zone_score,
                weight=0.30,
                description=f"最高リスクゾーン: {zone_score:.0f}点",
            ))

        # 4. 季節変動リスク
        max_var = max((l.seasonal_variation_m for l in profile.layers), default=0.0)
        variation_score = min(100.0, max_var * 20.0)
        factors.append(FloodRiskFactor(
            name="seasonal_variation",
            score=variation_score,
            weight=0.10,
            description=f"最大季節変動: {max_var:.1f}m",
        ))

        total = sum(f.weighted_score for f in factors)
        level = _score_to_level(total)

        return FloodRiskResult(
            city=city_name,
            station_id=station_id,
            station_depth_m=station_depth_m,
            total_score=total,
            level=level,
            factors=factors,
            recommendation=_recommendation(level),
        )

    def city_names(self) -> List[str]:
        return list(self._profiles.keys())


# ─── DefaultDataset ───────────────────────────────────────────────


class GroundwaterDataset:
    """5都市の地下水プリセット"""

    @classmethod
    def build(cls) -> FloodRiskAssessor:
        assessor = FloodRiskAssessor()
        for city, layers, mon in cls._data():
            assessor.register(CityWaterProfile(
                city=city, layers=layers, monitoring_stations=mon,
            ))
        return assessor

    @classmethod
    def _data(cls):
        return [
            (CityName.TOKYO, [
                GroundwaterLayer("tky-w1", CityName.TOKYO, WaterLevelZone.SHALLOW,
                                 4.5, 30.0, 1.5, False),
                GroundwaterLayer("tky-w2", CityName.TOKYO, WaterLevelZone.INTERMEDIATE,
                                 18.0, 120.0, 0.5, True),
            ], 42),
            (CityName.OSAKA, [
                GroundwaterLayer("osk-w1", CityName.OSAKA, WaterLevelZone.SHALLOW,
                                 2.8, 45.0, 2.0, False),
                GroundwaterLayer("osk-w2", CityName.OSAKA, WaterLevelZone.INTERMEDIATE,
                                 14.0, 200.0, 1.0, True),
            ], 35),
            (CityName.NAGOYA, [
                GroundwaterLayer("ngy-w1", CityName.NAGOYA, WaterLevelZone.SHALLOW,
                                 6.0, 25.0, 1.2, False),
                GroundwaterLayer("ngy-w2", CityName.NAGOYA, WaterLevelZone.INTERMEDIATE,
                                 22.0, 90.0, 0.4, False),
            ], 28),
            (CityName.YOKOHAMA, [
                GroundwaterLayer("yok-w1", CityName.YOKOHAMA, WaterLevelZone.SHALLOW,
                                 5.5, 35.0, 1.0, False),
                GroundwaterLayer("yok-w2", CityName.YOKOHAMA, WaterLevelZone.DEEP,
                                 35.0, 180.0, 0.3, True),
            ], 22),
            (CityName.FUKUOKA, [
                GroundwaterLayer("fuk-w1", CityName.FUKUOKA, WaterLevelZone.SHALLOW,
                                 3.5, 40.0, 1.8, False),
                GroundwaterLayer("fuk-w2", CityName.FUKUOKA, WaterLevelZone.INTERMEDIATE,
                                 16.0, 150.0, 0.7, True),
            ], 18),
        ]
