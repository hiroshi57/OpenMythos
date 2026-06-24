"""
Sprint 74B — アクセシビリティ分析

駅のエレベーター/バリアフリー度スコアを計算する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from open_mythos.skills.city_map import CityName


# ─── Enums ────────────────────────────────────────────────────────


class AccessFeature(str, Enum):
    ELEVATOR = "elevator"
    ESCALATOR = "escalator"
    TACTILE_PAVING = "tactile_paving"       # 点字ブロック
    ACCESSIBLE_TOILET = "accessible_toilet"  # 多機能トイレ
    WHEELCHAIR_RAMP = "wheelchair_ramp"
    AUDIO_GUIDE = "audio_guide"             # 音声案内
    BRAILLE_SIGNAGE = "braille_signage"     # 点字サイン
    WIDE_GATE = "wide_gate"                 # 幅広改札


class AccessLevel(str, Enum):
    EXCELLENT = "excellent"   # 90〜100
    GOOD = "good"             # 70〜89
    FAIR = "fair"             # 50〜69
    POOR = "poor"             # 30〜49
    CRITICAL = "critical"     # 0〜29


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class AccessibilityProfile:
    """駅のアクセシビリティプロファイル"""
    station_id: str
    station_name: str
    city: CityName
    features: List[AccessFeature] = field(default_factory=list)
    note: str = ""

    def has_feature(self, feature: AccessFeature) -> bool:
        return feature in self.features

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "city": self.city.value,
            "features": [f.value for f in self.features],
            "note": self.note,
        }


@dataclass
class AccessibilityScore:
    """アクセシビリティスコア"""
    station_id: str
    station_name: str
    score: float                 # 0〜100
    level: AccessLevel
    features_present: List[str]
    features_missing: List[str]
    breakdown: Dict[str, float]  # feature → 寄与スコア

    def to_dict(self) -> dict:
        return {
            "station_id": self.station_id,
            "station_name": self.station_name,
            "score": round(self.score, 1),
            "level": self.level.value,
            "features_present": self.features_present,
            "features_missing": self.features_missing,
            "breakdown": {k: round(v, 1) for k, v in self.breakdown.items()},
        }


@dataclass
class AccessibilityReport:
    """アクセシビリティレポート"""
    city: str
    scores: List[AccessibilityScore]

    @property
    def average_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    @property
    def best_station(self) -> Optional[AccessibilityScore]:
        return max(self.scores, key=lambda s: s.score) if self.scores else None

    @property
    def worst_station(self) -> Optional[AccessibilityScore]:
        return min(self.scores, key=lambda s: s.score) if self.scores else None

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "average_score": round(self.average_score, 1),
            "best_station": self.best_station.to_dict() if self.best_station else None,
            "worst_station": self.worst_station.to_dict() if self.worst_station else None,
            "scores": [s.to_dict() for s in self.scores],
        }


# ─── Scoring ──────────────────────────────────────────────────────

# 各機能の重み (合計100)
_FEATURE_WEIGHTS: Dict[AccessFeature, float] = {
    AccessFeature.ELEVATOR:          25.0,
    AccessFeature.WHEELCHAIR_RAMP:   20.0,
    AccessFeature.ACCESSIBLE_TOILET: 15.0,
    AccessFeature.TACTILE_PAVING:    15.0,
    AccessFeature.WIDE_GATE:         10.0,
    AccessFeature.ESCALATOR:          7.0,
    AccessFeature.AUDIO_GUIDE:        5.0,
    AccessFeature.BRAILLE_SIGNAGE:    3.0,
}


def _score_to_level(score: float) -> AccessLevel:
    if score >= 90:
        return AccessLevel.EXCELLENT
    elif score >= 70:
        return AccessLevel.GOOD
    elif score >= 50:
        return AccessLevel.FAIR
    elif score >= 30:
        return AccessLevel.POOR
    return AccessLevel.CRITICAL


# ─── AccessibilityAnalyzer ────────────────────────────────────────


class AccessibilityAnalyzer:
    """アクセシビリティスコアを計算するアナライザー"""

    def __init__(self) -> None:
        self._profiles: Dict[str, AccessibilityProfile] = {}

    def register(self, profile: AccessibilityProfile) -> None:
        self._profiles[profile.station_id] = profile

    def get_profile(self, station_id: str) -> Optional[AccessibilityProfile]:
        return self._profiles.get(station_id)

    def score(self, station_id: str) -> Optional[AccessibilityScore]:
        profile = self._profiles.get(station_id)
        if profile is None:
            return None

        breakdown: Dict[str, float] = {}
        total = 0.0
        present = []
        missing = []

        for feature, weight in _FEATURE_WEIGHTS.items():
            if profile.has_feature(feature):
                breakdown[feature.value] = weight
                total += weight
                present.append(feature.value)
            else:
                breakdown[feature.value] = 0.0
                missing.append(feature.value)

        return AccessibilityScore(
            station_id=station_id,
            station_name=profile.station_name,
            score=total,
            level=_score_to_level(total),
            features_present=present,
            features_missing=missing,
            breakdown=breakdown,
        )

    def city_report(self, city: CityName) -> AccessibilityReport:
        city_profiles = [
            p for p in self._profiles.values() if p.city == city
        ]
        scores = [self.score(p.station_id) for p in city_profiles]
        return AccessibilityReport(
            city=city.value,
            scores=[s for s in scores if s is not None],
        )

    def rank(self, city: Optional[CityName] = None) -> List[AccessibilityScore]:
        """スコア降順でランキングを返す"""
        if city is not None:
            ids = [p.station_id for p in self._profiles.values() if p.city == city]
        else:
            ids = list(self._profiles.keys())
        scores = [self.score(sid) for sid in ids]
        return sorted(
            [s for s in scores if s is not None],
            key=lambda x: x.score, reverse=True,
        )

    def all_station_ids(self) -> List[str]:
        return list(self._profiles.keys())


# ─── DefaultDataset ───────────────────────────────────────────────


class AccessibilityDataset:
    """主要駅のアクセシビリティプリセット"""

    ALL = [
        AccessFeature.ELEVATOR, AccessFeature.ESCALATOR,
        AccessFeature.TACTILE_PAVING, AccessFeature.ACCESSIBLE_TOILET,
        AccessFeature.WHEELCHAIR_RAMP, AccessFeature.AUDIO_GUIDE,
        AccessFeature.BRAILLE_SIGNAGE, AccessFeature.WIDE_GATE,
    ]
    BASIC = [
        AccessFeature.ELEVATOR, AccessFeature.TACTILE_PAVING,
        AccessFeature.WIDE_GATE,
    ]
    MINIMAL = [AccessFeature.TACTILE_PAVING]

    _PRESETS = [
        # (station_id, name, city, features_preset)
        ("tokyo-shinjuku",     "新宿",    CityName.TOKYO,    "all"),
        ("tokyo-tokyo",        "東京",    CityName.TOKYO,    "all"),
        ("tokyo-ginza",        "銀座",    CityName.TOKYO,    "all"),
        ("tokyo-otemachi",     "大手町",  CityName.TOKYO,    "basic"),
        ("tokyo-kasumigaseki", "霞ケ関",  CityName.TOKYO,    "basic"),
        ("tokyo-ogikubo",      "荻窪",    CityName.TOKYO,    "basic"),
        ("tokyo-nakano",       "中野",    CityName.TOKYO,    "minimal"),
        ("tokyo-yotsuya",      "四ツ谷",  CityName.TOKYO,    "minimal"),
        ("osaka-namba",        "なんば",  CityName.OSAKA,    "all"),
        ("osaka-shinsaibashi", "心斎橋",  CityName.OSAKA,    "basic"),
        ("osaka-senri-chuo",   "千里中央",CityName.OSAKA,    "minimal"),
        ("nagoya-nagoya",      "名古屋",  CityName.NAGOYA,   "all"),
        ("nagoya-sakae",       "栄",      CityName.NAGOYA,   "basic"),
        ("yokohama-yokohama",  "横浜",    CityName.YOKOHAMA, "all"),
        ("fukuoka-hakata",     "博多",    CityName.FUKUOKA,  "all"),
        ("fukuoka-tenjin",     "天神",    CityName.FUKUOKA,  "basic"),
    ]

    @classmethod
    def build(cls) -> AccessibilityAnalyzer:
        analyzer = AccessibilityAnalyzer()
        feat_map = {"all": cls.ALL, "basic": cls.BASIC, "minimal": cls.MINIMAL}
        for sid, name, city, preset in cls._PRESETS:
            analyzer.register(AccessibilityProfile(
                station_id=sid,
                station_name=name,
                city=city,
                features=list(feat_map[preset]),
            ))
        return analyzer
