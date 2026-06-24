"""
Sprint 75B — 乗り換え最適化

混雑状況・アクセシビリティ・所要時間を統合し、
利用者特性に応じた乗り換えルートを最適化する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from open_mythos.skills.accessibility import AccessibilityAnalyzer
from open_mythos.skills.crowd_simulator import CrowdSimulator
from open_mythos.skills.route_finder import RouteGraph, RouteGraphBuilder, RouteFinder, RouteResult
from open_mythos.skills.city_map import CityMapStore


# ─── Weight Presets ───────────────────────────────────────────────


@dataclass
class OptimizationWeight:
    """統合スコア計算の重み設定 (合計 1.0)"""
    crowd_w: float = 0.4
    access_w: float = 0.3
    time_w:  float = 0.3
    label:   str   = "balanced"

    def __post_init__(self) -> None:
        total = self.crowd_w + self.access_w + self.time_w
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "crowd_w": self.crowd_w,
            "access_w": self.access_w,
            "time_w": self.time_w,
        }


# 標準プリセット
WEIGHT_TIME_FIRST        = OptimizationWeight(0.1, 0.1, 0.8, "time_first")
WEIGHT_BALANCED          = OptimizationWeight(0.4, 0.3, 0.3, "balanced")
WEIGHT_ACCESSIBILITY     = OptimizationWeight(0.2, 0.6, 0.2, "accessibility")
WEIGHT_CROWD_AVOIDANCE   = OptimizationWeight(0.6, 0.2, 0.2, "crowd_avoidance")

DEFAULT_WEIGHT_PRESETS = [
    WEIGHT_TIME_FIRST,
    WEIGHT_BALANCED,
    WEIGHT_ACCESSIBILITY,
]


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class TransferOption:
    """乗り換えオプション (重みプリセット別)"""
    label:        str              # プリセット名
    weight:       OptimizationWeight
    route:        RouteResult      # RouteFinder の結果
    path_ids:     List[str]        # 経路の station_id リスト
    crowd_cost:   float            # 混雑コスト (0〜1)
    access_cost:  float            # アクセシビリティコスト (0〜1、低いほど良い)
    time_cost:    float            # 所要時間コスト (正規化後 0〜1)
    total_cost:   float            # 統合コスト (低いほど優秀)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "weight": self.weight.to_dict(),
            "path_ids": self.path_ids,
            "crowd_cost": round(self.crowd_cost, 3),
            "access_cost": round(self.access_cost, 3),
            "time_cost": round(self.time_cost, 3),
            "total_cost": round(self.total_cost, 3),
            "route_summary": self.route.message if self.route.found else "経路なし",
            "transfer_count": self.route.transfer_count,
            "raw_time_cost": round(self.route.total_cost, 2),
        }


# ─── TransferOptimizer ────────────────────────────────────────────

# 時間コスト正規化の基準値 (分)
_TIME_COST_MAX = 50.0


class TransferOptimizer:
    """
    混雑・アクセシビリティ・時間を統合して乗り換え最適化を行うクラス。

    同一経路を複数の重みプリセットでスコアリングし、
    プリセット別のコスト比較を提供する。
    """

    def __init__(
        self,
        crowd: CrowdSimulator,
        analyzer: AccessibilityAnalyzer,
        finder: RouteFinder,
        weight_presets: Optional[List[OptimizationWeight]] = None,
    ) -> None:
        self._crowd    = crowd
        self._analyzer = analyzer
        self._finder   = finder
        self._presets  = weight_presets or DEFAULT_WEIGHT_PRESETS

    # ── public ──────────────────────────────────────────────────

    def optimize(
        self,
        from_id: str,
        to_id: str,
        hour: int,
    ) -> List[TransferOption]:
        """
        指定区間をデフォルトプリセット全てでスコアリングして返す。
        found=False の場合は空リストを返す。
        """
        route = self._finder.find(from_id, to_id)
        if not route.found:
            return []
        path_ids = [step.station_id for step in route.steps]
        crowd_cost  = self._calc_crowd_cost(path_ids, hour)
        access_cost = self._calc_access_cost(path_ids)
        time_cost   = self._calc_time_cost(route.total_cost)

        options = []
        for preset in self._presets:
            total = (
                preset.crowd_w * crowd_cost
                + preset.access_w * access_cost
                + preset.time_w  * time_cost
            )
            options.append(TransferOption(
                label=preset.label,
                weight=preset,
                route=route,
                path_ids=path_ids,
                crowd_cost=crowd_cost,
                access_cost=access_cost,
                time_cost=time_cost,
                total_cost=total,
            ))

        # total_cost 昇順でソート
        return sorted(options, key=lambda o: o.total_cost)

    def score_route(
        self,
        from_id: str,
        to_id: str,
        hour: int,
        weight: Optional[OptimizationWeight] = None,
    ) -> Optional[TransferOption]:
        """指定した重みで 1 経路をスコアリングして返す"""
        w = weight or WEIGHT_BALANCED
        route = self._finder.find(from_id, to_id)
        if not route.found:
            return None
        path_ids    = [step.station_id for step in route.steps]
        crowd_cost  = self._calc_crowd_cost(path_ids, hour)
        access_cost = self._calc_access_cost(path_ids)
        time_cost   = self._calc_time_cost(route.total_cost)
        total = (
            w.crowd_w  * crowd_cost
            + w.access_w * access_cost
            + w.time_w   * time_cost
        )
        return TransferOption(
            label=w.label,
            weight=w,
            route=route,
            path_ids=path_ids,
            crowd_cost=crowd_cost,
            access_cost=access_cost,
            time_cost=time_cost,
            total_cost=total,
        )

    # ── private ─────────────────────────────────────────────────

    def _calc_crowd_cost(self, path_ids: List[str], hour: int) -> float:
        """経由駅の平均 occupancy_rate を混雑コストとして返す (0〜1)"""
        rates = []
        for sid in path_ids:
            snap = self._crowd.snapshot(sid, hour)
            if snap is not None:
                rates.append(min(1.0, snap.occupancy_rate))
        return sum(rates) / len(rates) if rates else 0.5

    def _calc_access_cost(self, path_ids: List[str]) -> float:
        """
        経由駅の平均アクセシビリティスコアを逆転してコストへ変換する (0〜1)。
        スコアが低い (バリアフリー不足) ほどコストが高い。
        """
        scores = []
        for sid in path_ids:
            sc = self._analyzer.score(sid)
            if sc is not None:
                scores.append(sc.score / 100.0)
        avg_score = sum(scores) / len(scores) if scores else 0.5
        return 1.0 - avg_score  # スコア高 → コスト低

    def _calc_time_cost(self, raw_cost: float) -> float:
        """Dijkstra 総コストを [0, 1] に正規化する"""
        return min(1.0, raw_cost / _TIME_COST_MAX)


# ─── OptimizationDataset ──────────────────────────────────────────


class OptimizationDataset:
    """TransferOptimizer 用データセット (3 モジュール統合)"""

    @classmethod
    def build(cls) -> "OptimizationDataset":
        from open_mythos.skills.crowd_simulator import CrowdDataset
        from open_mythos.skills.accessibility import AccessibilityDataset
        from open_mythos.skills.city_map import CityMapDataset

        crowd    = CrowdDataset.build()
        analyzer = AccessibilityDataset.build()

        map_store = CityMapDataset.build()
        graph = RouteGraphBuilder.build(map_store)
        finder = RouteFinder(graph)

        return cls(crowd=crowd, analyzer=analyzer, finder=finder, graph=graph)

    def __init__(
        self,
        crowd: CrowdSimulator,
        analyzer: AccessibilityAnalyzer,
        finder: RouteFinder,
        graph: RouteGraph,
    ) -> None:
        self.crowd    = crowd
        self.analyzer = analyzer
        self.finder   = finder
        self.graph    = graph

    def optimizer(
        self,
        presets: Optional[List[OptimizationWeight]] = None,
    ) -> TransferOptimizer:
        return TransferOptimizer(
            crowd=self.crowd,
            analyzer=self.analyzer,
            finder=self.finder,
            weight_presets=presets,
        )
