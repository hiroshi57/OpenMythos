"""
Sprint 72A — 地図比較ビュー (都市間断面比較)

2都市の地質層・駅深度を横並び SVG で比較する。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from open_mythos.skills.city_map import (
    CityName, GeologyLayer, Station, MetroLine, CityMapStore,
)


# ─── Config / Result ──────────────────────────────────────────────


@dataclass
class ComparisonConfig:
    """比較 SVG の描画設定"""
    width: int = 1400
    height: int = 600
    margin: int = 40
    label_height: int = 50
    max_depth_m: float = 60.0
    station_radius: int = 6
    font_size: int = 11

    @property
    def panel_width(self) -> int:
        """各都市パネルの幅"""
        return (self.width - self.margin * 3) // 2

    @property
    def panel_height(self) -> int:
        return self.height - self.margin * 2 - self.label_height

    def depth_to_y(self, depth_m: float, panel_top: int) -> float:
        ratio = min(depth_m / self.max_depth_m, 1.0)
        return panel_top + ratio * self.panel_height


@dataclass
class DepthStats:
    """駅深度統計"""
    city: str
    min_depth_m: float
    max_depth_m: float
    avg_depth_m: float
    station_count: int

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "min_depth_m": round(self.min_depth_m, 1),
            "max_depth_m": round(self.max_depth_m, 1),
            "avg_depth_m": round(self.avg_depth_m, 1),
            "station_count": self.station_count,
        }


@dataclass
class ComparisonResult:
    """比較結果"""
    city_a: str
    city_b: str
    svg: str
    stats_a: DepthStats
    stats_b: DepthStats
    deeper_city: str        # より深い駅を持つ都市
    geology_diff: List[dict]  # 地質層の差分サマリー

    def to_dict(self) -> dict:
        return {
            "city_a": self.city_a,
            "city_b": self.city_b,
            "svg": self.svg,
            "stats_a": self.stats_a.to_dict(),
            "stats_b": self.stats_b.to_dict(),
            "deeper_city": self.deeper_city,
            "geology_diff": self.geology_diff,
        }


# ─── SVG helpers ──────────────────────────────────────────────────


def _el(tag: str, **attrs) -> ET.Element:
    e = ET.Element(tag)
    for k, v in attrs.items():
        e.set(k.replace("_", "-"), str(v))
    return e


def _txt(parent: ET.Element, x: float, y: float, text: str,
         font_size: int = 11, anchor: str = "middle",
         fill: str = "#333", **extra) -> None:
    e = _el("text", x=x, y=y, text_anchor=anchor,
            font_size=font_size, font_family="Arial,sans-serif", fill=fill)
    for k, v in extra.items():
        e.set(k.replace("_", "-"), str(v))
    e.text = text
    parent.append(e)


# ─── MapComparator ────────────────────────────────────────────────


class MapComparator:
    """2都市の地図データを比較してSVGを生成するクラス"""

    def __init__(
        self,
        store: CityMapStore,
        config: Optional[ComparisonConfig] = None,
    ) -> None:
        self._store = store
        self._cfg = config or ComparisonConfig()

    def compare(
        self,
        city_a: CityName,
        city_b: CityName,
    ) -> ComparisonResult:
        """2都市を比較した ComparisonResult を返す"""
        stations_a = self._store.stations.list_by_city(city_a)
        stations_b = self._store.stations.list_by_city(city_b)
        geology_a = self._store.geology.list_by_city(city_a)
        geology_b = self._store.geology.list_by_city(city_b)

        stats_a = self._calc_stats(city_a.value, stations_a)
        stats_b = self._calc_stats(city_b.value, stations_b)

        deeper = city_a.value if stats_a.max_depth_m >= stats_b.max_depth_m else city_b.value
        geo_diff = self._geology_diff(city_a.value, geology_a, city_b.value, geology_b)

        svg = self._render_svg(
            city_a.value, stations_a, geology_a,
            city_b.value, stations_b, geology_b,
        )

        return ComparisonResult(
            city_a=city_a.value,
            city_b=city_b.value,
            svg=svg,
            stats_a=stats_a,
            stats_b=stats_b,
            deeper_city=deeper,
            geology_diff=geo_diff,
        )

    # ── stats ──────────────────────────────────────────────────────

    def _calc_stats(self, city: str, stations: List[Station]) -> DepthStats:
        if not stations:
            return DepthStats(city, 0.0, 0.0, 0.0, 0)
        depths = [s.depth_m for s in stations]
        return DepthStats(
            city=city,
            min_depth_m=min(depths),
            max_depth_m=max(depths),
            avg_depth_m=sum(depths) / len(depths),
            station_count=len(stations),
        )

    def _geology_diff(
        self,
        city_a: str, layers_a: List[GeologyLayer],
        city_b: str, layers_b: List[GeologyLayer],
    ) -> List[dict]:
        result = []
        types_a = {gl.layer_type.value: gl for gl in layers_a}
        types_b = {gl.layer_type.value: gl for gl in layers_b}
        all_types = sorted(set(types_a) | set(types_b))
        for t in all_types:
            ga = types_a.get(t)
            gb = types_b.get(t)
            result.append({
                "layer_type": t,
                city_a: {"depth_from_m": ga.depth_from_m, "thickness_m": ga.thickness_m} if ga else None,
                city_b: {"depth_from_m": gb.depth_from_m, "thickness_m": gb.thickness_m} if gb else None,
            })
        return result

    # ── SVG render ─────────────────────────────────────────────────

    def _render_svg(
        self,
        name_a: str, stations_a: List[Station], geology_a: List[GeologyLayer],
        name_b: str, stations_b: List[Station], geology_b: List[GeologyLayer],
    ) -> str:
        cfg = self._cfg
        root = _el("svg", xmlns="http://www.w3.org/2000/svg",
                   width=cfg.width, height=cfg.height,
                   viewBox=f"0 0 {cfg.width} {cfg.height}")
        root.append(_el("rect", x=0, y=0, width=cfg.width, height=cfg.height,
                        fill="#FAFAFA"))

        # タイトル
        _txt(root, cfg.width / 2, 28,
             f"{name_a.upper()}  vs  {name_b.upper()} — 断面比較",
             font_size=15, fill="#222", font_weight="bold")

        panel_top = cfg.margin + cfg.label_height

        # 左パネル (city_a)
        x_a = cfg.margin
        self._render_panel(root, name_a, stations_a, geology_a, x_a, panel_top)

        # 右パネル (city_b)
        x_b = cfg.margin * 2 + cfg.panel_width
        self._render_panel(root, name_b, stations_b, geology_b, x_b, panel_top)

        # 中央区切り線
        cx = cfg.margin + cfg.panel_width + cfg.margin // 2
        root.append(_el("line", x1=cx, y1=panel_top - 10,
                        x2=cx, y2=panel_top + cfg.panel_height + 10,
                        stroke="#CCC", stroke_width=1, stroke_dasharray="4,3"))

        return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")

    def _render_panel(
        self,
        root: ET.Element, city_name: str,
        stations: List[Station], geology: List[GeologyLayer],
        x0: int, panel_top: int,
    ) -> None:
        cfg = self._cfg
        w = cfg.panel_width
        surface_y = panel_top

        # 都市名ラベル
        _txt(root, x0 + w / 2, panel_top - 8, city_name.upper(),
             font_size=13, fill="#444", font_weight="bold")

        # 地質層
        for gl in sorted(geology, key=lambda g: g.depth_from_m):
            y0 = cfg.depth_to_y(gl.depth_from_m, panel_top)
            y1 = cfg.depth_to_y(min(gl.depth_to_m, cfg.max_depth_m), panel_top)
            if y1 > y0:
                root.append(_el("rect", x=x0, y=y0, width=w, height=y1 - y0,
                                fill=gl.color, opacity=0.6,
                                stroke="#CCC", stroke_width=0.5))
                if y1 - y0 > 12:
                    _txt(root, x0 + 4, y0 + min((y1 - y0) / 2, 12),
                         gl.name[:8], font_size=8, anchor="start", fill="#555")

        # 地表線
        root.append(_el("line", x1=x0, y1=surface_y,
                        x2=x0 + w, y2=surface_y,
                        stroke="#444", stroke_width=2))

        # 駅
        n = len(stations)
        if n > 0:
            x_step = w / max(n, 1)
            for i, st in enumerate(sorted(stations, key=lambda s: s.id)):
                sx = x0 + x_step * i + x_step / 2
                sy = cfg.depth_to_y(st.depth_m, panel_top)
                # 垂直線
                root.append(_el("line", x1=sx, y1=surface_y, x2=sx, y2=sy - cfg.station_radius,
                                stroke="#AAA", stroke_width=1, stroke_dasharray="2,2"))
                # 駅円
                root.append(_el("circle", cx=sx, cy=sy, r=cfg.station_radius,
                                fill="white", stroke="#333", stroke_width=1.5))
                # 深度
                _txt(root, sx, sy + 3, f"{st.depth_m:.0f}m", font_size=7, fill="#333")

        # 深度軸
        for d in range(0, int(cfg.max_depth_m) + 1, 10):
            y = cfg.depth_to_y(d, panel_top)
            _txt(root, x0 - 4, y + 3, f"{d}m",
                 font_size=8, anchor="end", fill="#888")
            root.append(_el("line", x1=x0 - 2, y1=y, x2=x0, y2=y,
                            stroke="#BBB", stroke_width=1))
