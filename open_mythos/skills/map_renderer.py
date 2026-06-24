"""
Sprint 71B — 3D断面図 SVG レンダラー

station/geology データから SVG 断面図を生成する。
外部ライブラリなし (標準 xml.etree.ElementTree のみ)。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

from open_mythos.skills.city_map import (
    CityName, GeologyLayer, MetroLine, Station, CityMapStore,
)


# ─── Config ────────────────────────────────────────────────────────


@dataclass
class CrossSectionConfig:
    """断面図レンダリング設定"""
    width: int = 1200
    height: int = 600
    margin_top: int = 60
    margin_bottom: int = 40
    margin_left: int = 80
    margin_right: int = 40
    max_depth_m: float = 60.0       # 表示する最大深度 (m)
    surface_y_offset: int = 80      # 地表線の Y 位置 (margin_top からのオフセット)
    station_radius: int = 8
    font_size: int = 11
    title_font_size: int = 16
    show_depth_axis: bool = True
    show_legend: bool = True

    @property
    def plot_width(self) -> int:
        return self.width - self.margin_left - self.margin_right

    @property
    def plot_height(self) -> int:
        return self.height - self.margin_top - self.margin_bottom

    @property
    def surface_y(self) -> int:
        return self.margin_top + self.surface_y_offset

    def depth_to_y(self, depth_m: float) -> float:
        """深度 (m) → SVG Y 座標に変換"""
        ratio = depth_m / self.max_depth_m
        available = self.height - self.surface_y - self.margin_bottom
        return self.surface_y + ratio * available


@dataclass
class CrossSectionResult:
    """断面図生成結果"""
    city: str
    line_id: str
    svg: str                    # SVG XML 文字列
    station_count: int
    geology_count: int
    width: int
    height: int
    format: str = "svg"

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "line_id": self.line_id,
            "svg": self.svg,
            "station_count": self.station_count,
            "geology_count": self.geology_count,
            "width": self.width,
            "height": self.height,
            "format": self.format,
        }


# ─── SVG Builder helpers ──────────────────────────────────────────


def _svg_el(tag: str, **attrs) -> ET.Element:
    el = ET.Element(tag)
    for k, v in attrs.items():
        el.set(k.replace("_", "-"), str(v))
    return el


def _svg_text(parent: ET.Element, x: float, y: float, text: str,
              font_size: int = 11, anchor: str = "middle",
              fill: str = "#333", **extra) -> None:
    el = _svg_el("text", x=x, y=y,
                 text_anchor=anchor, font_size=font_size,
                 font_family="Arial,sans-serif", fill=fill)
    for k, v in extra.items():
        el.set(k.replace("_", "-"), str(v))
    el.text = text
    parent.append(el)


# ─── Renderer ─────────────────────────────────────────────────────


class SVGCrossSectionRenderer:
    """地下鉄断面図 SVG レンダラー"""

    def __init__(self, config: Optional[CrossSectionConfig] = None) -> None:
        self.config = config or CrossSectionConfig()

    def render(
        self,
        line: MetroLine,
        stations: List[Station],
        geology_layers: List[GeologyLayer],
        title: Optional[str] = None,
    ) -> str:
        """SVG 文字列を返す"""
        cfg = self.config
        root = _svg_el(
            "svg",
            xmlns="http://www.w3.org/2000/svg",
            width=cfg.width,
            height=cfg.height,
            viewBox=f"0 0 {cfg.width} {cfg.height}",
        )

        # 背景
        root.append(_svg_el("rect", x=0, y=0,
                             width=cfg.width, height=cfg.height,
                             fill="#FAFAFA"))

        # 地質層
        self._render_geology(root, geology_layers)

        # 地表線
        root.append(_svg_el(
            "line",
            x1=cfg.margin_left, y1=cfg.surface_y,
            x2=cfg.width - cfg.margin_right, y2=cfg.surface_y,
            stroke="#555", stroke_width=2,
        ))

        # 駅とトンネル
        if stations:
            self._render_stations(root, stations)

        # 深度軸
        if cfg.show_depth_axis:
            self._render_depth_axis(root)

        # タイトル
        t = title or f"{line.name} ({line.name_en}) — 断面図"
        _svg_text(root, cfg.width / 2, 35, t,
                  font_size=cfg.title_font_size, fill="#222")

        # 凡例
        if cfg.show_legend and geology_layers:
            self._render_legend(root, geology_layers)

        return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
            root, encoding="unicode"
        )

    def _render_geology(
        self, root: ET.Element, layers: List[GeologyLayer]
    ) -> None:
        cfg = self.config
        sorted_layers = sorted(layers, key=lambda l: l.depth_from_m)
        x0 = cfg.margin_left
        w = cfg.plot_width

        for gl in sorted_layers:
            y0 = cfg.depth_to_y(gl.depth_from_m)
            y1 = cfg.depth_to_y(min(gl.depth_to_m, cfg.max_depth_m))
            if y1 <= y0:
                continue
            h = y1 - y0
            root.append(_svg_el(
                "rect", x=x0, y=y0, width=w, height=h,
                fill=gl.color, opacity=0.65,
                stroke="#CCC", stroke_width=0.5,
            ))
            # 層名ラベル (薄く)
            if h > 14:
                _svg_text(root, x0 + 6, y0 + min(h / 2, 14),
                          gl.name, font_size=9, anchor="start", fill="#555")

    def _render_stations(
        self, root: ET.Element, stations: List[Station]
    ) -> None:
        cfg = self.config
        n = len(stations)
        if n == 0:
            return

        x_start = cfg.margin_left + 30
        x_end = cfg.width - cfg.margin_right - 30
        step = (x_end - x_start) / max(n - 1, 1)

        # トンネル線
        if n > 1:
            pts = []
            for i, st in enumerate(stations):
                sx = x_start + i * step
                sy = cfg.depth_to_y(st.depth_m)
                pts.append(f"{sx:.1f},{sy:.1f}")
            root.append(_svg_el(
                "polyline",
                points=" ".join(pts),
                fill="none", stroke="#666", stroke_width=3,
                stroke_dasharray="6,3",
            ))

        # 駅円
        for i, st in enumerate(stations):
            sx = x_start + i * step
            sy = cfg.depth_to_y(st.depth_m)

            # 垂直線 (地表→駅)
            root.append(_svg_el(
                "line",
                x1=sx, y1=cfg.surface_y,
                x2=sx, y2=sy - cfg.station_radius,
                stroke="#999", stroke_width=1,
                stroke_dasharray="3,2",
            ))

            # 駅円
            root.append(_svg_el(
                "circle", cx=sx, cy=sy, r=cfg.station_radius,
                fill="white", stroke="#333", stroke_width=2,
            ))

            # 深度ラベル
            _svg_text(root, sx, sy + 4, f"{st.depth_m:.0f}m",
                      font_size=8, fill="#333")

            # 駅名
            angle = -45
            _svg_text(
                root, sx, cfg.surface_y - 8, st.name,
                font_size=cfg.font_size, fill="#222",
                transform=f"rotate({angle},{sx},{cfg.surface_y - 8})",
            )

    def _render_depth_axis(self, root: ET.Element) -> None:
        cfg = self.config
        x = cfg.margin_left - 8
        # 0, 10, 20, 30, 40, 50, 60 m
        for depth in range(0, int(cfg.max_depth_m) + 1, 10):
            y = cfg.depth_to_y(depth)
            _svg_text(root, x, y + 4, f"{depth}m",
                      font_size=9, anchor="end", fill="#666")
            root.append(_svg_el(
                "line",
                x1=cfg.margin_left - 4, y1=y,
                x2=cfg.margin_left, y2=y,
                stroke="#999", stroke_width=1,
            ))
        # 軸線
        root.append(_svg_el(
            "line",
            x1=cfg.margin_left, y1=cfg.surface_y,
            x2=cfg.margin_left, y2=cfg.height - cfg.margin_bottom,
            stroke="#999", stroke_width=1,
        ))

    def _render_legend(
        self, root: ET.Element, layers: List[GeologyLayer]
    ) -> None:
        cfg = self.config
        lx = cfg.width - cfg.margin_right - 180
        ly = cfg.margin_top + 5
        box_w, box_h = 12, 10
        row_h = 16

        _svg_text(root, lx, ly + 10, "地質凡例",
                  font_size=10, anchor="start", fill="#333",
                  font_weight="bold")

        seen: dict = {}
        sorted_layers = sorted(layers, key=lambda l: l.depth_from_m)
        for gl in sorted_layers:
            if gl.layer_type.value in seen:
                continue
            seen[gl.layer_type.value] = True
            row_y = ly + 20 + len(seen) * row_h
            root.append(_svg_el(
                "rect", x=lx, y=row_y - box_h + 2,
                width=box_w, height=box_h,
                fill=gl.color, opacity=0.75, stroke="#CCC",
            ))
            _svg_text(root, lx + box_w + 4, row_y,
                      gl.name[:10], font_size=9, anchor="start", fill="#444")


# ─── Cross-Section Engine ─────────────────────────────────────────


class CrossSectionEngine:
    """都市・路線を指定して断面図を生成するエンジン"""

    def __init__(
        self,
        store: CityMapStore,
        config: Optional[CrossSectionConfig] = None,
    ) -> None:
        self._store = store
        self._renderer = SVGCrossSectionRenderer(config)

    def generate(
        self,
        city: CityName,
        line_id: str,
        title: Optional[str] = None,
    ) -> Optional[CrossSectionResult]:
        line = self._store.lines.get(line_id)
        if line is None or line.city != city:
            return None

        stations = self._store.stations.list_by_line(line_id)
        stations_sorted = sorted(stations, key=lambda s: s.id)
        geology_layers = self._store.geology.list_by_city(city)

        svg = self._renderer.render(line, stations_sorted, geology_layers, title)

        return CrossSectionResult(
            city=city.value,
            line_id=line_id,
            svg=svg,
            station_count=len(stations_sorted),
            geology_count=len(geology_layers),
            width=self._renderer.config.width,
            height=self._renderer.config.height,
        )

    def generate_all_lines(self, city: CityName) -> List[CrossSectionResult]:
        lines = self._store.lines.list_by_city(city)
        results = []
        for line in lines:
            result = self.generate(city, line.id)
            if result is not None:
                results.append(result)
        return results
