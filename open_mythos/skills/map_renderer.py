"""
Sprint 71 — 主要都市地図ビジュアライザ (候補B: 3D断面図 SVGレンダラー)

参照: https://tokyo-danmenzu.pages.dev/ （東京地下断面図・3D地質断面ビューア）

city_map.CrossSection / Line から SVG 文字列を組み立てる。外部描画ライブラリは使わず、
標準ライブラリのみで SVG を直接構築する (ad_router が HTML 文字列を組むのと同じ流儀)。
PNG 変換は cairosvg を lazy import し、無ければ None を返す (フォールバック)。

オブジェクト:
  SvgStyle                  : 描画スタイル (寸法・スケール・フォント)
  BaseRenderer              : 抽象基底
  CrossSectionSvgRenderer   : 地質層 × 駅深度の地下断面 SVG
  FrontViewSvgRenderer      : 路線を正面から見た駅並び SVG
  MapRenderer               : ファサード (cross_section_svg / front_view_svg / to_png)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from open_mythos.skills.city_map import CrossSection, GeologyLayer, Line, Station


def _esc(text: str) -> str:
    """SVG/XML テキストエスケープ。"""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# スタイル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class SvgStyle:
    """SVG 描画スタイル。"""
    width:        int = 960
    height:       int = 480
    margin:       int = 60
    depth_scale:  float = 4.0    # 1m あたりの px (縦方向)
    legend_width: int = 160
    font_family:  str = "Inter, 'Hiragino Sans', sans-serif"
    bg_color:     str = "#F9FAFB"
    line_color:   str = "#111827"
    station_r:    int = 6


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# レンダラー
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaseRenderer(ABC):
    """SVG レンダラーの抽象基底。"""

    def __init__(self, style: Optional[SvgStyle] = None) -> None:
        self.style = style or SvgStyle()

    @abstractmethod
    def render(self, obj: object) -> str:
        """対象オブジェクトを SVG 文字列へ。"""
        ...

    # -- 共通ユーティリティ ------------------------------------------------
    def _svg_open(self) -> str:
        s = self.style
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'width="' + str(s.width) + '" height="' + str(s.height) + '" '
            'viewBox="0 0 ' + str(s.width) + ' ' + str(s.height) + '" '
            'font-family="' + _esc(s.font_family) + '">'
            '<rect x="0" y="0" width="' + str(s.width) + '" height="' + str(s.height) + '" '
            'fill="' + s.bg_color + '"/>'
        )

    @staticmethod
    def _svg_close() -> str:
        return "</svg>"


class CrossSectionSvgRenderer(BaseRenderer):
    """
    地下断面図 SVG。
    縦軸 = 地下深度 (地表 0 が上)。地質層を色帯 (<rect>)、駅を <circle>+<text>、
    路線を <polyline> で描く。右側に地質層の凡例。
    """

    def render(self, cs: CrossSection) -> str:  # type: ignore[override]
        s = self.style
        plot_left = s.margin
        plot_right = s.width - s.legend_width - s.margin
        plot_w = max(1, plot_right - plot_left)
        surface_y = s.margin  # 地表の y 座標

        def depth_to_y(depth_m: float) -> float:
            return surface_y + depth_m * s.depth_scale

        parts: List[str] = [self._svg_open()]

        # ── 地質層 (色帯) ──
        for layer in cs.layers:
            y0 = depth_to_y(layer.top_m)
            y1 = depth_to_y(layer.bottom_m)
            parts.append(
                '<rect x="' + str(plot_left) + '" y="' + str(round(y0, 1)) + '" '
                'width="' + str(plot_w) + '" height="' + str(round(y1 - y0, 1)) + '" '
                'fill="' + layer.color + '" fill-opacity="0.85"/>'
            )
            parts.append(
                '<text x="' + str(plot_left + 6) + '" y="' + str(round((y0 + y1) / 2 + 4, 1)) + '" '
                'font-size="11" fill="#3a2f1b">' + _esc(layer.name) + '</text>'
            )

        # ── 地表ライン ──
        parts.append(
            '<line x1="' + str(plot_left) + '" y1="' + str(surface_y) + '" '
            'x2="' + str(plot_right) + '" y2="' + str(surface_y) + '" '
            'stroke="#6B7280" stroke-width="1.5" stroke-dasharray="4 3"/>'
        )

        # ── 駅の x 座標 (順序で等間隔) ──
        n = max(1, len(cs.stations))
        def station_x(i: int) -> float:
            if n == 1:
                return plot_left + plot_w / 2
            return plot_left + plot_w * (i / (n - 1))

        # ── 路線 polyline (駅の深度をつなぐ) ──
        pts: List[str] = []
        for i, st in enumerate(cs.stations):
            depth = st.depth_m if st.depth_m is not None else 0.0
            pts.append(str(round(station_x(i), 1)) + "," + str(round(depth_to_y(depth), 1)))
        if pts:
            parts.append(
                '<polyline points="' + " ".join(pts) + '" fill="none" '
                'stroke="' + s.line_color + '" stroke-width="2.5"/>'
            )

        # ── 駅マーカー + ラベル ──
        for i, st in enumerate(cs.stations):
            depth = st.depth_m if st.depth_m is not None else 0.0
            x = round(station_x(i), 1)
            y = round(depth_to_y(depth), 1)
            parts.append(
                '<circle cx="' + str(x) + '" cy="' + str(y) + '" r="' + str(s.station_r) + '" '
                'fill="#FFFFFF" stroke="' + s.line_color + '" stroke-width="2"/>'
            )
            parts.append(
                '<text x="' + str(x) + '" y="' + str(round(y - 12, 1)) + '" '
                'font-size="11" text-anchor="middle" fill="#111827">' + _esc(st.name) + '</text>'
            )
            parts.append(
                '<text x="' + str(x) + '" y="' + str(round(y + 20, 1)) + '" '
                'font-size="9" text-anchor="middle" fill="#6B7280">'
                + _esc(("%.0f" % depth) + "m") + '</text>'
            )

        # ── タイトル ──
        parts.append(
            '<text x="' + str(plot_left) + '" y="24" font-size="16" font-weight="700" '
            'fill="#111827">' + _esc(cs.line_name + " 地下断面図") + '</text>'
        )

        # ── 凡例 ──
        parts.append(self._legend(cs.layers))
        parts.append(self._svg_close())
        return "".join(parts)

    def _legend(self, layers: List[GeologyLayer]) -> str:
        s = self.style
        lx = s.width - s.legend_width - s.margin // 2
        ly = s.margin
        out: List[str] = ['<g font-size="11">']
        out.append(
            '<text x="' + str(lx) + '" y="' + str(ly - 8) + '" font-weight="700" '
            'fill="#111827">地質層</text>'
        )
        for i, layer in enumerate(layers):
            ry = ly + i * 22
            out.append(
                '<rect x="' + str(lx) + '" y="' + str(ry) + '" width="14" height="14" '
                'fill="' + layer.color + '"/>'
            )
            out.append(
                '<text x="' + str(lx + 20) + '" y="' + str(ry + 12) + '" fill="#374151">'
                + _esc(layer.name) + '</text>'
            )
        out.append("</g>")
        return "".join(out)


class FrontViewSvgRenderer(BaseRenderer):
    """路線を正面 (進行方向) から見た駅並び SVG。駅を横一列に等間隔配置。"""

    def render(self, line: Line) -> str:  # type: ignore[override]
        s = self.style
        plot_left = s.margin
        plot_right = s.width - s.margin
        plot_w = max(1, plot_right - plot_left)
        mid_y = s.height // 2

        parts: List[str] = [self._svg_open()]
        parts.append(
            '<text x="' + str(plot_left) + '" y="24" font-size="16" font-weight="700" '
            'fill="#111827">' + _esc(line.name + " 路線図") + '</text>'
        )

        stations: List[Station] = line.stations
        n = max(1, len(stations))
        def x_of(i: int) -> float:
            if n == 1:
                return plot_left + plot_w / 2
            return plot_left + plot_w * (i / (n - 1))

        # 路線ライン
        if stations:
            parts.append(
                '<line x1="' + str(round(x_of(0), 1)) + '" y1="' + str(mid_y) + '" '
                'x2="' + str(round(x_of(len(stations) - 1), 1)) + '" y2="' + str(mid_y) + '" '
                'stroke="' + (line.color or "#111827") + '" stroke-width="6" stroke-linecap="round"/>'
            )
        # 駅
        for i, st in enumerate(stations):
            x = round(x_of(i), 1)
            parts.append(
                '<circle cx="' + str(x) + '" cy="' + str(mid_y) + '" r="' + str(s.station_r + 2) + '" '
                'fill="#FFFFFF" stroke="' + (line.color or "#111827") + '" stroke-width="3"/>'
            )
            parts.append(
                '<text x="' + str(x) + '" y="' + str(mid_y - 18) + '" font-size="11" '
                'text-anchor="middle" fill="#111827">' + _esc(st.name) + '</text>'
            )
        parts.append(self._svg_close())
        return "".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MapRenderer — ファサード
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MapRenderer:
    """断面図・路線図 SVG 生成のファサード。PNG は cairosvg が在れば対応。"""

    def __init__(self, style: Optional[SvgStyle] = None) -> None:
        self.style = style or SvgStyle()
        self._cross = CrossSectionSvgRenderer(self.style)
        self._front = FrontViewSvgRenderer(self.style)

    def cross_section_svg(self, cs: CrossSection, style: Optional[SvgStyle] = None) -> str:
        renderer = CrossSectionSvgRenderer(style) if style else self._cross
        return renderer.render(cs)

    def front_view_svg(self, line: Line, style: Optional[SvgStyle] = None) -> str:
        renderer = FrontViewSvgRenderer(style) if style else self._front
        return renderer.render(line)

    def to_png(self, svg: str) -> Optional[bytes]:
        """
        SVG → PNG。cairosvg が無ければ None を返す (フォールバック)。
        """
        try:
            import cairosvg  # type: ignore
        except Exception:
            return None
        try:
            return cairosvg.svg2png(bytestring=svg.encode("utf-8"))
        except Exception:
            return None
