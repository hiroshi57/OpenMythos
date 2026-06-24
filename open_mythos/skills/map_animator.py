"""
Sprint 73A — 地図アニメーション (時系列断面変化)

地質調査年次ごとの層変化を SVG SMIL アニメーションで表現する。
外部ライブラリなし (xml.etree.ElementTree のみ)。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from open_mythos.skills.city_map import (
    CityName, GeologyLayer, GeologyLayerType,
)


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class SurveySnapshot:
    """ある年次の地質調査スナップショット"""
    year: int
    city: CityName
    layers: List[GeologyLayer]

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "city": self.city.value,
            "layers": [gl.to_dict() for gl in self.layers],
        }


@dataclass
class AnimationConfig:
    """SVG アニメーション設定"""
    width: int = 900
    height: int = 500
    margin_left: int = 80
    margin_top: int = 60
    margin_bottom: int = 40
    margin_right: int = 40
    max_depth_m: float = 60.0
    frame_duration_s: float = 1.5   # 各フレームの表示秒数
    repeat_count: str = "indefinite"

    @property
    def plot_width(self) -> int:
        return self.width - self.margin_left - self.margin_right

    @property
    def surface_y(self) -> int:
        return self.margin_top + 30

    def depth_to_y(self, depth_m: float) -> float:
        available = self.height - self.surface_y - self.margin_bottom
        return self.surface_y + min(depth_m / self.max_depth_m, 1.0) * available

    def depth_to_h(self, d_from: float, d_to: float) -> Tuple[float, float]:
        y0 = self.depth_to_y(d_from)
        y1 = self.depth_to_y(min(d_to, self.max_depth_m))
        return y0, max(y1 - y0, 0.0)


@dataclass
class AnimationResult:
    """アニメーション SVG 生成結果"""
    city: str
    years: List[int]
    svg: str
    frame_count: int
    duration_s: float
    width: int
    height: int

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "years": self.years,
            "svg": self.svg,
            "frame_count": self.frame_count,
            "duration_s": self.duration_s,
            "width": self.width,
            "height": self.height,
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


# ─── SurveyDataset ────────────────────────────────────────────────


class SurveyDataset:
    """時系列地質調査データセット (静的プリセット)"""

    @classmethod
    def build_tokyo(cls) -> List[SurveySnapshot]:
        """東京の年次別地質調査スナップショット (1960・1980・2000・2020)"""
        base_layers = [
            (GeologyLayerType.FILL,      "盛土",           0.0, 3.0,  "#D2B48C"),
            (GeologyLayerType.ALLUVIUM,  "沖積粘土層",      3.0, 12.0, "#90EE90"),
            (GeologyLayerType.SAND,      "沖積砂層",       12.0, 20.0, "#F4D03F"),
            (GeologyLayerType.DILUVIUM,  "洪積粘土層",     20.0, 35.0, "#85C1E9"),
            (GeologyLayerType.GRAVEL,    "洪積砂礫層",     35.0, 55.0, "#F0A500"),
            (GeologyLayerType.BEDROCK,   "岩盤",           55.0, 100.0,"#AAB7B8"),
        ]
        snapshots = []
        # 年次ごとに盛土厚と沖積粘土厚を若干変化させる
        variations = {
            1960: (0.0, 2.0,  3.0, 10.0),  # fill: 0-2  alluvium: 2-10
            1980: (0.0, 2.8,  2.8, 11.5),  # fill: 0-2.8
            2000: (0.0, 3.2,  3.2, 12.0),
            2020: (0.0, 3.5,  3.5, 12.5),
        }
        for year, (f_from, f_to, a_from, a_to) in variations.items():
            layers = []
            for i, (lt, name, d_from, d_to, color) in enumerate(base_layers):
                if lt == GeologyLayerType.FILL:
                    d_from, d_to = f_from, f_to
                elif lt == GeologyLayerType.ALLUVIUM:
                    d_from, d_to = a_from, a_to
                layers.append(GeologyLayer(
                    id=f"tky-{year}-{lt.value}",
                    city=CityName.TOKYO,
                    layer_type=lt, name=name,
                    depth_from_m=d_from, depth_to_m=d_to,
                    color=color,
                ))
            snapshots.append(SurveySnapshot(year=year, city=CityName.TOKYO, layers=layers))
        return sorted(snapshots, key=lambda s: s.year)

    @classmethod
    def build_osaka(cls) -> List[SurveySnapshot]:
        """大阪の年次別地質調査スナップショット"""
        variations = {
            1960: (0.0, 3.5,  3.5, 15.0),
            1980: (0.0, 4.0,  4.0, 16.5),
            2000: (0.0, 4.3,  4.3, 17.5),
            2020: (0.0, 4.6,  4.6, 18.5),
        }
        snapshots = []
        for year, (f_from, f_to, a_from, a_to) in variations.items():
            layers = [
                GeologyLayer(f"osk-{year}-fill",   CityName.OSAKA, GeologyLayerType.FILL,
                             "盛土", f_from, f_to, "#D2B48C"),
                GeologyLayer(f"osk-{year}-alluvium", CityName.OSAKA, GeologyLayerType.ALLUVIUM,
                             "沖積粘土層", a_from, a_to, "#90EE90"),
                GeologyLayer(f"osk-{year}-sand",   CityName.OSAKA, GeologyLayerType.SAND,
                             "沖積砂層", a_to, a_to + 10.0, "#F4D03F"),
                GeologyLayer(f"osk-{year}-clay",   CityName.OSAKA, GeologyLayerType.CLAY,
                             "大阪粘土層", a_to + 10.0, 60.0, "#7FB3D3"),
                GeologyLayer(f"osk-{year}-bedrock", CityName.OSAKA, GeologyLayerType.BEDROCK,
                             "基盤岩", 60.0, 100.0, "#AAB7B8"),
            ]
            snapshots.append(SurveySnapshot(year=year, city=CityName.OSAKA, layers=layers))
        return sorted(snapshots, key=lambda s: s.year)

    @classmethod
    def build(cls, city: CityName) -> List[SurveySnapshot]:
        if city == CityName.TOKYO:
            return cls.build_tokyo()
        elif city == CityName.OSAKA:
            return cls.build_osaka()
        # 他都市は Tokyo を流用してラベルだけ変える
        snapshots = cls.build_tokyo()
        for s in snapshots:
            s.city = city
        return snapshots


# ─── MapAnimator ──────────────────────────────────────────────────


class MapAnimator:
    """時系列地質断面 SVG アニメーターを生成するクラス"""

    def __init__(self, config: Optional[AnimationConfig] = None) -> None:
        self._cfg = config or AnimationConfig()

    def animate(
        self,
        snapshots: List[SurveySnapshot],
        title: Optional[str] = None,
    ) -> AnimationResult:
        """複数スナップショットから SMIL アニメーション SVG を生成"""
        if not snapshots:
            return AnimationResult(
                city="", years=[], svg="<svg/>",
                frame_count=0, duration_s=0.0,
                width=self._cfg.width, height=self._cfg.height,
            )
        city = snapshots[0].city.value
        years = [s.year for s in snapshots]
        svg = self._render(snapshots, title or f"{city.upper()} 地質断面 時系列変化")
        total_dur = len(snapshots) * self._cfg.frame_duration_s

        return AnimationResult(
            city=city, years=years, svg=svg,
            frame_count=len(snapshots), duration_s=total_dur,
            width=self._cfg.width, height=self._cfg.height,
        )

    def _render(self, snapshots: List[SurveySnapshot], title: str) -> str:
        cfg = self._cfg
        root = _el("svg",
                   xmlns="http://www.w3.org/2000/svg",
                   width=cfg.width, height=cfg.height,
                   viewBox=f"0 0 {cfg.width} {cfg.height}")
        root.append(_el("rect", x=0, y=0,
                        width=cfg.width, height=cfg.height, fill="#FAFAFA"))

        # タイトル
        _txt(root, cfg.width / 2, 30, title, font_size=14, fill="#222", font_weight="bold")

        # 地表線
        root.append(_el("line",
                        x1=cfg.margin_left, y1=cfg.surface_y,
                        x2=cfg.width - cfg.margin_right, y2=cfg.surface_y,
                        stroke="#555", stroke_width=2))

        # 深度軸
        self._render_axis(root)

        # 地質層アニメーション
        n = len(snapshots)
        total_dur_s = n * cfg.frame_duration_s
        layer_types = self._collect_layer_types(snapshots)

        for lt in layer_types:
            self._render_animated_layer(root, lt, snapshots, total_dur_s)

        # 年次ラベルアニメーション
        self._render_year_labels(root, snapshots, total_dur_s)

        return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")

    def _collect_layer_types(
        self, snapshots: List[SurveySnapshot]
    ) -> List[GeologyLayerType]:
        seen = []
        seen_set = set()
        for s in snapshots:
            for gl in s.layers:
                if gl.layer_type not in seen_set:
                    seen.append(gl.layer_type)
                    seen_set.add(gl.layer_type)
        return seen

    def _render_animated_layer(
        self,
        root: ET.Element,
        lt: GeologyLayerType,
        snapshots: List[SurveySnapshot],
        total_dur_s: float,
    ) -> None:
        cfg = self._cfg
        n = len(snapshots)
        frame_dur = cfg.frame_duration_s

        # 各フレームの y / height 値を収集
        y_vals: List[str] = []
        h_vals: List[str] = []
        colors: List[str] = []

        for s in snapshots:
            gl = next((g for g in s.layers if g.layer_type == lt), None)
            if gl:
                y0, h = cfg.depth_to_h(gl.depth_from_m, gl.depth_to_m)
                y_vals.append(f"{y0:.1f}")
                h_vals.append(f"{max(h, 0.1):.1f}")
                colors.append(gl.color)
            else:
                y_vals.append(f"{cfg.surface_y:.1f}")
                h_vals.append("0")
                colors.append("none")

        # 最初フレームで rect を描画、SMIL で y/height を変化させる
        first_gl = next(
            (g for s in snapshots for g in s.layers if g.layer_type == lt), None
        )
        if first_gl is None:
            return

        y0_init, h_init = cfg.depth_to_h(first_gl.depth_from_m, first_gl.depth_to_m)
        color_init = first_gl.color

        rect = _el("rect",
                   x=cfg.margin_left, y=f"{y0_init:.1f}",
                   width=cfg.plot_width, height=f"{max(h_init, 0.1):.1f}",
                   fill=color_init, opacity="0.65",
                   stroke="#CCC", stroke_width="0.5")

        # SMIL animate: y
        key_times = ";".join(
            f"{i / n:.3f}" for i in range(n)
        ) + ";1.000"
        y_values = ";".join(y_vals) + f";{y_vals[0]}"
        h_values = ";".join(h_vals) + f";{h_vals[0]}"

        anim_y = _el("animate",
                     attributeName="y",
                     values=y_values,
                     keyTimes=key_times,
                     dur=f"{total_dur_s:.1f}s",
                     repeatCount=cfg.repeat_count,
                     calcMode="discrete")
        anim_h = _el("animate",
                     attributeName="height",
                     values=h_values,
                     keyTimes=key_times,
                     dur=f"{total_dur_s:.1f}s",
                     repeatCount=cfg.repeat_count,
                     calcMode="discrete")
        rect.append(anim_y)
        rect.append(anim_h)
        root.append(rect)

        # 層名ラベル (最初フレームの位置に固定)
        if h_init > 14:
            _txt(root, cfg.margin_left + 4, y0_init + min(h_init / 2, 13),
                 first_gl.name, font_size=9, anchor="start", fill="#555")

    def _render_year_labels(
        self,
        root: ET.Element,
        snapshots: List[SurveySnapshot],
        total_dur_s: float,
    ) -> None:
        cfg = self._cfg
        n = len(snapshots)
        key_times = ";".join(f"{i / n:.3f}" for i in range(n)) + ";1.000"
        year_values = ";".join(str(s.year) for s in snapshots) + f";{snapshots[0].year}"

        lbl = _el("text",
                  x=cfg.margin_left + cfg.plot_width - 10,
                  y=cfg.surface_y + 20,
                  text_anchor="end",
                  font_size="22",
                  font_family="Arial,sans-serif",
                  fill="#333",
                  font_weight="bold",
                  opacity="0.7")
        lbl.text = str(snapshots[0].year)

        anim = _el("animate",
                   attributeName="text",
                   values=year_values,
                   keyTimes=key_times,
                   dur=f"{total_dur_s:.1f}s",
                   repeatCount=cfg.repeat_count,
                   calcMode="discrete")
        # SVG text の内容を animate で変えるのは非標準なので textContent 代替として
        # SVG text 要素の visibility を切り替える方式にする
        # → シンプルに固定ラベルを複数配置し animate visibility で切替
        root.append(lbl)

        # 各年次のラベルを visibility アニメで切替
        for i, s in enumerate(snapshots):
            lbl_i = _el("text",
                        x=cfg.margin_left + cfg.plot_width - 10,
                        y=cfg.surface_y + 20,
                        text_anchor="end",
                        font_size="22",
                        font_family="Arial,sans-serif",
                        fill="#E60012",
                        font_weight="bold")
            lbl_i.text = str(s.year)

            # visibility: visible のみ当該フレーム、それ以外は hidden
            vis_values = []
            for j in range(n):
                vis_values.append("visible" if j == i else "hidden")
            vis_values.append(vis_values[0])

            anim_vis = _el("animate",
                           attributeName="visibility",
                           values=";".join(vis_values),
                           keyTimes=key_times,
                           dur=f"{total_dur_s:.1f}s",
                           repeatCount=cfg.repeat_count,
                           calcMode="discrete")
            lbl_i.append(anim_vis)
            root.append(lbl_i)

    def _render_axis(self, root: ET.Element) -> None:
        cfg = self._cfg
        for d in range(0, int(cfg.max_depth_m) + 1, 10):
            y = cfg.depth_to_y(d)
            _txt(root, cfg.margin_left - 6, y + 4,
                 f"{d}m", font_size=9, anchor="end", fill="#888")
            root.append(_el("line",
                            x1=cfg.margin_left - 3, y1=y,
                            x2=cfg.margin_left, y2=y,
                            stroke="#BBB", stroke_width=1))
        root.append(_el("line",
                        x1=cfg.margin_left, y1=cfg.surface_y,
                        x2=cfg.margin_left,
                        y2=cfg.height - cfg.margin_bottom,
                        stroke="#999", stroke_width=1))
