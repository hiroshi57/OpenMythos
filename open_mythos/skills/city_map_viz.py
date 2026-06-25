"""
Sprint 78A — 都市マップビジュアライゼーション

交通量・騒音・群衆・災害アラート・エネルギーの各データを
SVG ベースのインタラクティブ HTML マップに統合する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ─── Enums ────────────────────────────────────────────────────────


class MapLayer(str, Enum):
    TRAFFIC  = "traffic"   # 交通量
    NOISE    = "noise"     # 騒音
    CROWD    = "crowd"     # 群衆
    ENERGY   = "energy"    # エネルギー
    DISASTER = "disaster"  # 災害アラート


# ─── カラーパレット ────────────────────────────────────────────────

_TRAFFIC_COLORS = {
    "clear":     "#4caf50",
    "moderate":  "#ffeb3b",
    "congested": "#ff9800",
    "gridlock":  "#f44336",
}

_NOISE_COLORS = {
    "compliant":  "#4caf50",
    "near_limit": "#ffeb3b",
    "violation":  "#f44336",
}

_CROWD_COLORS = {
    "sparse":  "#e3f2fd",
    "normal":  "#4fc3f7",
    "crowded": "#ff9800",
    "packed":  "#f44336",
}

_DISASTER_COLORS = {
    "info":     "#2196f3",
    "watch":    "#ffeb3b",
    "warning":  "#ff9800",
    "critical": "#f44336",
}

_ENERGY_COLORS = {
    "normal":   "#4caf50",
    "high":     "#ffeb3b",
    "critical": "#f44336",
}


# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class DistrictData:
    """地区単位の統合データ。"""
    name: str
    x: float          # SVG x 座標 (0–100)
    y: float          # SVG y 座標 (0–100)
    width: float = 18.0
    height: float = 14.0

    # 各レイヤーのステータス (省略可)
    traffic_level: Optional[str] = None   # clear/moderate/congested/gridlock
    noise_status: Optional[str] = None    # compliant/near_limit/violation
    crowd_level: Optional[str] = None     # sparse/normal/crowded/packed
    energy_status: Optional[str] = None   # normal/high/critical
    disaster_level: Optional[str] = None  # info/watch/warning/critical

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "traffic_level": self.traffic_level,
            "noise_status": self.noise_status,
            "crowd_level": self.crowd_level,
            "energy_status": self.energy_status,
            "disaster_level": self.disaster_level,
        }


@dataclass
class CityMapData:
    """都市マップの全データ。"""
    city: str
    districts: List[DistrictData] = field(default_factory=list)
    active_layer: MapLayer = MapLayer.TRAFFIC

    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "active_layer": self.active_layer.value,
            "districts": [d.to_dict() for d in self.districts],
        }


# ─── SVG/HTML ジェネレーター ──────────────────────────────────────


def _district_color(district: DistrictData, layer: MapLayer) -> str:
    """レイヤーに応じた地区の色を返す。ステータスが None の場合は灰色 #e0e0e0 を返す。"""
    if layer == MapLayer.TRAFFIC:
        return _TRAFFIC_COLORS.get(district.traffic_level, "#e0e0e0")
    elif layer == MapLayer.NOISE:
        return _NOISE_COLORS.get(district.noise_status, "#e0e0e0")
    elif layer == MapLayer.CROWD:
        return _CROWD_COLORS.get(district.crowd_level, "#e0e0e0")
    elif layer == MapLayer.ENERGY:
        return _ENERGY_COLORS.get(district.energy_status, "#e0e0e0")
    elif layer == MapLayer.DISASTER:
        if district.disaster_level:
            return _DISASTER_COLORS.get(district.disaster_level, "#e0e0e0")
        return "#e8f5e9"
    return "#e0e0e0"


def _disaster_badge(district: DistrictData) -> str:
    if not district.disaster_level:
        return ""
    colors = {"info": "#2196f3", "watch": "#ffeb3b",
               "warning": "#ff9800", "critical": "#f44336"}
    c = colors.get(district.disaster_level, "#999")
    cx = district.x + district.width - 3
    cy = district.y + 3
    return f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="{c}" stroke="white" stroke-width="0.8"/>'


def _legend_html(layer: MapLayer) -> str:
    if layer == MapLayer.TRAFFIC:
        items = [
            ("#4caf50", "通常走行"),
            ("#ffeb3b", "やや混雑"),
            ("#ff9800", "渋滞"),
            ("#f44336", "完全停滞"),
        ]
    elif layer == MapLayer.NOISE:
        items = [
            ("#4caf50", "規制内"),
            ("#ffeb3b", "規制値近傍"),
            ("#f44336", "規制超過"),
        ]
    elif layer == MapLayer.CROWD:
        items = [
            ("#e3f2fd", "閑散"),
            ("#4fc3f7", "通常"),
            ("#ff9800", "混雑"),
            ("#f44336", "超混雑"),
        ]
    elif layer == MapLayer.ENERGY:
        items = [
            ("#4caf50", "通常"),
            ("#ffeb3b", "高使用"),
            ("#f44336", "異常"),
        ]
    elif layer == MapLayer.DISASTER:
        items = [
            ("#e8f5e9", "アラートなし"),
            ("#2196f3", "情報"),
            ("#ffeb3b", "注意"),
            ("#ff9800", "警戒"),
            ("#f44336", "緊急"),
        ]
    else:
        items = []

    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0">'
        f'<div style="width:14px;height:14px;background:{c};border:1px solid #ccc;border-radius:2px"></div>'
        f'<span style="font-size:11px">{label}</span></div>'
        for c, label in items
    )
    return rows


def generate_html(map_data: CityMapData, width: int = 700, height: int = 500) -> str:
    """インタラクティブな HTML 都市マップを生成する。"""
    layer = map_data.active_layer
    svg_w, svg_h = 100, 100  # SVG 座標系は 0–100

    # SVG 地区セル
    cells = []
    for d in map_data.districts:
        color = _district_color(d, layer)
        badge = _disaster_badge(d) if layer != MapLayer.DISASTER else ""
        # ツールチップ用 title
        tip_lines = [d.name]
        if d.traffic_level:  tip_lines.append(f"交通: {d.traffic_level}")
        if d.noise_status:   tip_lines.append(f"騒音: {d.noise_status}")
        if d.crowd_level:    tip_lines.append(f"群衆: {d.crowd_level}")
        if d.energy_status:  tip_lines.append(f"電力: {d.energy_status}")
        if d.disaster_level: tip_lines.append(f"災害: {d.disaster_level}")
        tip = "&#10;".join(tip_lines)

        cells.append(
            f'<rect x="{d.x}" y="{d.y}" width="{d.width}" height="{d.height}" '
            f'fill="{color}" stroke="#555" stroke-width="0.4" rx="1" opacity="0.88">'
            f'<title>{tip}</title></rect>'
            f'<text x="{d.x + d.width/2}" y="{d.y + d.height/2 + 1.5}" '
            f'text-anchor="middle" font-size="3.5" fill="#222" font-family="sans-serif"'
            f' pointer-events="none">{d.name}</text>'
            + (badge if layer == MapLayer.DISASTER else "")
        )

    svg_cells = "\n    ".join(cells)
    legend = _legend_html(layer)
    layer_label = {
        MapLayer.TRAFFIC:  "交通量",
        MapLayer.NOISE:    "騒音",
        MapLayer.CROWD:    "群衆密度",
        MapLayer.ENERGY:   "エネルギー",
        MapLayer.DISASTER: "災害アラート",
    }.get(layer, layer.value)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenMythos 都市マップ — {map_data.city} / {layer_label}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a2e; color: #eee; font-family: system-ui, sans-serif; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 20px; }}
  h1 {{ font-size: 18px; margin-bottom: 4px; color: #90caf9; }}
  .sub {{ font-size: 12px; color: #888; margin-bottom: 16px; }}
  .container {{ display: flex; gap: 16px; align-items: flex-start; max-width: 900px; width: 100%; }}
  .map-wrap {{ flex: 1; background: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 12px; }}
  svg {{ width: 100%; height: auto; border-radius: 6px; }}
  .sidebar {{ width: 160px; shrink: 0; }}
  .legend-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
  .legend-box h3 {{ font-size: 12px; color: #8b949e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: .05em; }}
  .stats {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; }}
  .stats h3 {{ font-size: 12px; color: #8b949e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: .05em; }}
  .stat-row {{ display: flex; justify-content: space-between; font-size: 11px; padding: 3px 0; border-bottom: 1px solid #21262d; }}
  .stat-val {{ color: #58a6ff; font-weight: bold; }}
  .layer-btn {{ display: block; width: 100%; padding: 6px 10px; margin: 3px 0; border-radius: 5px; border: 1px solid #30363d; background: #21262d; color: #ccc; font-size: 11px; cursor: pointer; text-align: left; }}
  .layer-btn.active {{ background: #1f6feb; border-color: #388bfd; color: #fff; }}
  .layer-select {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-bottom: 12px; }}
  .layer-select h3 {{ font-size: 12px; color: #8b949e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: .05em; }}
</style>
</head>
<body>
<h1>🗺 OpenMythos 都市マップ</h1>
<div class="sub">{map_data.city} — {layer_label}レイヤー | {len(map_data.districts)} 地区</div>
<div class="container">
  <div class="map-wrap">
    <svg viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg"
         style="background:#0d2137">
      <!-- 背景グリッド -->
      <defs>
        <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
          <path d="M 10 0 L 0 0 0 10" fill="none" stroke="#1a3a5c" stroke-width="0.3"/>
        </pattern>
      </defs>
      <rect width="100" height="100" fill="url(#grid)"/>
      <!-- 地区セル -->
      {svg_cells}
    </svg>
  </div>
  <div class="sidebar">
    <div class="layer-select">
      <h3>レイヤー</h3>
      {''.join(
        f'<div class="layer-btn{" active" if MapLayer(k)==layer else ""}">{v}</div>'
        for k, v in [
          ("traffic","交通量"), ("noise","騒音"),
          ("crowd","群衆密度"), ("energy","エネルギー"), ("disaster","災害アラート")
        ]
      )}
    </div>
    <div class="legend-box">
      <h3>凡例</h3>
      {legend}
    </div>
    <div class="stats">
      <h3>統計</h3>
      <div class="stat-row"><span>地区数</span><span class="stat-val">{len(map_data.districts)}</span></div>
      <div class="stat-row"><span>表示レイヤー</span><span class="stat-val">{layer_label}</span></div>
      <div class="stat-row"><span>都市</span><span class="stat-val">{map_data.city}</span></div>
    </div>
  </div>
</div>
</body>
</html>"""
    return html


# ─── Store / Builder ──────────────────────────────────────────────


class CityMapStore:
    """都市マップデータのインメモリ管理。"""

    def __init__(self) -> None:
        self._maps: Dict[str, CityMapData] = {}

    def set(self, city: str, data: CityMapData) -> None:
        self._maps[city] = data

    def get(self, city: str) -> Optional[CityMapData]:
        return self._maps.get(city)

    def list_cities(self) -> List[str]:
        return list(self._maps.keys())

    def count(self) -> int:
        return len(self._maps)


class CityMapBuilder:
    """都市マップデータのビルダー。"""

    def __init__(self, store: Optional[CityMapStore] = None) -> None:
        self.store = store or CityMapStore()

    def build(
        self,
        city: str,
        districts: List[DistrictData],
        active_layer: MapLayer = MapLayer.TRAFFIC,
    ) -> CityMapData:
        data = CityMapData(city=city, districts=districts, active_layer=active_layer)
        self.store.set(city, data)
        return data

    def get_html(self, city: str, layer: Optional[MapLayer] = None) -> Optional[str]:
        data = self.store.get(city)
        if data is None:
            return None
        if layer is not None:
            data.active_layer = layer
        return generate_html(data)


# ─── プリセット: 東京主要地区 ────────────────────────────────────

TOKYO_DISTRICTS: List[DistrictData] = [
    DistrictData("新宿",   x=5,  y=5,  traffic_level="congested", noise_status="violation",  crowd_level="packed",  energy_status="high",    disaster_level=None),
    DistrictData("渋谷",   x=25, y=5,  traffic_level="moderate",  noise_status="near_limit", crowd_level="crowded", energy_status="normal",  disaster_level=None),
    DistrictData("池袋",   x=45, y=5,  traffic_level="congested", noise_status="compliant",  crowd_level="crowded", energy_status="normal",  disaster_level=None),
    DistrictData("上野",   x=65, y=5,  traffic_level="clear",     noise_status="compliant",  crowd_level="normal",  energy_status="normal",  disaster_level=None),
    DistrictData("秋葉原", x=5,  y=22, traffic_level="moderate",  noise_status="near_limit", crowd_level="crowded", energy_status="high",    disaster_level=None),
    DistrictData("千代田", x=25, y=22, traffic_level="gridlock",  noise_status="violation",  crowd_level="packed",  energy_status="critical",disaster_level="warning"),
    DistrictData("中央",   x=45, y=22, traffic_level="congested", noise_status="near_limit", crowd_level="crowded", energy_status="high",    disaster_level=None),
    DistrictData("台東",   x=65, y=22, traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",  disaster_level=None),
    DistrictData("品川",   x=5,  y=40, traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",  disaster_level=None),
    DistrictData("目黒",   x=25, y=40, traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",  disaster_level=None),
    DistrictData("江東",   x=45, y=40, traffic_level="moderate",  noise_status="near_limit", crowd_level="normal",  energy_status="high",    disaster_level="watch"),
    DistrictData("墨田",   x=65, y=40, traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",  disaster_level=None),
    DistrictData("世田谷", x=5,  y=58, traffic_level="clear",     noise_status="compliant",  crowd_level="normal",  energy_status="normal",  disaster_level=None),
    DistrictData("杉並",   x=25, y=58, traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",  disaster_level=None),
    DistrictData("江戸川", x=45, y=58, traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",  disaster_level=None),
    DistrictData("葛飾",   x=65, y=58, traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",  disaster_level=None),
    DistrictData("練馬",   x=5,  y=76, traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",  disaster_level=None),
    DistrictData("板橋",   x=25, y=76, traffic_level="moderate",  noise_status="near_limit", crowd_level="normal",  energy_status="normal",  disaster_level=None),
    DistrictData("足立",   x=45, y=76, traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",  disaster_level=None),
    DistrictData("荒川",   x=65, y=76, traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",  disaster_level=None),
]
