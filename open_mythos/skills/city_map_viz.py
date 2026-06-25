"""
Sprint 78A — 都市マップビジュアライゼーション

Google Maps 風ライトテーマ + SVG インタラクティブ都市マップ。
交通量・騒音・群衆・エネルギー・災害アラートをレイヤー切替で表示する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ─── Enums ────────────────────────────────────────────────────────


class MapLayer(str, Enum):
    TRAFFIC  = "traffic"
    NOISE    = "noise"
    CROWD    = "crowd"
    ENERGY   = "energy"
    DISASTER = "disaster"


# ─── カラーパレット（Google Maps ライトテーマに合わせた彩度低め）────────


_TRAFFIC_COLORS = {
    "clear":     "#34a853",   # Google green
    "moderate":  "#fbbc04",   # Google yellow
    "congested": "#ea8600",   # orange
    "gridlock":  "#d93025",   # Google red
}

_NOISE_COLORS = {
    "compliant":  "#34a853",
    "near_limit": "#fbbc04",
    "violation":  "#d93025",
}

_CROWD_COLORS = {
    "sparse":  "#e8f0fe",    # very light blue
    "normal":  "#4285f4",    # Google blue
    "crowded": "#ea8600",
    "packed":  "#d93025",
}

_DISASTER_COLORS = {
    "info":     "#4285f4",
    "watch":    "#fbbc04",
    "warning":  "#ea8600",
    "critical": "#d93025",
}

_ENERGY_COLORS = {
    "normal":   "#34a853",
    "high":     "#fbbc04",
    "critical": "#d93025",
}

_NO_DATA_COLOR = "#e8e0d0"   # map land color — district blends into base

# ─── Data Classes ─────────────────────────────────────────────────


@dataclass
class DistrictData:
    """地区単位の統合データ。"""
    name: str
    x: float
    y: float
    width: float = 18.0
    height: float = 14.0
    traffic_level: Optional[str] = None
    noise_status: Optional[str] = None
    crowd_level: Optional[str] = None
    energy_status: Optional[str] = None
    disaster_level: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
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


# ─── ヘルパー ──────────────────────────────────────────────────────


def _district_color(district: DistrictData, layer: MapLayer) -> str:
    """レイヤーに応じた地区の塗りつぶし色。None → _NO_DATA_COLOR。"""
    if layer == MapLayer.TRAFFIC:
        return _TRAFFIC_COLORS.get(district.traffic_level, _NO_DATA_COLOR)
    elif layer == MapLayer.NOISE:
        return _NOISE_COLORS.get(district.noise_status, _NO_DATA_COLOR)
    elif layer == MapLayer.CROWD:
        return _CROWD_COLORS.get(district.crowd_level, _NO_DATA_COLOR)
    elif layer == MapLayer.ENERGY:
        return _ENERGY_COLORS.get(district.energy_status, _NO_DATA_COLOR)
    elif layer == MapLayer.DISASTER:
        if district.disaster_level:
            return _DISASTER_COLORS.get(district.disaster_level, _NO_DATA_COLOR)
        return "#e8f5e9"
    return _NO_DATA_COLOR


def _legend_html(layer: MapLayer) -> str:
    if layer == MapLayer.TRAFFIC:
        items = [("#34a853", "通常走行"), ("#fbbc04", "やや混雑"),
                 ("#ea8600", "渋滞"), ("#d93025", "完全停滞")]
    elif layer == MapLayer.NOISE:
        items = [("#34a853", "規制内"), ("#fbbc04", "規制値近傍"), ("#d93025", "規制超過")]
    elif layer == MapLayer.CROWD:
        items = [("#e8f0fe", "閑散"), ("#4285f4", "通常"),
                 ("#ea8600", "混雑"), ("#d93025", "超混雑")]
    elif layer == MapLayer.ENERGY:
        items = [("#34a853", "通常"), ("#fbbc04", "高使用"), ("#d93025", "異常")]
    elif layer == MapLayer.DISASTER:
        items = [("#e8f5e9", "アラートなし"), ("#4285f4", "情報"),
                 ("#fbbc04", "注意"), ("#ea8600", "警戒"), ("#d93025", "緊急")]
    else:
        items = []
    rows = "".join(
        f'<div class="leg-row">'
        f'<span class="leg-dot" style="background:{c}"></span>'
        f'<span>{label}</span></div>'
        for c, label in items
    )
    return rows


def generate_html(map_data: CityMapData, width: int = 1000, height: int = 700) -> str:
    """Google Maps 風プロフェッショナル HTML 都市マップを生成する。"""
    layer = map_data.active_layer
    vw, vh = 100, 76   # SVG 座標系

    layer_label = {
        MapLayer.TRAFFIC:  "交通量",
        MapLayer.NOISE:    "騒音",
        MapLayer.CROWD:    "群衆密度",
        MapLayer.ENERGY:   "エネルギー",
        MapLayer.DISASTER: "災害アラート",
    }.get(layer, layer.value)

    layer_icon = {
        MapLayer.TRAFFIC:  "🚗",
        MapLayer.NOISE:    "🔊",
        MapLayer.CROWD:    "👥",
        MapLayer.ENERGY:   "⚡",
        MapLayer.DISASTER: "⚠️",
    }.get(layer, "📍")

    # ── 地区 SVG ──────────────────────────────────────────────────
    district_rects = []
    district_data_json_parts = []

    for i, d in enumerate(map_data.districts):
        color = _district_color(d, layer)
        # ダミー street lines（地区内に2本の薄い白線）
        cx, cy = d.x + d.width / 2, d.y + d.height / 2
        street_h = (f'<line x1="{d.x+1}" y1="{cy:.1f}" x2="{d.x+d.width-1:.1f}" '
                    f'y1="{cy:.1f}" x2="{d.x+d.width-1:.1f}" y2="{cy:.1f}" '
                    f'stroke="rgba(255,255,255,0.35)" stroke-width="0.3" pointer-events="none"/>')
        street_v = (f'<line x1="{cx:.1f}" y1="{d.y+1}" x2="{cx:.1f}" y2="{d.y+d.height-1:.1f}" '
                    f'stroke="rgba(255,255,255,0.35)" stroke-width="0.3" pointer-events="none"/>')

        # 災害バッジ
        badge = ""
        if d.disaster_level and layer != MapLayer.DISASTER:
            bc = _DISASTER_COLORS.get(d.disaster_level, "#999")
            bx, by = d.x + d.width - 2.5, d.y + 2.5
            badge = (f'<circle cx="{bx}" cy="{by}" r="2.2" fill="{bc}" '
                     f'stroke="white" stroke-width="0.6" pointer-events="none"/>')

        tip_lines = [d.name]
        if d.traffic_level:  tip_lines.append(f"交通: {d.traffic_level}")
        if d.noise_status:   tip_lines.append(f"騒音: {d.noise_status}")
        if d.crowd_level:    tip_lines.append(f"群衆: {d.crowd_level}")
        if d.energy_status:  tip_lines.append(f"電力: {d.energy_status}")
        if d.disaster_level: tip_lines.append(f"災害: {d.disaster_level}")

        district_rects.append(
            f'<g class="district" data-idx="{i}" '
            f'onmouseenter="showTip(event,{i})" onmouseleave="hideTip()" onclick="selectDistrict({i})">'
            f'<rect x="{d.x}" y="{d.y}" width="{d.width}" height="{d.height}" '
            f'fill="{color}" fill-opacity="0.78" stroke="white" stroke-width="0.35" rx="0.8"/>'
            + street_h + street_v +
            f'<text x="{cx:.1f}" y="{cy+0.8:.1f}" text-anchor="middle" '
            f'font-size="2.8" fill="#202124" font-family="system-ui,sans-serif" '
            f'font-weight="500" paint-order="stroke" stroke="white" stroke-width="0.8" '
            f'pointer-events="none">{d.name}</text>'
            + badge +
            f'</g>'
        )

        district_data_json_parts.append(
            "{" +
            f'"name":"{d.name}",'
            f'"traffic":"{d.traffic_level or ""}",'
            f'"noise":"{d.noise_status or ""}",'
            f'"crowd":"{d.crowd_level or ""}",'
            f'"energy":"{d.energy_status or ""}",'
            f'"disaster":"{d.disaster_level or ""}"'
            + "}"
        )

    svg_content = "\n    ".join(district_rects)
    district_data_json = "[" + ",".join(district_data_json_parts) + "]"
    legend = _legend_html(layer)

    # ── 統計 ──────────────────────────────────────────────────────
    total = len(map_data.districts)
    if layer == MapLayer.TRAFFIC:
        alert_n = sum(1 for d in map_data.districts if d.traffic_level in ("congested", "gridlock"))
        stat_label, stat_val = "混雑地区", f"{alert_n} / {total}"
    elif layer == MapLayer.NOISE:
        alert_n = sum(1 for d in map_data.districts if d.noise_status == "violation")
        stat_label, stat_val = "規制超過", f"{alert_n} / {total}"
    elif layer == MapLayer.CROWD:
        alert_n = sum(1 for d in map_data.districts if d.crowd_level in ("crowded", "packed"))
        stat_label, stat_val = "混雑地区", f"{alert_n} / {total}"
    elif layer == MapLayer.ENERGY:
        alert_n = sum(1 for d in map_data.districts if d.energy_status in ("high", "critical"))
        stat_label, stat_val = "高消費地区", f"{alert_n} / {total}"
    elif layer == MapLayer.DISASTER:
        alert_n = sum(1 for d in map_data.districts if d.disaster_level in ("warning", "critical"))
        stat_label, stat_val = "警戒以上", f"{alert_n} / {total}"
    else:
        stat_label, stat_val = "地区数", str(total)

    # ── HTML ──────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenMythos City Map — {map_data.city}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  html,body{{width:100%;height:100%;font-family:'Google Sans',system-ui,'Segoe UI',sans-serif;background:#e8e0d0;overflow:hidden}}

  /* ── Map container ─────────────────────────── */
  #map-wrap{{position:absolute;inset:0;user-select:none}}
  #map-svg{{width:100%;height:100%;display:block;cursor:grab}}
  #map-svg:active{{cursor:grabbing}}

  /* ── Searchbar (top center) ──────────────────── */
  #searchbar{{
    position:absolute;top:14px;left:50%;transform:translateX(-50%);
    background:white;border-radius:24px;
    box-shadow:0 2px 10px rgba(0,0,0,0.2),0 0 0 1px rgba(0,0,0,0.06);
    padding:10px 18px;min-width:320px;
    display:flex;align-items:center;gap:10px;z-index:20;
  }}
  #searchbar .map-icon{{font-size:18px;opacity:.7}}
  #searchbar .title{{font-size:15px;font-weight:600;color:#202124}}
  #searchbar .sub{{font-size:12px;color:#5f6368;margin-left:auto;white-space:nowrap}}

  /* ── Right panel ─────────────────────────────── */
  #panel{{
    position:absolute;top:14px;right:14px;width:194px;z-index:20;
    display:flex;flex-direction:column;gap:10px;
  }}
  .card{{
    background:white;border-radius:12px;
    box-shadow:0 2px 8px rgba(0,0,0,0.18),0 0 0 1px rgba(0,0,0,0.05);
    overflow:hidden;
  }}
  .card-header{{
    padding:10px 14px 8px;border-bottom:1px solid #f1f3f4;
    font-size:11px;font-weight:600;color:#5f6368;text-transform:uppercase;letter-spacing:.06em;
  }}

  /* Layer buttons */
  .layer-btn{{
    display:flex;align-items:center;gap:10px;
    padding:9px 14px;border:none;background:transparent;
    width:100%;cursor:pointer;font-size:13px;color:#202124;
    border-bottom:1px solid #f1f3f4;transition:background .15s;
  }}
  .layer-btn:last-child{{border-bottom:none}}
  .layer-btn:hover{{background:#f8f9fa}}
  .layer-btn.active{{background:#e8f0fe;color:#1a73e8;font-weight:600}}
  .layer-btn .licon{{font-size:16px;width:20px;text-align:center}}
  .layer-btn .lcheck{{margin-left:auto;color:#1a73e8;font-size:16px;opacity:0}}
  .layer-btn.active .lcheck{{opacity:1}}

  /* Legend */
  .leg-row{{display:flex;align-items:center;gap:8px;padding:5px 14px;font-size:12px;color:#3c4043}}
  .leg-dot{{width:13px;height:13px;border-radius:3px;flex-shrink:0;border:1px solid rgba(0,0,0,.1)}}

  /* Stats */
  .stat-grid{{padding:10px 14px;display:grid;grid-template-columns:1fr 1fr;gap:8px}}
  .stat-item{{text-align:center}}
  .stat-val{{font-size:20px;font-weight:700;color:#1a73e8;line-height:1}}
  .stat-label{{font-size:10px;color:#5f6368;margin-top:2px}}

  /* ── Zoom controls ───────────────────────────── */
  #zoom-ctrl{{
    position:absolute;bottom:40px;right:14px;z-index:20;
    display:flex;flex-direction:column;gap:1px;
    box-shadow:0 2px 8px rgba(0,0,0,0.2);border-radius:4px;overflow:hidden;
  }}
  .zoom-btn{{
    background:white;border:none;cursor:pointer;
    width:40px;height:40px;font-size:20px;color:#5f6368;
    display:flex;align-items:center;justify-content:center;
    transition:background .15s;
  }}
  .zoom-btn:hover{{background:#f8f9fa}}

  /* ── Attribution ─────────────────────────────── */
  #attr{{
    position:absolute;bottom:6px;left:50%;transform:translateX(-50%);
    font-size:10px;color:#70757a;background:rgba(255,255,255,.75);
    padding:2px 8px;border-radius:3px;pointer-events:none;z-index:10;
  }}

  /* ── Tooltip ─────────────────────────────────── */
  #tooltip{{
    position:absolute;background:white;border-radius:10px;
    box-shadow:0 4px 20px rgba(0,0,0,0.25),0 0 0 1px rgba(0,0,0,0.06);
    padding:0;pointer-events:none;display:none;min-width:170px;z-index:50;
    font-size:13px;overflow:hidden;
  }}
  #tip-header{{background:#1a73e8;color:white;padding:9px 12px;font-weight:600;font-size:14px}}
  #tip-body{{padding:8px 12px}}
  .tip-row{{display:flex;justify-content:space-between;align-items:center;
    padding:3px 0;border-bottom:1px solid #f1f3f4;font-size:12px;color:#3c4043}}
  .tip-row:last-child{{border-bottom:none}}
  .tip-key{{color:#5f6368}}
  .tip-badge{{padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600;color:white}}
</style>
</head>
<body>

<!-- ── Map SVG ────────────────────────────────── -->
<div id="map-wrap">
<svg id="map-svg" viewBox="0 0 {vw} {vh}" xmlns="http://www.w3.org/2000/svg"
     preserveAspectRatio="xMidYMid meet">
  <defs>
    <!-- 陸地テクスチャ (非常に薄いドット) -->
    <pattern id="land-pat" x="0" y="0" width="4" height="4" patternUnits="userSpaceOnUse">
      <rect width="4" height="4" fill="#f5f0e8"/>
      <circle cx="2" cy="2" r="0.4" fill="#ebe5d8" opacity="0.6"/>
    </pattern>
    <!-- 水域グラデーション -->
    <linearGradient id="water-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#b8d8f0"/>
      <stop offset="100%" stop-color="#9dc8e8"/>
    </linearGradient>
    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="0" dy="0.3" stdDeviation="0.6" flood-color="rgba(0,0,0,0.15)"/>
    </filter>
  </defs>

  <!-- 陸地ベース -->
  <rect width="{vw}" height="{vh}" fill="url(#land-pat)"/>

  <!-- 東京湾 -->
  <polygon points="68,56 {vw},50 {vw},{vh} 60,{vh}"
           fill="url(#water-grad)" opacity="0.85"/>
  <!-- 東京湾ラベル -->
  <text x="84" y="68" text-anchor="middle" font-size="3.2"
        fill="#4a90c4" font-style="italic" font-family="system-ui,sans-serif" opacity="0.8">東京湾</text>

  <!-- 隅田川 -->
  <path d="M 67,16 Q 66,24 65,32 Q 64,42 65,50 Q 66,55 67,57"
        stroke="#9dc8e8" stroke-width="1.1" fill="none" stroke-linecap="round"/>
  <!-- 多摩川 -->
  <path d="M 1,68 Q 8,65 16,63 Q 26,61 36,60 Q 48,59 60,62 Q 64,63 68,65"
        stroke="#9dc8e8" stroke-width="0.8" fill="none" stroke-linecap="round"/>

  <!-- 公園エリア（代々木・皇居外苑など） -->
  <ellipse cx="29" cy="37" rx="3.5" ry="2.5" fill="#c8e6c9" opacity="0.6"/>
  <ellipse cx="42" cy="35" rx="4" ry="3" fill="#c8e6c9" opacity="0.55"/>

  <!-- 幹線道路ネットワーク -->
  <g stroke="#ffffff" stroke-width="0.55" fill="none" stroke-linecap="round" opacity="0.8">
    <!-- 環状7号線 (縦) -->
    <path d="M 24,1 L 24,62"/>
    <!-- 環状8号線 (縦) -->
    <path d="M 8,20 L 8,70"/>
    <!-- 甲州街道 (横) -->
    <path d="M 1,39 L 65,39"/>
    <!-- 明治通り (横) -->
    <path d="M 22,16 L 66,16"/>
    <!-- 山手通り (縦) -->
    <path d="M 43,1 L 43,60"/>
    <!-- 靖国通り (横) -->
    <path d="M 22,31 L 67,31"/>
    <!-- 蔵前橋通り (横) -->
    <path d="M 43,17 L 80,17"/>
  </g>

  <!-- 地区セル -->
  {svg_content}
</svg>
</div>

<!-- ── Search bar ─────────────────────────────── -->
<div id="searchbar">
  <span class="map-icon">🗺</span>
  <span class="title">{map_data.city} 都市マップ</span>
  <span class="sub">{layer_icon} {layer_label}レイヤー &nbsp;｜&nbsp; {total} 地区</span>
</div>

<!-- ── Right panel ────────────────────────────── -->
<div id="panel">
  <!-- Layer selector -->
  <div class="card">
    <div class="card-header">レイヤー選択</div>
    {''.join(
        f'<button class="layer-btn{" active" if MapLayer(lv)==layer else ""}" onclick="switchLayer(\'{lv}\')">'
        f'<span class="licon">{ic}</span><span>{lb}</span>'
        f'<span class="lcheck">✓</span></button>'
        for lv, ic, lb in [
            ("traffic","🚗","交通量"),
            ("noise","🔊","騒音"),
            ("crowd","👥","群衆密度"),
            ("energy","⚡","エネルギー"),
            ("disaster","⚠️","災害アラート"),
        ]
    )}
  </div>

  <!-- Legend -->
  <div class="card">
    <div class="card-header">凡例</div>
    {legend}
  </div>

  <!-- Stats -->
  <div class="card">
    <div class="card-header">統計</div>
    <div class="stat-grid">
      <div class="stat-item">
        <div class="stat-val">{total}</div>
        <div class="stat-label">地区数</div>
      </div>
      <div class="stat-item">
        <div class="stat-val" style="color:{'#d93025' if alert_n > 0 else '#34a853'}">{alert_n}</div>
        <div class="stat-label">{stat_label}</div>
      </div>
    </div>
  </div>
</div>

<!-- ── Zoom controls ───────────────────────────── -->
<div id="zoom-ctrl">
  <button class="zoom-btn" onclick="zoom(1.3)" title="拡大">+</button>
  <button class="zoom-btn" onclick="zoom(1/1.3)" title="縮小">−</button>
</div>

<!-- ── Attribution ────────────────────────────── -->
<div id="attr">© OpenMythos City Intelligence Platform</div>

<!-- ── Tooltip ────────────────────────────────── -->
<div id="tooltip">
  <div id="tip-header"></div>
  <div id="tip-body"></div>
</div>

<script>
const DISTRICTS = {district_data_json};
const LAYER_LABELS = {{
  traffic:"交通量", noise:"騒音", crowd:"群衆密度",
  energy:"エネルギー", disaster:"災害アラート"
}};
const STATUS_LABELS = {{
  clear:"通常走行", moderate:"やや混雑", congested:"渋滞", gridlock:"完全停滞",
  compliant:"規制内", near_limit:"規制近傍", violation:"規制超過",
  sparse:"閑散", normal:"通常", crowded:"混雑", packed:"超混雑",
  high:"高使用", critical:"異常",
  info:"情報", watch:"注意", warning:"警戒",
}};
const STATUS_COLORS = {{
  clear:"#34a853", moderate:"#fbbc04", congested:"#ea8600", gridlock:"#d93025",
  compliant:"#34a853", near_limit:"#fbbc04", violation:"#d93025",
  sparse:"#4285f4", normal:"#4285f4", crowded:"#ea8600", packed:"#d93025",
  high:"#fbbc04", critical:"#d93025",
  info:"#4285f4", watch:"#fbbc04", warning:"#ea8600",
}};

// ── Zoom / Pan ───────────────────────────────────────────────
let scale = 1, tx = 0, ty = 0;
let dragging = false, dragStart = {{x:0,y:0}}, dragOrigin = {{x:0,y:0}};
const svg = document.getElementById('map-svg');

function applyTransform(){{
  svg.style.transform = `translate(${{tx}}px,${{ty}}px) scale(${{scale}})`;
  svg.style.transformOrigin = 'center center';
}}
function zoom(factor){{
  scale = Math.max(0.6, Math.min(6, scale*factor));
  applyTransform();
}}
svg.addEventListener('wheel', e=>{{
  e.preventDefault();
  zoom(e.deltaY<0?1.12:1/1.12);
}}, {{passive:false}});
svg.addEventListener('mousedown', e=>{{
  if(e.button!==0) return;
  dragging=true;
  dragStart={{x:e.clientX,y:e.clientY}};
  dragOrigin={{x:tx,y:ty}};
}});
window.addEventListener('mousemove', e=>{{
  if(!dragging) return;
  tx = dragOrigin.x + (e.clientX-dragStart.x);
  ty = dragOrigin.y + (e.clientY-dragStart.y);
  applyTransform();
}});
window.addEventListener('mouseup', ()=>dragging=false);

// ── Tooltip ──────────────────────────────────────────────────
const tt = document.getElementById('tooltip');
const ttH = document.getElementById('tip-header');
const ttB = document.getElementById('tip-body');
let hideTimer;

function showTip(e, idx){{
  clearTimeout(hideTimer);
  const d = DISTRICTS[idx];
  ttH.textContent = d.name;
  const rows = [
    ['🚗 交通', d.traffic], ['🔊 騒音', d.noise],
    ['👥 群衆', d.crowd],   ['⚡ 電力', d.energy], ['⚠️ 災害', d.disaster]
  ].filter(r=>r[1]);
  ttB.innerHTML = rows.map(([k,v])=>{{
    const lbl = STATUS_LABELS[v]||v;
    const col = STATUS_COLORS[v]||'#5f6368';
    return `<div class="tip-row"><span class="tip-key">${{k}}</span>`+
           `<span class="tip-badge" style="background:${{col}}">${{lbl}}</span></div>`;
  }}).join('') || '<div style="padding:4px 0;color:#5f6368;font-size:12px">データなし</div>';
  tt.style.display='block';
  positionTip(e);
}}
function positionTip(e){{
  const x = Math.min(e.clientX+14, window.innerWidth-190);
  const y = Math.min(e.clientY-10, window.innerHeight-160);
  tt.style.left = x+'px';
  tt.style.top  = y+'px';
}}
document.getElementById('map-wrap').addEventListener('mousemove', e=>{{
  if(tt.style.display==='block') positionTip(e);
}});
function hideTip(){{
  hideTimer = setTimeout(()=>tt.style.display='none', 120);
}}
function selectDistrict(idx){{
  showTip({{clientX:window.innerWidth/2, clientY:window.innerHeight/2}}, idx);
}}

// ── Layer switch ─────────────────────────────────────────────
function switchLayer(lv){{
  const url = new URL(window.location.href);
  // API 呼び出しまたはページ再読み込み (サーバー接続時)
  // スタンドアロン HTML では UI のみ更新
  document.querySelectorAll('.layer-btn').forEach(b=>b.classList.remove('active'));
  event.currentTarget.classList.add('active');
  // 凡例・ヘッダー更新ヒント
  document.querySelector('#searchbar .sub').textContent =
    ({{traffic:'🚗',noise:'🔊',crowd:'👥',energy:'⚡',disaster:'⚠️'}}[lv]||'') +
    ' ' + (LAYER_LABELS[lv]||lv) + ' レイヤー | {total} 地区';
}}
</script>
</body>
</html>"""
    return html


# ─── Store / Builder ──────────────────────────────────────────────


class CityMapStore:
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


# ─── プリセット: 東京主要地区（地理的配置）───────────────────────────
#
#  viewBox 0 0 100 76  (東西:南北 ≈ 4:3)
#  北部 y=1, 北中部 y=17, 中部 y=31, 南部 y=47, 湾岸 y=62
#  東京湾: 右下 (x>68, y>56)  隅田川: x≈65 縦断

TOKYO_DISTRICTS: List[DistrictData] = [
    # ── 北部 (row 1: y=1) ─────────────────────────────────────────
    DistrictData("練馬",   x=1,  y=1,  width=21, height=15,
                 traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("板橋",   x=23, y=1,  width=19, height=14,
                 traffic_level="moderate",  noise_status="near_limit", crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("足立",   x=51, y=1,  width=22, height=15,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),
    DistrictData("葛飾",   x=74, y=1,  width=24, height=16,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),

    # ── 北中部 (row 2: y=16) ──────────────────────────────────────
    DistrictData("杉並",   x=1,  y=17, width=21, height=14,
                 traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("池袋",   x=23, y=16, width=19, height=15,
                 traffic_level="congested", noise_status="near_limit", crowd_level="crowded", energy_status="normal",   disaster_level=None),
    DistrictData("上野",   x=43, y=16, width=21, height=14,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("荒川",   x=65, y=17, width=18, height=13,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),
    DistrictData("墨田",   x=65, y=31, width=18, height=14,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),
    DistrictData("江戸川", x=84, y=17, width=15, height=28,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),

    # ── 中部 (row 3: y=31) ────────────────────────────────────────
    DistrictData("世田谷", x=1,  y=32, width=21, height=23,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("新宿",   x=23, y=31, width=19, height=15,
                 traffic_level="congested", noise_status="violation",  crowd_level="packed",  energy_status="high",     disaster_level=None),
    DistrictData("秋葉原", x=43, y=31, width=10, height=13,
                 traffic_level="moderate",  noise_status="near_limit", crowd_level="crowded", energy_status="high",     disaster_level=None),
    DistrictData("千代田", x=54, y=31, width=10, height=13,
                 traffic_level="gridlock",  noise_status="violation",  crowd_level="packed",  energy_status="critical", disaster_level="warning"),
    DistrictData("台東",   x=65, y=46, width=18, height=14,
                 traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("江東",   x=84, y=46, width=15, height=22,
                 traffic_level="moderate",  noise_status="near_limit", crowd_level="normal",  energy_status="high",     disaster_level="watch"),

    # ── 南部 (row 4: y=47) ────────────────────────────────────────
    DistrictData("目黒",   x=1,  y=56, width=21, height=13,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),
    DistrictData("渋谷",   x=23, y=47, width=19, height=13,
                 traffic_level="moderate",  noise_status="near_limit", crowd_level="crowded", energy_status="normal",   disaster_level=None),
    DistrictData("中央",   x=43, y=45, width=21, height=13,
                 traffic_level="congested", noise_status="near_limit", crowd_level="crowded", energy_status="high",     disaster_level=None),
    DistrictData("品川",   x=23, y=61, width=41, height=13,
                 traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
]
