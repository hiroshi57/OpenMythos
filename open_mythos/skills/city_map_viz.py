"""
Sprint 78A — 都市マップビジュアライゼーション  (v3: Three.js 3D)

Google Maps 風 3D ライトテーマ。Three.js によるブロック都市マップ。
レイヤー切替時に 3D ブロックの色と高さがリアルタイムで変化する。
"""
from __future__ import annotations

import json
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


# ─── カラーパレット（Google スタイル）─────────────────────────────

_TRAFFIC_COLORS = {
    "clear":     "#34a853",
    "moderate":  "#fbbc04",
    "congested": "#ea8600",
    "gridlock":  "#d93025",
}
_NOISE_COLORS = {
    "compliant":  "#34a853",
    "near_limit": "#fbbc04",
    "violation":  "#d93025",
}
_CROWD_COLORS = {
    "sparse":  "#e8f0fe",
    "normal":  "#4285f4",
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
_NO_DATA_COLOR = "#e8e0d0"

# ─── 3D ブロック高さ（値が高いほど深刻 → 高いブロック）──────────

_H_TRAFFIC  = {"clear": 3, "moderate": 8, "congested": 17, "gridlock": 27}
_H_NOISE    = {"compliant": 3, "near_limit": 10, "violation": 22}
_H_CROWD    = {"sparse": 3, "normal": 8, "crowded": 17, "packed": 27}
_H_ENERGY   = {"normal": 4, "high": 14, "critical": 25}
_H_DISASTER = {"info": 7, "watch": 13, "warning": 19, "critical": 30}


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
    """レイヤーに応じた地区の塗り色。None → _NO_DATA_COLOR。"""
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


def _district_3d_height(district: DistrictData, layer: MapLayer) -> float:
    """レイヤー×ステータスに応じた 3D ブロック高さ。"""
    if layer == MapLayer.TRAFFIC:
        return float(_H_TRAFFIC.get(district.traffic_level, 3))
    elif layer == MapLayer.NOISE:
        return float(_H_NOISE.get(district.noise_status, 3))
    elif layer == MapLayer.CROWD:
        return float(_H_CROWD.get(district.crowd_level, 3))
    elif layer == MapLayer.ENERGY:
        return float(_H_ENERGY.get(district.energy_status, 4))
    elif layer == MapLayer.DISASTER:
        return float(_H_DISASTER.get(district.disaster_level, 3))
    return 4.0


def _legend_html(layer: MapLayer) -> str:
    if layer == MapLayer.TRAFFIC:
        items = [("#34a853","通常走行"),("#fbbc04","やや混雑"),("#ea8600","渋滞"),("#d93025","完全停滞")]
    elif layer == MapLayer.NOISE:
        items = [("#34a853","規制内"),("#fbbc04","規制値近傍"),("#d93025","規制超過")]
    elif layer == MapLayer.CROWD:
        items = [("#e8f0fe","閑散"),("#4285f4","通常"),("#ea8600","混雑"),("#d93025","超混雑")]
    elif layer == MapLayer.ENERGY:
        items = [("#34a853","通常"),("#fbbc04","高使用"),("#d93025","異常")]
    elif layer == MapLayer.DISASTER:
        items = [("#e8f5e9","アラートなし"),("#4285f4","情報"),("#fbbc04","注意"),("#ea8600","警戒"),("#d93025","緊急")]
    else:
        items = []
    return "".join(
        f'<div class="leg-row"><span class="leg-dot" style="background:{c}"></span><span>{lb}</span></div>'
        for c, lb in items
    )


# ─── HTML テンプレート（Three.js 3D）─────────────────────────────
# __PLACEHOLDER__ マーカーを generate_html() で .replace() 置換する

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenMythos 3D City Map — __CITY_NAME__</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;overflow:hidden;font-family:'Google Sans',system-ui,'Segoe UI',sans-serif;background:#d0c8bc}
#canvas-wrap{position:absolute;inset:0}
canvas{display:block}

/* ── Searchbar ── */
#sb{position:absolute;top:14px;left:50%;transform:translateX(-50%);background:white;border-radius:28px;
z-index:30;box-shadow:0 2px 14px rgba(0,0,0,.26),0 0 0 1px rgba(0,0,0,.06);
padding:11px 22px;display:flex;align-items:center;gap:12px;min-width:360px}
#sb .mi{font-size:18px} #sb .ti{font-size:15px;font-weight:600;color:#202124}
#sb .su{font-size:12px;color:#5f6368;margin-left:auto;white-space:nowrap}

/* ── Panel ── */
#panel{position:absolute;top:14px;right:14px;width:196px;z-index:30;display:flex;flex-direction:column;gap:10px}
.card{background:white;border-radius:12px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,.2),0 0 0 1px rgba(0,0,0,.05)}
.ch{padding:10px 14px 8px;border-bottom:1px solid #f1f3f4;font-size:11px;font-weight:700;color:#5f6368;text-transform:uppercase;letter-spacing:.07em}
.lb{display:flex;align-items:center;gap:10px;padding:9px 14px;border:none;background:transparent;
width:100%;cursor:pointer;font-size:13px;color:#202124;border-bottom:1px solid #f1f3f4;
transition:background .12s;text-align:left}
.lb:last-child{border-bottom:none} .lb:hover{background:#f8f9fa}
.lb.active{background:#e8f0fe;color:#1a73e8;font-weight:600}
.lb .li{font-size:15px;width:22px;text-align:center;flex-shrink:0}
.lb .lc{margin-left:auto;color:#1a73e8;font-size:14px;opacity:0} .lb.active .lc{opacity:1}
.leg-row{display:flex;align-items:center;gap:8px;padding:5px 14px;font-size:12px;color:#3c4043}
.leg-dot{width:13px;height:13px;border-radius:3px;flex-shrink:0;border:1px solid rgba(0,0,0,.12)}
.sg{padding:10px 14px;display:grid;grid-template-columns:1fr 1fr;gap:8px}
.si{text-align:center} .sv{font-size:22px;font-weight:700;color:#1a73e8;line-height:1.1}
.sl{font-size:10px;color:#5f6368;margin-top:2px}

/* ── Map controls ── */
#mc{position:absolute;bottom:40px;right:14px;z-index:30;display:flex;flex-direction:column;gap:8px;align-items:center}
.cb{background:white;border:none;cursor:pointer;width:40px;height:40px;border-radius:6px;
font-size:18px;color:#5f6368;display:flex;align-items:center;justify-content:center;
box-shadow:0 2px 8px rgba(0,0,0,.2);transition:background .12s} .cb:hover{background:#f8f9fa}
#zw{display:flex;flex-direction:column;gap:1px;border-radius:6px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.2)}
#zw .cb{box-shadow:none;border-radius:0}

/* ── Labels ── */
.dl{position:absolute;pointer-events:none;z-index:20;font-size:11px;font-weight:600;color:#202124;
text-shadow:0 0 5px white,0 0 5px white,0 0 5px white;transform:translate(-50%,-50%);white-space:nowrap}

/* ── Tooltip ── */
#tt{position:absolute;z-index:50;pointer-events:none;display:none;background:white;border-radius:12px;
min-width:185px;overflow:hidden;box-shadow:0 6px 24px rgba(0,0,0,.28),0 0 0 1px rgba(0,0,0,.06)}
#tth{background:#1a73e8;color:white;padding:10px 14px;font-weight:700;font-size:15px}
#ttb{padding:8px 12px}
.tr{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid #f1f3f4;font-size:12px}
.tr:last-child{border-bottom:none} .tk{color:#5f6368}
.tbg{padding:2px 9px;border-radius:10px;font-size:11px;font-weight:600;color:white}

/* ── Height hint ── */
#hh{position:absolute;bottom:14px;left:14px;z-index:30;background:rgba(255,255,255,.92);
border-radius:8px;padding:8px 12px;box-shadow:0 2px 8px rgba(0,0,0,.15);font-size:11px;color:#5f6368}
#hh strong{color:#202124;display:block;margin-bottom:3px}

/* ── Attribution ── */
#at{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);z-index:10;
font-size:10px;color:#70757a;background:rgba(255,255,255,.75);padding:2px 10px;border-radius:3px}

/* ── Loading ── */
#ld{position:absolute;inset:0;z-index:100;background:rgba(255,255,255,.95);
display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px}
.sp{width:38px;height:38px;border:3px solid #e8f0fe;border-top-color:#1a73e8;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#ld p{color:#5f6368;font-size:14px;font-weight:500}
</style>
</head>
<body>
<div id="canvas-wrap"></div>

<div id="sb">
  <span class="mi">🗺</span>
  <span class="ti">__CITY_NAME__ City Map</span>
  <span class="su" id="sub">__LAYER_ICON__ __LAYER_LABEL__ &nbsp;|&nbsp; __DISTRICT_COUNT__ 地区</span>
</div>

<div id="panel">
  <div class="card"><div class="ch">レイヤー選択</div>__LAYER_BTNS__</div>
  <div class="card"><div class="ch">凡例</div><div id="lgd">__LEGEND_HTML__</div></div>
  <div class="card"><div class="ch">統計</div>
    <div class="sg">
      <div class="si"><div class="sv">__DISTRICT_COUNT__</div><div class="sl">地区数</div></div>
      <div class="si"><div class="sv" id="av" style="color:__ALERT_COLOR__">__ALERT_COUNT__</div><div class="sl" id="al">__ALERT_LABEL__</div></div>
    </div>
  </div>
</div>

<div id="mc">
  <div class="cb" onclick="resetCam()" title="カメラリセット" style="font-size:20px">🧭</div>
  <div id="zw">
    <button class="cb" onclick="doZoom(0.78)">+</button>
    <button class="cb" onclick="doZoom(1.28)">−</button>
  </div>
</div>

<div id="labels"></div>
<div id="tt"><div id="tth"></div><div id="ttb"></div></div>

<div id="hh"><strong>📊 高さ = データ強度</strong>ブロックが高いほど値が大きい</div>
<div id="at">© OpenMythos City Intelligence &nbsp;｜&nbsp; ドラッグ: 回転 &nbsp;/&nbsp; スクロール: ズーム &nbsp;/&nbsp; 右ドラッグ: 移動</div>
<div id="ld"><div class="sp"></div><p>3D マップを読み込み中...</p></div>

<!-- ========== Data ========== -->
<script>
var DISTRICTS = __DISTRICTS_JSON__;
var ACTIVE_LAYER = "__ACTIVE_LAYER__";
var LAYER_META = {
  traffic: {icon:"🚗",label:"交通量",   stat:"混雑地区",  keys:["congested","gridlock"],field:"traffic"},
  noise:   {icon:"🔊",label:"騒音",      stat:"規制超過",  keys:["violation"],           field:"noise"},
  crowd:   {icon:"👥",label:"群衆密度",  stat:"混雑地区",  keys:["crowded","packed"],    field:"crowd"},
  energy:  {icon:"⚡",label:"エネルギー",stat:"高消費地区",keys:["high","critical"],     field:"energy"},
  disaster:{icon:"⚠️",label:"災害アラート",stat:"警戒以上",keys:["warning","critical"],  field:"disaster"}
};
var SL={clear:"通常走行",moderate:"やや混雑",congested:"渋滞",gridlock:"完全停滞",
        compliant:"規制内",near_limit:"規制近傍",violation:"規制超過",
        sparse:"閑散",normal:"通常",crowded:"混雑",packed:"超混雑",
        high:"高使用",critical:"異常",info:"情報",watch:"注意",warning:"警戒"};
var SC={clear:"#34a853",moderate:"#fbbc04",congested:"#ea8600",gridlock:"#d93025",
        compliant:"#34a853",near_limit:"#fbbc04",violation:"#d93025",
        sparse:"#4285f4",normal:"#4285f4",crowded:"#ea8600",packed:"#d93025",
        high:"#fbbc04",critical:"#d93025",info:"#4285f4",watch:"#fbbc04",warning:"#ea8600"};
var LEGENDS={
  traffic: [["#34a853","通常走行"],["#fbbc04","やや混雑"],["#ea8600","渋滞"],["#d93025","完全停滞"]],
  noise:   [["#34a853","規制内"],["#fbbc04","規制値近傍"],["#d93025","規制超過"]],
  crowd:   [["#e8f0fe","閑散"],["#4285f4","通常"],["#ea8600","混雑"],["#d93025","超混雑"]],
  energy:  [["#34a853","通常"],["#fbbc04","高使用"],["#d93025","異常"]],
  disaster:[["#e8f5e9","アラートなし"],["#4285f4","情報"],["#fbbc04","注意"],["#ea8600","警戒"],["#d93025","緊急"]]
};
</script>

<!-- ========== Three.js ========== -->
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.min.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Renderer ────────────────────────────────────────────────────
const wrap = document.getElementById('canvas-wrap');
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(wrap.clientWidth, wrap.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
wrap.appendChild(renderer.domElement);

// ── Scene ───────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd8d0c4);
scene.fog = new THREE.FogExp2(0xd8d0c4, 0.0038);

// ── Camera ──────────────────────────────────────────────────────
const CAM0 = new THREE.Vector3(50, 56, 120);
const TGT0 = new THREE.Vector3(50, 0, 38);
const camera = new THREE.PerspectiveCamera(45, wrap.clientWidth/wrap.clientHeight, 0.1, 600);
camera.position.copy(CAM0);

// ── Lights ──────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0xffffff, 1.3));
const sun = new THREE.DirectionalLight(0xfff8e0, 2.2);
sun.position.set(40, 90, 65); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, {near:1,far:280,left:-90,right:90,top:90,bottom:-90});
scene.add(sun);
scene.add(new THREE.HemisphereLight(0x90c8f4, 0xd8c8a8, 0.8));

// ── Ground ──────────────────────────────────────────────────────
const gnd = new THREE.Mesh(
  new THREE.PlaneGeometry(260, 220),
  new THREE.MeshLambertMaterial({color:0xf0ebe0})
);
gnd.rotation.x = -Math.PI/2; gnd.position.set(50,-0.02,38); gnd.receiveShadow=true;
scene.add(gnd);

// ── Tokyo Bay ───────────────────────────────────────────────────
const bvs=[68,56, 99,50, 99,76, 60,76];
const bs=new THREE.Shape(); bs.moveTo(bvs[0],bvs[1]);
for(let i=2;i<bvs.length;i+=2) bs.lineTo(bvs[i],bvs[i+1]); bs.closePath();
const bay=new THREE.Mesh(new THREE.ShapeGeometry(bs),
  new THREE.MeshLambertMaterial({color:0x5ba8d4,transparent:true,opacity:0.72,side:THREE.DoubleSide}));
bay.rotation.x=-Math.PI/2; bay.position.y=0.03; scene.add(bay);

function makeSprite(txt,color,fs){
  const c=document.createElement('canvas'); c.width=200; c.height=52;
  const ctx=c.getContext('2d'); ctx.font=`italic ${fs}px system-ui`;
  ctx.fillStyle=color; ctx.fillText(txt,6,fs);
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true}));
  s.scale.set(18,4.5,1); return s;
}
const bsp=makeSprite('東京湾','#2a78a8',30); bsp.position.set(83,1.5,65); scene.add(bsp);

// ── Animated Rivers ─────────────────────────────────────────────
const riverUniforms={time:{value:0}};
const riverMat=new THREE.ShaderMaterial({
  uniforms:riverUniforms,
  vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`,
  fragmentShader:`uniform float time;varying vec2 vUv;void main(){float w=sin(vUv.x*5.0-time*2.0)*0.12+sin(vUv.y*3.0-time*1.5)*0.08;vec3 c=mix(vec3(0.38,0.67,0.84),vec3(0.22,0.51,0.72),w+0.5);gl_FragColor=vec4(c,0.75);}`,
  transparent:true,side:THREE.DoubleSide,depthWrite:false
});
// 隅田川 (N-S, x≈64.5)
const rv1=new THREE.Mesh(new THREE.PlaneGeometry(2,30),riverMat); rv1.rotation.x=-Math.PI/2; rv1.position.set(64.5,0.05,30); scene.add(rv1);
// 多摩川 (E-W, z≈63)
const rv2=new THREE.Mesh(new THREE.PlaneGeometry(52,2.2),riverMat); rv2.rotation.x=-Math.PI/2; rv2.position.set(26,0.05,63); scene.add(rv2);
// 荒川 (N-S, x≈50.5)
const rv3=new THREE.Mesh(new THREE.PlaneGeometry(1.5,27),riverMat); rv3.rotation.x=-Math.PI/2; rv3.position.set(50.5,0.05,14); scene.add(rv3);

// ── Parks & Trees ────────────────────────────────────────────────
function seededR(seed){let s=seed;return ()=>{s=(s*1664525+1013904223)>>>0;return s/4294967296;};}
const pkm=new THREE.MeshLambertMaterial({color:0x8dc87a,transparent:true,opacity:0.55});
const treeM=new THREE.MeshLambertMaterial({color:0x4a8840});
const trng=seededR(77);
[[29,39,4.5],[54,36,3.5],[6,42,3.0],[44,35,2.5],[58,22,2.0]].forEach(([px,pz,r])=>{
  const pk=new THREE.Mesh(new THREE.CylinderGeometry(r,r,0.12,20),pkm); pk.position.set(px,0.05,pz); scene.add(pk);
  for(let i=0;i<5;i++){
    const tx=px+(trng()-0.5)*r*1.6,tz=pz+(trng()-0.5)*r*1.6,th=1.0+trng()*1.0;
    const tree=new THREE.Mesh(new THREE.ConeGeometry(0.55,th,6),treeM); tree.position.set(tx,th/2,tz); scene.add(tree);
  }
});

// ── 富士山・背景の丘 ──────────────────────────────────────────────
const fuji=new THREE.Mesh(new THREE.ConeGeometry(22,32,8),new THREE.MeshLambertMaterial({color:0x8899a6}));
fuji.position.set(-32,16,-26); scene.add(fuji);
const fujiSnow=new THREE.Mesh(new THREE.ConeGeometry(6,10,8),new THREE.MeshLambertMaterial({color:0xeef2f6}));
fujiSnow.position.set(-32,30,-26); scene.add(fujiSnow);
[[-14,5,-22,0x799e78],[6,4,-24,0x6a8e70],[28,3.5,-20,0x7a9e80],[-3,3,-14,0x8aae90],[55,3.5,-22,0x799e78]].forEach(([hx,hh,hz,hc])=>{
  const hill=new THREE.Mesh(new THREE.SphereGeometry(hh*3,7,5,0,Math.PI*2,0,Math.PI/2),new THREE.MeshLambertMaterial({color:hc}));
  hill.position.set(hx,0,hz); scene.add(hill);
});

// ── 背景ビル (InstancedMesh) ──────────────────────────────────────
const brng=seededR(42);
const NBLD=320;
const ibld=new THREE.InstancedMesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshLambertMaterial({color:0xbab4a8}),NBLD);
const dummy=new THREE.Object3D(); let bi=0;
for(let at=0;at<2500&&bi<NBLD;at++){
  const bx=brng()*97+1,bz=brng()*71+1,bw=brng()*1.6+0.5,bd=brng()*1.6+0.5,bh=brng()*5+0.5;
  let ok=true;
  for(const dd of DISTRICTS){if(bx>dd.x-0.5&&bx<dd.x+dd.w+0.5&&bz>dd.z-0.5&&bz<dd.z+dd.d+0.5){ok=false;break;}}
  if(Math.abs(bx-64.5)<2.5||Math.abs(bx-50.5)<2.5||(bz>61&&bz<65)) ok=false;
  if(bx>63&&bz>53) ok=false;
  if(!ok) continue;
  dummy.position.set(bx,bh/2,bz); dummy.scale.set(bw,bh,bd); dummy.updateMatrix();
  ibld.setMatrixAt(bi++,dummy.matrix);
}
ibld.count=bi; ibld.castShadow=true; ibld.instanceMatrix.needsUpdate=true; scene.add(ibld);

// ── ランドマーク ──────────────────────────────────────────────────
const _lm=(geo,col,x,y,z)=>{const m=new THREE.Mesh(geo,new THREE.MeshLambertMaterial({color:col}));m.position.set(x,y,z);m.castShadow=true;scene.add(m);};
_lm(new THREE.BoxGeometry(2,22,2),     0x7888a0, 29,11,35);   // 東京都庁
_lm(new THREE.ConeGeometry(0.55,18,4), 0xe05020, 43,9, 48);   // 東京タワー
_lm(new THREE.ConeGeometry(0.45,26,4), 0x5878b8, 71,13,22);   // 東京スカイツリー
_lm(new THREE.BoxGeometry(16,0.04,1.2),0xb0a898, 80,0.04,51); // 羽田滑走路1
_lm(new THREE.BoxGeometry(16,0.04,1.2),0xb0a898, 80,0.04,54); // 羽田滑走路2

// ── 電車路線 & 電車アニメーション ─────────────────────────────────
const trainClock=new THREE.Clock();
const trainObjs=[];
function makeRail(pts,col,closed,rr=0.28){
  const curve=new THREE.CatmullRomCurve3(pts.map(p=>new THREE.Vector3(p[0],0.3,p[1])),closed);
  scene.add(new THREE.Mesh(new THREE.TubeGeometry(curve,pts.length*6,rr,5,closed),new THREE.MeshLambertMaterial({color:col})));
  return curve;
}
// 山手線 (緑ループ)
const yC=makeRail([[29,14],[29,28],[29,35],[29,42],[30,52],[36,58],[44,60],[52,56],[50,48],[50,35],[50,28],[44,14],[36,11]],0x80bb44,true);
// 中央線 (オレンジ E-W)
const cC=makeRail([[3,30],[10,30],[22,30],[29,30],[37,30],[44,30],[52,30],[62,30]],0xf07020,false,0.22);
// 東海道線 (赤 N-S)
const tC=makeRail([[44,29],[44,36],[44,43],[44,50],[44,57]],0xd02030,false,0.22);
[[yC,0x3d9e3d,0.0,true],[cC,0xf07020,0.3,false],[tC,0xc02828,0.7,false]].forEach(([curve,col,phase,cl])=>{
  const m=new THREE.Mesh(new THREE.BoxGeometry(2.2,0.7,0.9),new THREE.MeshLambertMaterial({color:col}));
  m.castShadow=true; scene.add(m); trainObjs.push({m,curve,t:phase,cl});
});

// ── Districts ────────────────────────────────────────────────────
let currentLayer = ACTIVE_LAYER;
const meshes = [];

DISTRICTS.forEach((d,i)=>{
  const ld=d.layers[currentLayer];
  const geo=new THREE.BoxGeometry(d.w-0.5, ld.h, d.d-0.5);
  const mat=new THREE.MeshLambertMaterial({color:ld.c, transparent:true, opacity:0.88});
  const mesh=new THREE.Mesh(geo,mat);
  mesh.position.set(d.x+d.w/2, ld.h/2, d.z+d.d/2);
  mesh.castShadow=true; mesh.receiveShadow=true;
  mesh.userData={idx:i};
  scene.add(mesh); meshes.push(mesh);
});

// ── HTML Labels (3D→2D projection) ──────────────────────────────
const labDiv=document.getElementById('labels');
const lEls=DISTRICTS.map((d,i)=>{
  const el=document.createElement('div'); el.className='dl';
  el.textContent=d.name; labDiv.appendChild(el); return el;
});
const _v3=new THREE.Vector3();
function updateLabels(){
  const rect=renderer.domElement.getBoundingClientRect();
  DISTRICTS.forEach((d,i)=>{
    const m=meshes[i];
    _v3.set(m.position.x, m.position.y*2+2.5, m.position.z).project(camera);
    if(_v3.z>1){lEls[i].style.display='none';return;}
    const sx=(_v3.x*0.5+0.5)*rect.width+rect.left;
    const sy=(-_v3.y*0.5+0.5)*rect.height+rect.top;
    lEls[i].style.display='block';
    lEls[i].style.left=sx+'px'; lEls[i].style.top=sy+'px';
    const dist=camera.position.distanceTo(m.position);
    lEls[i].style.fontSize=Math.max(8,Math.min(14,700/dist))+'px';
    lEls[i].style.opacity=Math.max(0,Math.min(1,(160-dist)/80));
  });
}

// ── OrbitControls ────────────────────────────────────────────────
const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true; controls.dampingFactor=0.07;
controls.minPolarAngle=0.1; controls.maxPolarAngle=Math.PI/2.05;
controls.minDistance=18; controls.maxDistance=270;
controls.target.copy(TGT0); controls.update();

window.resetCam=()=>{ camera.position.copy(CAM0); controls.target.copy(TGT0); controls.update(); };
window.doZoom=(f)=>{ const d=camera.position.clone().sub(controls.target); camera.position.copy(controls.target).addScaledVector(d,f); controls.update(); };

// ── Raycasting (hover) ───────────────────────────────────────────
const raycaster=new THREE.Raycaster(), mouse=new THREE.Vector2();
const tt=document.getElementById('tt'), tth=document.getElementById('tth'), ttb=document.getElementById('ttb');
let hovIdx=-1;

renderer.domElement.addEventListener('mousemove',e=>{
  const rect=renderer.domElement.getBoundingClientRect();
  mouse.set(((e.clientX-rect.left)/rect.width)*2-1, -((e.clientY-rect.top)/rect.height)*2+1);
  raycaster.setFromCamera(mouse,camera);
  const hits=raycaster.intersectObjects(meshes);
  if(hits.length){
    const idx=hits[0].object.userData.idx;
    if(idx!==hovIdx){
      if(hovIdx>=0) meshes[hovIdx].material.emissive.setHex(0x000000);
      hovIdx=idx;
      meshes[idx].material.emissive.setHex(0x333333);
      meshes[idx].material.emissiveIntensity=0.28;
    }
    const d=DISTRICTS[idx];
    tth.textContent=d.name;
    const rows=[['🚗交通',d.traffic],['🔊騒音',d.noise],['👥群衆',d.crowd],['⚡電力',d.energy],['⚠️災害',d.disaster]].filter(r=>r[1]);
    ttb.innerHTML=rows.map(([k,v])=>`<div class="tr"><span class="tk">${k}</span><span class="tbg" style="background:${SC[v]||'#5f6368'}">${SL[v]||v}</span></div>`).join('')
      ||'<div style="padding:5px 0;color:#5f6368;font-size:12px">データなし</div>';
    tt.style.cssText=`left:${Math.min(e.clientX+16,window.innerWidth-200)}px;top:${Math.min(e.clientY-10,window.innerHeight-170)}px;display:block`;
    renderer.domElement.style.cursor='pointer';
  } else {
    if(hovIdx>=0){meshes[hovIdx].material.emissive.setHex(0x000000);hovIdx=-1;}
    tt.style.display='none'; renderer.domElement.style.cursor='grab';
  }
});
renderer.domElement.addEventListener('mouseleave',()=>tt.style.display='none');

// ── Layer switching (色と高さを同時に更新) ────────────────────────
window.switchLayer=function(lv, btn){
  currentLayer=lv;

  // ★ 3D メッシュの色と高さを全地区更新
  DISTRICTS.forEach((d,i)=>{
    const ld=d.layers[lv];
    meshes[i].material.color.setHex(ld.c);           // 色を変更
    meshes[i].geometry.dispose();
    meshes[i].geometry=new THREE.BoxGeometry(d.w-0.5, ld.h, d.d-0.5);  // 高さを変更
    meshes[i].position.y=ld.h/2;                     // y座標をブロック中心に合わせる
  });

  // ボタン状態
  document.querySelectorAll('.lb').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');

  // ヘッダー
  const m=LAYER_META[lv];
  document.getElementById('sub').textContent=m.icon+' '+m.label+' | '+DISTRICTS.length+' 地区';

  // 凡例
  document.getElementById('lgd').innerHTML=(LEGENDS[lv]||[]).map(([c,lb])=>
    `<div class="leg-row"><span class="leg-dot" style="background:${c}"></span><span>${lb}</span></div>`
  ).join('');

  // 統計
  const n=DISTRICTS.filter(d=>m.keys.includes(d[m.field])).length;
  const av=document.getElementById('av'); av.textContent=n;
  av.style.color=n>0?'#d93025':'#34a853';
  document.getElementById('al').textContent=m.stat;
};

// ── Resize ───────────────────────────────────────────────────────
window.addEventListener('resize',()=>{
  const W=wrap.clientWidth,H=wrap.clientHeight;
  camera.aspect=W/H; camera.updateProjectionMatrix(); renderer.setSize(W,H);
});

// ── Render loop ──────────────────────────────────────────────────
(function animate(){
  requestAnimationFrame(animate);
  const dt=Math.min(trainClock.getDelta(),0.05);
  controls.update();
  // 川アニメーション
  riverUniforms.time.value+=dt;
  // 電車移動
  trainObjs.forEach(o=>{
    o.t+=0.025*dt;
    const tp=o.cl?o.t%1:Math.abs((o.t%2)-1);
    const p=o.curve.getPointAt(tp);
    const tn=o.curve.getTangentAt(tp);
    o.m.position.copy(p);
    o.m.lookAt(p.clone().add(tn));
  });
  renderer.render(scene,camera);
  updateLabels();
})();

document.getElementById('ld').style.display='none';
</script>
</body>
</html>"""


# ─── generate_html ────────────────────────────────────────────────


def generate_html(map_data: CityMapData, width: int = 1000, height: int = 700) -> str:
    """Three.js 3D 都市マップ HTML を生成する。レイヤー切替で色・高さが変化。"""
    layer = map_data.active_layer

    layer_label = {
        MapLayer.TRAFFIC:"交通量", MapLayer.NOISE:"騒音",
        MapLayer.CROWD:"群衆密度", MapLayer.ENERGY:"エネルギー",
        MapLayer.DISASTER:"災害アラート",
    }.get(layer, layer.value)

    layer_icon = {
        MapLayer.TRAFFIC:"🚗", MapLayer.NOISE:"🔊",
        MapLayer.CROWD:"👥", MapLayer.ENERGY:"⚡", MapLayer.DISASTER:"⚠️",
    }.get(layer, "📍")

    total = len(map_data.districts)

    # 統計
    if layer == MapLayer.TRAFFIC:
        alert_n = sum(1 for d in map_data.districts if d.traffic_level in ("congested","gridlock"))
        stat_label = "混雑地区"
    elif layer == MapLayer.NOISE:
        alert_n = sum(1 for d in map_data.districts if d.noise_status == "violation")
        stat_label = "規制超過"
    elif layer == MapLayer.CROWD:
        alert_n = sum(1 for d in map_data.districts if d.crowd_level in ("crowded","packed"))
        stat_label = "混雑地区"
    elif layer == MapLayer.ENERGY:
        alert_n = sum(1 for d in map_data.districts if d.energy_status in ("high","critical"))
        stat_label = "高消費地区"
    elif layer == MapLayer.DISASTER:
        alert_n = sum(1 for d in map_data.districts if d.disaster_level in ("warning","critical"))
        stat_label = "警戒以上"
    else:
        alert_n = 0; stat_label = "地区数"

    alert_color = "#d93025" if alert_n > 0 else "#34a853"

    # レイヤーボタン HTML (onclick に button 自身を渡す → switchLayer で active 管理)
    layer_btns = "".join(
        f'<button class="lb{"  active" if MapLayer(lv) == layer else ""}" '
        f'onclick="switchLayer(\'{lv}\',this)">'
        f'<span class="li">{ic}</span><span>{lb}</span>'
        f'<span class="lc">✓</span></button>'
        for lv, ic, lb in [
            ("traffic","🚗","交通量"), ("noise","🔊","騒音"),
            ("crowd","👥","群衆密度"), ("energy","⚡","エネルギー"),
            ("disaster","⚠️","災害アラート"),
        ]
    )

    legend = _legend_html(layer)

    # ── 全レイヤーの色・高さを地区ごとに事前計算して JS に埋め込む ──
    dist_data = []
    for d in map_data.districts:
        layers_js: Dict[str, dict] = {}
        for lyr in MapLayer:
            c = _district_color(d, lyr)
            h = _district_3d_height(d, lyr)
            layers_js[lyr.value] = {"c": int(c.lstrip("#"), 16), "h": h}
        dist_data.append({
            "name": d.name,
            "x": d.x, "z": d.y, "w": d.width, "d": d.height,
            "traffic":  d.traffic_level  or "",
            "noise":    d.noise_status   or "",
            "crowd":    d.crowd_level    or "",
            "energy":   d.energy_status  or "",
            "disaster": d.disaster_level or "",
            "layers":   layers_js,
        })

    dist_json = json.dumps(dist_data, ensure_ascii=False)

    return (
        _PAGE_TEMPLATE
        .replace("__CITY_NAME__",      map_data.city)
        .replace("__LAYER_LABEL__",    layer_label)
        .replace("__LAYER_ICON__",     layer_icon)
        .replace("__DISTRICT_COUNT__", str(total))
        .replace("__ALERT_COUNT__",    str(alert_n))
        .replace("__ALERT_COLOR__",    alert_color)
        .replace("__ALERT_LABEL__",    stat_label)
        .replace("__DISTRICTS_JSON__", dist_json)
        .replace("__ACTIVE_LAYER__",   layer.value)
        .replace("__LEGEND_HTML__",    legend)
        .replace("__LAYER_BTNS__",     layer_btns)
    )


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


# ─── プリセット: 東京 23 区（全特別区・地理的配置）────────────────
#
#  ※ 東京特別区は 23 区（24 区ではない）
#  viewBox 0 0 100 76  /  Three.js 世界座標: x=0-99, z=0-75
#
#  レイアウト（行 × 列）:
#   row1(y=1) : 練馬 板橋 北区 足立[wide] 葛飾
#   row2(y=15): 杉並 豊島 文京 荒川 墨田 江戸川[tall]
#   row3(y=29): 中野 新宿 千代田 台東 江東
#   row4(y=43): 世田谷[tall] 渋谷 港 中央 大田[wide+tall]
#   row5(y=56): 目黒 品川[wide]
#
#  東京湾: 右下 (x>65, z>55)  /  隅田川: x≈64.5 縦断  /  多摩川: z≈63 横断

TOKYO_DISTRICTS: List[DistrictData] = [

    # ── 北部 row 1 (y=1) ──────────────────────────────────────────
    DistrictData("練馬",   x=1,  y=1,  width=20, height=13,
                 traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("板橋",   x=22, y=1,  width=14, height=13,
                 traffic_level="moderate",  noise_status="near_limit", crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("北区",   x=37, y=1,  width=13, height=13,
                 traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("足立",   x=51, y=1,  width=27, height=14,   # 大きい区
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),
    DistrictData("葛飾",   x=79, y=1,  width=20, height=14,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),

    # ── 北中部 row 2 (y=15) ───────────────────────────────────────
    DistrictData("杉並",   x=1,  y=15, width=20, height=13,
                 traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("豊島",   x=22, y=14, width=14, height=14,   # 池袋エリア
                 traffic_level="congested", noise_status="near_limit", crowd_level="crowded", energy_status="normal",   disaster_level=None),
    DistrictData("文京",   x=37, y=15, width=13, height=13,   # 東大・東京ドーム
                 traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("荒川",   x=51, y=15, width=13, height=13,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),
    DistrictData("墨田",   x=65, y=15, width=13, height=13,
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),
    DistrictData("江戸川", x=79, y=15, width=20, height=27,   # 東端・広大
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),

    # ── 中部 row 3 (y=29) ─────────────────────────────────────────
    DistrictData("中野",   x=1,  y=29, width=20, height=13,   # 杉並の東
                 traffic_level="clear",     noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("新宿",   x=22, y=28, width=14, height=14,   # 西新宿・歌舞伎町
                 traffic_level="congested", noise_status="violation",  crowd_level="packed",  energy_status="high",     disaster_level=None),
    DistrictData("千代田", x=37, y=28, width=13, height=13,   # 皇居・丸の内
                 traffic_level="gridlock",  noise_status="violation",  crowd_level="packed",  energy_status="critical", disaster_level="warning"),
    DistrictData("台東",   x=51, y=28, width=13, height=13,   # 浅草・上野
                 traffic_level="moderate",  noise_status="near_limit", crowd_level="crowded", energy_status="normal",   disaster_level=None),
    DistrictData("江東",   x=65, y=28, width=13, height=14,   # 湾岸・お台場隣接
                 traffic_level="moderate",  noise_status="near_limit", crowd_level="normal",  energy_status="high",     disaster_level="watch"),

    # ── 南中部 row 4 (y=42-43) ────────────────────────────────────
    DistrictData("世田谷", x=1,  y=43, width=20, height=22,   # 最大区・住宅地
                 traffic_level="clear",     noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
    DistrictData("渋谷",   x=22, y=42, width=14, height=13,   # 若者の街
                 traffic_level="moderate",  noise_status="near_limit", crowd_level="crowded", energy_status="normal",   disaster_level=None),
    DistrictData("港",     x=37, y=42, width=13, height=13,   # 六本木・芝公園
                 traffic_level="moderate",  noise_status="near_limit", crowd_level="crowded", energy_status="high",     disaster_level=None),
    DistrictData("中央",   x=51, y=42, width=13, height=13,   # 銀座・築地
                 traffic_level="congested", noise_status="near_limit", crowd_level="crowded", energy_status="high",     disaster_level=None),
    DistrictData("大田",   x=65, y=42, width=34, height=22,   # 羽田空港・最大面積
                 traffic_level="clear",     noise_status="compliant",  crowd_level="normal",  energy_status="high",     disaster_level=None),

    # ── 南部 row 5 (y=56) ─────────────────────────────────────────
    DistrictData("目黒",   x=22, y=56, width=14, height=13,   # 目黒川
                 traffic_level="clear",     noise_status="compliant",  crowd_level="sparse",  energy_status="normal",   disaster_level=None),
    DistrictData("品川",   x=37, y=56, width=27, height=13,   # 品川駅・リニア予定地
                 traffic_level="moderate",  noise_status="compliant",  crowd_level="normal",  energy_status="normal",   disaster_level=None),
]
