"""
serve/map_router.py — Sprint 71: 主要都市地図ビジュアライザ API

GET  /v1/map/cities                         : 対応都市一覧 (13都市)
GET  /v1/map/{city}/lines                   : 都市の路線一覧
GET  /v1/map/{city}/{line}/cross-section    : 地下断面データ + SVG (?format=svg|json|png)
GET  /v1/map/{city}/{line}/front-view       : 路線正面図 SVG
POST /v1/map/cross-section                  : body 指定で断面生成 (gtfs_url 指定可・フォールバック付き)
GET  /map                                   : ブラウザUI (都市・路線セレクタ + SVG プレビュー)

city_map (候補A) + map_renderer (候補B) を統合する薄い HTTP 層。
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from open_mythos.skills.city_map import (
    CityMapFactory,
    CrossSectionBuilder,
    CityMapStore,
)
from open_mythos.skills.map_renderer import MapRenderer

router = APIRouter()

# ── モジュール singleton ──────────────────────────────────────
_sample_source = CityMapFactory.from_sample()
_builder = CrossSectionBuilder()
_renderer = MapRenderer()
_store = CityMapStore()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# リクエスト / レスポンス モデル
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CityMeta(BaseModel):
    city_id:    str
    name:       str
    name_en:    str
    line_count: int

class CitiesResponse(BaseModel):
    count:  int
    cities: list[dict]

class LinesResponse(BaseModel):
    city:  str
    count: int
    lines: list[dict]

class CrossSectionResponse(BaseModel):
    cross_section: dict
    svg:           Optional[str] = None
    png_available: bool = False

class CrossSectionRequest(BaseModel):
    city:     str           = Field(..., description="都市ID (例: tokyo)")
    line:     str           = Field(..., description="路線ID (例: marunouchi)")
    source:   str           = Field("sample", description="データ源: sample | gtfs")
    gtfs_url: Optional[str] = Field(None, description="GTFS zip URL (source=gtfs 時)")
    format:   str           = Field("svg", description="出力: svg | json")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ヘルパ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resolve_source(source: str, city: str, gtfs_url: Optional[str]):
    if source == "gtfs":
        return CityMapFactory.from_gtfs(city, gtfs_url)
    return _sample_source


def _build_cross_section(city: str, line: str, source):
    try:
        cs = _builder.build(city, line, source)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="未知の都市または路線です: city=" + city + ", line=" + line,
        )
    _store.save(cs)
    return cs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@router.get("/v1/map/cities", response_model=CitiesResponse, tags=["map"])
def list_cities():
    """対応都市 (政令指定都市＋首都圏主要市・13都市) の一覧。"""
    cities = CityMapFactory.available_cities()
    return CitiesResponse(count=len(cities), cities=cities)


@router.get("/v1/map/{city}/lines", response_model=LinesResponse, tags=["map"])
def list_lines(city: str):
    """指定都市の路線一覧。"""
    lines = _sample_source.load_lines(city)
    if not lines and _sample_source.load_city(city) is None:
        raise HTTPException(status_code=404, detail="未知の都市です: " + city)
    return LinesResponse(city=city, count=len(lines), lines=[ln.to_dict() for ln in lines])


@router.get(
    "/v1/map/{city}/{line}/cross-section",
    response_model=CrossSectionResponse,
    tags=["map"],
)
def get_cross_section(
    city: str,
    line: str,
    format: str = Query("svg", description="svg | json | png"),
    source: str = Query("sample", description="sample | gtfs"),
    gtfs_url: Optional[str] = Query(None, description="GTFS zip URL"),
):
    """路線の地下断面。format=png は cairosvg が在る場合のみ PNG を返す。"""
    src = _resolve_source(source, city, gtfs_url)
    cs = _build_cross_section(city, line, src)

    if format == "json":
        return CrossSectionResponse(cross_section=cs.to_dict(), svg=None, png_available=False)

    svg = _renderer.cross_section_svg(cs)

    if format == "png":
        png = _renderer.to_png(svg)
        if png is None:
            # cairosvg 不在 → SVG にフォールバック
            return Response(content=svg, media_type="image/svg+xml")
        return Response(content=png, media_type="image/png")

    # format == "svg" (デフォルト): データ + SVG 文字列を JSON で返す
    return CrossSectionResponse(
        cross_section=cs.to_dict(),
        svg=svg,
        png_available=_renderer.to_png(svg) is not None,
    )


@router.get("/v1/map/{city}/{line}/front-view", tags=["map"])
def get_front_view(city: str, line: str):
    """路線を正面から見た駅並び SVG (image/svg+xml)。"""
    lines = _sample_source.load_lines(city)
    target = next((ln for ln in lines if ln.line_id == line), None)
    if target is None:
        raise HTTPException(
            status_code=404,
            detail="未知の都市または路線です: city=" + city + ", line=" + line,
        )
    svg = _renderer.front_view_svg(target)
    return Response(content=svg, media_type="image/svg+xml")


@router.post("/v1/map/cross-section", response_model=CrossSectionResponse, tags=["map"])
def post_cross_section(req: CrossSectionRequest):
    """
    body 指定で地下断面を生成する。source=gtfs かつ gtfs_url 指定時は GTFS から取得、
    失敗時はサンプルへフォールバック (cross_section.generated_by で判別可)。
    """
    src = _resolve_source(req.source, req.city, req.gtfs_url)
    cs = _build_cross_section(req.city, req.line, src)

    if req.format == "json":
        return CrossSectionResponse(cross_section=cs.to_dict(), svg=None, png_available=False)

    svg = _renderer.cross_section_svg(cs)
    return CrossSectionResponse(
        cross_section=cs.to_dict(),
        svg=svg,
        png_available=_renderer.to_png(svg) is not None,
    )


@router.get("/map", response_class=HTMLResponse, include_in_schema=False)
def map_ui():
    """ブラウザUI: 都市・路線を選んで地下断面 SVG をプレビュー。"""
    return HTMLResponse(content=r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>主要都市 地下断面ビジュアライザ</title>
<style>
  body{font-family:Inter,'Hiragino Sans',sans-serif;margin:0;background:#F9FAFB;color:#111827}
  header{background:#059669;color:#fff;padding:16px 24px;font-size:18px;font-weight:700}
  .ctrl{padding:16px 24px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
  select{padding:8px 12px;border:1px solid #D1D5DB;border-radius:8px;font-size:14px}
  #view{padding:0 24px 24px;overflow:auto}
  #view svg{background:#fff;border:1px solid #E5E7EB;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
</style></head><body>
<header>主要都市 地下断面ビジュアライザ <span style="font-weight:400;font-size:13px">Sprint 71</span></header>
<div class="ctrl">
  <label>都市 <select id="city"></select></label>
  <label>路線 <select id="line"></select></label>
</div>
<div id="view">読み込み中…</div>
<script>
async function j(u){const r=await fetch(u);return r.json();}
async function loadCities(){
  const d=await j('/v1/map/cities');
  const sel=document.getElementById('city');
  sel.innerHTML=d.cities.map(c=>`<option value="${c.city_id}">${c.name}</option>`).join('');
  await loadLines();
}
async function loadLines(){
  const city=document.getElementById('city').value;
  const d=await j('/v1/map/'+city+'/lines');
  const sel=document.getElementById('line');
  sel.innerHTML=d.lines.map(l=>`<option value="${l.line_id}">${l.name}</option>`).join('');
  await draw();
}
async function draw(){
  const city=document.getElementById('city').value;
  const line=document.getElementById('line').value;
  if(!city||!line)return;
  const d=await j('/v1/map/'+city+'/'+line+'/cross-section?format=svg');
  document.getElementById('view').innerHTML=d.svg||'(no data)';
}
document.getElementById('city').addEventListener('change',loadLines);
document.getElementById('line').addEventListener('change',draw);
loadCities();
</script></body></html>""")
