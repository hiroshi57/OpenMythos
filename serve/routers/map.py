"""
serve/routers/map.py — 都市地図ドメイン API (Sprint 71〜77)

主要都市地図 / 断面図 / 比較・編集・レポート / アニメーション / 経路探索 /
インポート / 混雑・アクセシビリティ・地下水位 / 環境センサー・乗換・インフラ /
交通量・エネルギー・群衆予測 / 災害・水質・騒音 の全エンドポイント。
serve/api.py のモノリスから分割 (認証は app 全体の verify_api_key に委譲)。
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 71C — 主要都市地図 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.city_map import (  # noqa: E402
    CityName as _CityName,
    CityMapDataset as _CityMapDataset,
)
from open_mythos.skills.map_renderer import (  # noqa: E402
    CrossSectionEngine as _CrossSectionEngine,
)

_city_map_store = _CityMapDataset.build()
_cross_section_engine = _CrossSectionEngine(_city_map_store)


@router.get(
    "/v1/map/cities",
    tags=["map"],
    summary="利用可能な都市一覧 — Sprint 71C",
)
def map_cities():
    cities = _city_map_store.cities()
    return {
        "cities": cities,
        "count": len(cities),
    }


@router.get(
    "/v1/map/{city}/lines",
    tags=["map"],
    summary="都市の路線一覧 — Sprint 71C",
)
def map_city_lines(city: str):
    try:
        city_enum = _CityName(city)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    lines = _city_map_store.lines.list_by_city(city_enum)
    return {
        "city": city,
        "lines": [ln.to_dict() for ln in lines],
        "count": len(lines),
    }


@router.get(
    "/v1/map/{city}/stations",
    tags=["map"],
    summary="都市の全駅一覧 — Sprint 71C",
)
def map_city_stations(city: str):
    try:
        city_enum = _CityName(city)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    stations = _city_map_store.stations.list_by_city(city_enum)
    return {
        "city": city,
        "stations": [st.to_dict() for st in stations],
        "count": len(stations),
    }


@router.get(
    "/v1/map/{city}/geology",
    tags=["map"],
    summary="都市の地質層データ — Sprint 71C",
)
def map_city_geology(city: str):
    try:
        city_enum = _CityName(city)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    layers = _city_map_store.geology.list_by_city(city_enum)
    return {
        "city": city,
        "geology_layers": [gl.to_dict() for gl in layers],
        "count": len(layers),
    }


@router.get(
    "/v1/map/{city}/geojson",
    tags=["map"],
    summary="都市の GeoJSON — Sprint 71C",
)
def map_city_geojson(city: str):
    try:
        city_enum = _CityName(city)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    data = _city_map_store.get_city_data(city_enum)
    return data.to_geojson()


@router.get(
    "/v1/map/{city}/{line_id}/cross-section",
    tags=["map"],
    summary="路線断面図 SVG 生成 — Sprint 71C",
)
def map_cross_section(city: str, line_id: str, title: str = ""):
    try:
        city_enum = _CityName(city)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    result = _cross_section_engine.generate(
        city_enum, line_id, title or None
    )
    if result is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Line not found: {line_id} in {city}",
        )
    return result.to_dict()


@router.get(
    "/v1/map/{city}/summary",
    tags=["map"],
    summary="都市データサマリー — Sprint 71C",
)
def map_city_summary(city: str):
    try:
        city_enum = _CityName(city)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    data = _city_map_store.get_city_data(city_enum)
    max_depth = max((st.depth_m for st in data.stations), default=0.0)
    avg_depth = (
        sum(st.depth_m for st in data.stations) / len(data.stations)
        if data.stations else 0.0
    )
    return {
        "city": city,
        "lines": len(data.lines),
        "stations": len(data.stations),
        "geology_layers": len(data.geology_layers),
        "max_station_depth_m": max_depth,
        "avg_station_depth_m": round(avg_depth, 1),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 72A — 都市間断面比較
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.map_comparator import (  # noqa: E402
    MapComparator as _MapComparator,
)

_map_comparator = _MapComparator(_city_map_store)


@router.get(
    "/v1/map/compare/{city_a}/{city_b}",
    tags=["map"],
    summary="2都市断面比較 SVG — Sprint 72A",
)
def map_compare(city_a: str, city_b: str):
    from fastapi import HTTPException
    try:
        ca = _CityName(city_a)
        cb = _CityName(city_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"City not found: {e}")
    result = _map_comparator.compare(ca, cb)
    return result.to_dict()


@router.get(
    "/v1/map/compare/{city_a}/{city_b}/stats",
    tags=["map"],
    summary="2都市深度統計比較 — Sprint 72A",
)
def map_compare_stats(city_a: str, city_b: str):
    from fastapi import HTTPException
    try:
        ca = _CityName(city_a)
        cb = _CityName(city_b)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"City not found: {e}")
    result = _map_comparator.compare(ca, cb)
    return {
        "city_a": result.stats_a.to_dict(),
        "city_b": result.stats_b.to_dict(),
        "deeper_city": result.deeper_city,
        "geology_diff": result.geology_diff,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 72B — 路線データ編集 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.map_editor import (  # noqa: E402
    MapEditor as _MapEditor,
)
from open_mythos.skills.city_map import (  # noqa: E402
    Station as _Station, MetroLine as _MetroLine,
    GeoCoord as _GeoCoord,
    LineType as _LineType,
)

_map_editor = _MapEditor(_city_map_store)


class _AddLineReq(BaseModel):
    id: str
    name: str
    name_en: str
    city: str
    line_type: str = "subway"
    color: str = "#999999"
    total_length_km: float = 0.0
    opened_year: int = None


class _UpdateLineReq(BaseModel):
    name: str = None
    name_en: str = None
    color: str = None
    total_length_km: float = None
    opened_year: int = None


class _AddStationReq(BaseModel):
    id: str
    name: str
    name_en: str
    line_id: str
    city: str
    lat: float
    lon: float
    depth_m: float
    platform_count: int = 2
    opened_year: int = None


class _UpdateStationReq(BaseModel):
    name: str = None
    name_en: str = None
    depth_m: float = None
    platform_count: int = None
    opened_year: int = None


class _AddGeologyReq(BaseModel):
    id: str
    city: str
    layer_type: str
    name: str
    depth_from_m: float
    depth_to_m: float
    color: str = "#CCCCCC"
    n_value: float = None


class _UpdateGeologyReq(BaseModel):
    name: str = None
    depth_from_m: float = None
    depth_to_m: float = None
    color: str = None
    n_value: float = None


@router.post(
    "/v1/map-editor/lines",
    tags=["map-editor"],
    summary="路線追加 — Sprint 72B",
)
def editor_add_line(req: _AddLineReq):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(req.city)
        line_type_enum = _LineType(req.line_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    line = _MetroLine(
        id=req.id, name=req.name, name_en=req.name_en,
        city=city_enum, line_type=line_type_enum,
        color=req.color, total_length_km=req.total_length_km,
        opened_year=req.opened_year,
    )
    result = _map_editor.add_line(line)
    return result.to_dict()


@router.patch(
    "/v1/map-editor/lines/{line_id}",
    tags=["map-editor"],
    summary="路線更新 — Sprint 72B",
)
def editor_update_line(line_id: str, req: _UpdateLineReq):
    kwargs = {k: v for k, v in req.model_dump().items() if v is not None}
    result = _map_editor.update_line(line_id, **kwargs)
    return result.to_dict()


@router.delete(
    "/v1/map-editor/lines/{line_id}",
    tags=["map-editor"],
    summary="路線削除 — Sprint 72B",
)
def editor_delete_line(line_id: str):
    result = _map_editor.delete_line(line_id)
    return result.to_dict()


@router.post(
    "/v1/map-editor/stations",
    tags=["map-editor"],
    summary="駅追加 — Sprint 72B",
)
def editor_add_station(req: _AddStationReq):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(req.city)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    station = _Station(
        id=req.id, name=req.name, name_en=req.name_en,
        line_id=req.line_id, city=city_enum,
        coord=_GeoCoord(req.lat, req.lon),
        depth_m=req.depth_m, platform_count=req.platform_count,
        opened_year=req.opened_year,
    )
    result = _map_editor.add_station(station)
    return result.to_dict()


@router.patch(
    "/v1/map-editor/stations/{station_id}",
    tags=["map-editor"],
    summary="駅更新 — Sprint 72B",
)
def editor_update_station(station_id: str, req: _UpdateStationReq):
    kwargs = {k: v for k, v in req.model_dump().items() if v is not None}
    result = _map_editor.update_station(station_id, **kwargs)
    return result.to_dict()


@router.delete(
    "/v1/map-editor/stations/{station_id}",
    tags=["map-editor"],
    summary="駅削除 — Sprint 72B",
)
def editor_delete_station(station_id: str):
    result = _map_editor.delete_station(station_id)
    return result.to_dict()


@router.get(
    "/v1/map-editor/history",
    tags=["map-editor"],
    summary="編集履歴 — Sprint 72B",
)
def editor_history():
    return {
        "history": _map_editor.history_dicts(),
        "count": len(_map_editor.history),
    }


@router.get(
    "/v1/map-editor/summary",
    tags=["map-editor"],
    summary="エディタ状態サマリー — Sprint 72B",
)
def editor_summary():
    return _map_editor.summary()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 72C — 地図レポート生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.map_report import (  # noqa: E402
    MapReportEngine as _MapReportEngine,
)

_map_report_engine = _MapReportEngine(_city_map_store)


@router.get(
    "/v1/map/{city}/report/md",
    tags=["map-report"],
    summary="都市地図レポート (Markdown) — Sprint 72C",
)
def map_city_report_md(city: str):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(city)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    report = _map_report_engine.generate_city_report(city_enum)
    return report.to_dict()


@router.get(
    "/v1/map/report/compare",
    tags=["map-report"],
    summary="複数都市比較レポート — Sprint 72C",
)
def map_compare_report(cities: str = "tokyo,osaka,nagoya,yokohama,fukuoka"):
    from fastapi import HTTPException
    city_list = []
    for c in cities.split(","):
        c = c.strip()
        try:
            city_list.append(_CityName(c))
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Unknown city: {c}")
    if not city_list:
        raise HTTPException(status_code=422, detail="cities is empty")
    report = _map_report_engine.generate_multi_city_report(city_list)
    return report.to_dict()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 73A — 地図アニメーション
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.map_animator import (  # noqa: E402
    MapAnimator as _MapAnimator,
    SurveyDataset as _SurveyDataset,
)

_map_animator = _MapAnimator()


@router.get(
    "/v1/map/{city}/animate",
    tags=["map-animate"],
    summary="時系列地質断面 SVG アニメーション — Sprint 73A",
)
def map_animate(city: str, title: str = ""):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(city)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    snapshots = _SurveyDataset.build(city_enum)
    result = _map_animator.animate(snapshots, title or None)
    return result.to_dict()


@router.get(
    "/v1/map/{city}/animate/snapshots",
    tags=["map-animate"],
    summary="時系列スナップショット一覧 — Sprint 73A",
)
def map_animate_snapshots(city: str):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(city)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    snapshots = _SurveyDataset.build(city_enum)
    return {
        "city": city,
        "snapshots": [s.to_dict() for s in snapshots],
        "count": len(snapshots),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 73B — 経路探索 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.route_finder import (  # noqa: E402
    RouteGraphBuilder as _RouteGraphBuilder,
    RouteFinder as _RouteFinder,
)

_route_graph = _RouteGraphBuilder.build(_city_map_store)
_route_finder = _RouteFinder(_route_graph)


@router.get(
    "/v1/map/route/{from_id}/{to_id}",
    tags=["map-route"],
    summary="最短経路探索 — Sprint 73B",
)
def map_route(from_id: str, to_id: str):
    result = _route_finder.find(from_id, to_id)
    return result.to_dict()


@router.get(
    "/v1/map-route/graph/stats",
    tags=["map-route"],
    summary="路線グラフ統計 — Sprint 73B",
)
def map_route_graph_stats():
    return {
        "station_count": len(_route_graph.station_ids()),
        "edge_count": _route_graph.edge_count(),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 73C — データインポート API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.map_importer import (  # noqa: E402
    MapImporter as _MapImporter,
)

_map_importer = _MapImporter(_city_map_store)


class _CSVImportReq(BaseModel):
    csv_text: str


class _GeoJSONImportReq(BaseModel):
    geojson_text: str


@router.post(
    "/v1/map/import/stations/csv",
    tags=["map-import"],
    summary="駅データ CSV インポート — Sprint 73C",
)
def map_import_stations_csv(req: _CSVImportReq):
    result = _map_importer.import_stations_csv(req.csv_text)
    return result.to_dict()


@router.post(
    "/v1/map/import/lines/csv",
    tags=["map-import"],
    summary="路線データ CSV インポート — Sprint 73C",
)
def map_import_lines_csv(req: _CSVImportReq):
    result = _map_importer.import_lines_csv(req.csv_text)
    return result.to_dict()


@router.post(
    "/v1/map/import/geology/csv",
    tags=["map-import"],
    summary="地質層 CSV インポート — Sprint 73C",
)
def map_import_geology_csv(req: _CSVImportReq):
    result = _map_importer.import_geology_csv(req.csv_text)
    return result.to_dict()


@router.post(
    "/v1/map/import/geojson",
    tags=["map-import"],
    summary="GeoJSON インポート (駅) — Sprint 73C",
)
def map_import_geojson(req: _GeoJSONImportReq):
    result = _map_importer.import_geojson(req.geojson_text)
    return result.to_dict()


@router.get(
    "/v1/map-import/summary",
    tags=["map-import"],
    summary="インポート後ストアサマリー — Sprint 73C",
)
def map_import_summary():
    return _map_importer.summary()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 74A — 駅混雑シミュレーション API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.crowd_simulator import (  # noqa: E402
    CrowdDataset as _CrowdDataset,
)

_crowd_sim = _CrowdDataset.build()


@router.get(
    "/v1/crowd/{station_id}/snapshot",
    tags=["crowd"],
    summary="時点混雑スナップショット — Sprint 74A",
)
def crowd_snapshot(station_id: str, hour: int = 8):
    from fastapi import HTTPException
    result = _crowd_sim.snapshot(station_id, hour)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Station not found: {station_id}")
    return result.to_dict()


@router.get(
    "/v1/crowd/{station_id}/daily",
    tags=["crowd"],
    summary="1日の混雑プロファイル — Sprint 74A",
)
def crowd_daily(station_id: str):
    from fastapi import HTTPException
    result = _crowd_sim.daily_profile(station_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Station not found: {station_id}")
    return result.to_dict()


@router.get(
    "/v1/crowd/compare",
    tags=["crowd"],
    summary="複数駅の混雑比較 — Sprint 74A",
)
def crowd_compare(stations: str = "tokyo-shinjuku,tokyo-ginza", hour: int = 8):
    ids = [s.strip() for s in stations.split(",") if s.strip()]
    results = _crowd_sim.compare(ids, hour)
    return {
        "hour": hour,
        "stations": [r.to_dict() for r in results],
        "count": len(results),
    }


@router.get(
    "/v1/crowd/stations",
    tags=["crowd"],
    summary="混雑データ登録駅一覧 — Sprint 74A",
)
def crowd_stations():
    ids = _crowd_sim.all_station_ids()
    return {"station_ids": ids, "count": len(ids)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 74B — アクセシビリティ分析 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.accessibility import (  # noqa: E402
    AccessibilityDataset as _AccessDataset,
)

_access_analyzer = _AccessDataset.build()


@router.get(
    "/v1/access/{station_id}/score",
    tags=["accessibility"],
    summary="駅アクセシビリティスコア — Sprint 74B",
)
def access_score(station_id: str):
    from fastapi import HTTPException
    result = _access_analyzer.score(station_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Station not found: {station_id}")
    return result.to_dict()


@router.get(
    "/v1/access/rank",
    tags=["accessibility"],
    summary="アクセシビリティランキング — Sprint 74B",
)
def access_rank(city: str = ""):
    if city:
        try:
            city_enum = _CityName(city)
        except ValueError:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail=f"Unknown city: {city}")
        scores = _access_analyzer.rank(city_enum)
    else:
        scores = _access_analyzer.rank()
    return {"ranking": [s.to_dict() for s in scores], "count": len(scores)}


@router.get(
    "/v1/access/{city}/report",
    tags=["accessibility"],
    summary="都市アクセシビリティレポート — Sprint 74B",
)
def access_city_report(city: str):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(city)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    report = _access_analyzer.city_report(city_enum)
    return report.to_dict()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sprint 74C — 地下水位モニタリング API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from open_mythos.skills.groundwater import (  # noqa: E402
    GroundwaterDataset as _GWDataset,
    Season as _Season,
)

_gw_assessor = _GWDataset.build()


@router.get(
    "/v1/groundwater/{city}",
    tags=["groundwater"],
    summary="都市の地下水プロファイル — Sprint 74C",
)
def groundwater_profile(city: str):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(city)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    profile = _gw_assessor.get_profile(city_enum)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Groundwater data not found: {city}")
    return profile.to_dict()


@router.get(
    "/v1/groundwater/{city}/risk",
    tags=["groundwater"],
    summary="都市レベル浸水リスク評価 — Sprint 74C",
)
def groundwater_city_risk(city: str, season: str = "summer"):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(city)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    try:
        season_enum = _Season(season)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown season: {season}")
    result = _gw_assessor.city_risk(city_enum, season_enum)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Data not found: {city}")
    return result.to_dict()


@router.get(
    "/v1/groundwater/{city}/{station_id}/flood-risk",
    tags=["groundwater"],
    summary="駅別浸水リスク評価 — Sprint 74C",
)
def groundwater_station_risk(city: str, station_id: str, season: str = "summer"):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(city)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    try:
        season_enum = _Season(season)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown season: {season}")
    # 駅の深度を CityMapStore から取得
    station = _city_map_store.stations.get(station_id)
    depth = station.depth_m if station else 15.0
    result = _gw_assessor.station_risk(city_enum, station_id, depth, season_enum)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Data not found: {city}")
    return result.to_dict()


# ─────────────────────────────────────────────────────────────────
# Sprint 75A — 駅環境センサー
# ─────────────────────────────────────────────────────────────────

from open_mythos.skills.env_sensor import (
    EnvSensorDataset as _EnvSensorDataset,
    SensorStatus as _SensorStatus,
)

_env_analyzer = _EnvSensorDataset.build()


@router.get(
    "/v1/env/{station_id}/snapshot",
    tags=["env_sensor"],
    summary="駅環境スナップショット — Sprint 75A",
)
def env_snapshot(station_id: str):
    from fastapi import HTTPException
    env = _env_analyzer.snapshot(station_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Station not found: {station_id}")
    return env.to_dict()


@router.get(
    "/v1/env/compare",
    tags=["env_sensor"],
    summary="複数駅環境比較 — Sprint 75A",
)
def env_compare(stations: str):
    """stations: カンマ区切りの station_id リスト"""
    ids = [s.strip() for s in stations.split(",") if s.strip()]
    result = _env_analyzer.compare(ids)
    return result.to_dict()


@router.get(
    "/v1/env/{city}/alerts",
    tags=["env_sensor"],
    summary="都市内アラート駅一覧 — Sprint 75A",
)
def env_alerts(city: str, min_status: str = "warning"):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(city)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    try:
        status_enum = _SensorStatus(min_status)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Unknown status: {min_status}")
    alerts = _env_analyzer.alert_stations(city=city_enum, min_status=status_enum)
    return {"city": city, "min_status": min_status, "alerts": [e.to_dict() for e in alerts]}


# ─────────────────────────────────────────────────────────────────
# Sprint 75B — 乗り換え最適化
# ─────────────────────────────────────────────────────────────────

from open_mythos.skills.transfer_optimizer import (
    OptimizationDataset as _OptDataset,
    OptimizationWeight as _OptWeight,
)

_opt_dataset = _OptDataset.build()
_transfer_optimizer = _opt_dataset.optimizer()


@router.get(
    "/v1/transfer/optimize",
    tags=["transfer"],
    summary="乗り換え最適化 (全プリセット) — Sprint 75B",
)
def transfer_optimize(from_id: str, to_id: str, hour: int = 8):
    options = _transfer_optimizer.optimize(from_id, to_id, hour)
    return {
        "from_id": from_id,
        "to_id": to_id,
        "hour": hour,
        "options": [o.to_dict() for o in options],
    }


@router.get(
    "/v1/transfer/score",
    tags=["transfer"],
    summary="乗り換えスコア (カスタム重み) — Sprint 75B",
)
def transfer_score(
    from_id: str,
    to_id: str,
    hour: int = 8,
    crowd_w: float = 0.4,
    access_w: float = 0.3,
    time_w: float = 0.3,
):
    from fastapi import HTTPException
    try:
        weight = _OptWeight(crowd_w=crowd_w, access_w=access_w, time_w=time_w, label="custom")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    result = _transfer_optimizer.score_route(from_id, to_id, hour, weight)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Route not found: {from_id} -> {to_id}")
    return result.to_dict()


# ─────────────────────────────────────────────────────────────────
# Sprint 75C — 都市インフラダッシュボード
# ─────────────────────────────────────────────────────────────────

from open_mythos.skills.infra_dashboard import (
    DashboardDataset as _DashDataset,
)

_dash_dataset = _DashDataset.build()
_infra_dashboard = _dash_dataset.dashboard()


@router.get(
    "/v1/infra/summary",
    tags=["infra_dashboard"],
    summary="複数都市インフラサマリー — Sprint 75C",
)
def infra_summary(cities: str = "tokyo,osaka", hour: int = 8):
    city_names = []
    for c in cities.split(","):
        c = c.strip()
        try:
            city_names.append(_CityName(c))
        except ValueError:
            pass
    result = _infra_dashboard.multi_city_summary(city_names, hour)
    return {"hour": hour, "cities": result}


@router.get(
    "/v1/infra/{city}/alerts",
    tags=["infra_dashboard"],
    summary="都市別インフラアラート駅 — Sprint 75C",
)
def infra_alerts(city: str, hour: int = 8):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(city)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    panels = _infra_dashboard.alert_stations(city_enum, hour)
    return {
        "city": city,
        "hour": hour,
        "alert_count": len(panels),
        "panels": [p.to_dict() for p in panels],
    }


@router.get(
    "/v1/infra/{city}",
    tags=["infra_dashboard"],
    summary="都市インフラダッシュボード — Sprint 75C",
)
def infra_city(city: str, hour: int = 8):
    from fastapi import HTTPException
    try:
        city_enum = _CityName(city)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"City not found: {city}")
    db = _infra_dashboard.city_panel(city_enum, hour)
    return db.to_dict()


# ---------------------------------------------------------------------------
# Sprint 76 — 交通量分析 / エネルギーモニタリング / 群衆予測
# ---------------------------------------------------------------------------

from open_mythos.skills.traffic_analyzer import (
    TrafficAnalyzer as _TrafficAnalyzer,
    TrafficStore as _TrafficStore,
)
from open_mythos.skills.energy_monitor import (
    EnergyMonitor as _EnergyMonitor,
    EnergyStore as _EnergyStore,
    EnergyType as _EnergyType,
)
from open_mythos.skills.crowd_predictor import (
    CrowdPredictor as _CrowdPredictor,
    CrowdStore as _CrowdStore,
    WeatherCondition as _WeatherCondition,
    EventType as _EventType,
)

_traffic_store = _TrafficStore()
_traffic_analyzer = _TrafficAnalyzer(store=_traffic_store)

_energy_store = _EnergyStore()
_energy_monitor = _EnergyMonitor(store=_energy_store)

_crowd_store = _CrowdStore()
_crowd_predictor = _CrowdPredictor(store=_crowd_store)


# ── 交通量 ────────────────────────────────────────────────────────

class _TrafficSegmentIn(BaseModel):
    segment_id: str
    road_name: str
    city: str
    volume: int = Field(ge=0)
    speed_kmh: float = Field(ge=0.0, le=200.0)
    density: float = Field(ge=0.0)
    length_km: float = Field(default=1.0, ge=0.0)
    hour: int = Field(default=8, ge=0, le=23)


@router.post("/v1/traffic/segments", tags=["traffic"], summary="交通セグメント登録 — Sprint 76A")
def traffic_add_segment(body: _TrafficSegmentIn):
    seg = _traffic_analyzer.add_segment(
        segment_id=body.segment_id,
        road_name=body.road_name,
        city=body.city,
        volume=body.volume,
        speed_kmh=body.speed_kmh,
        density=body.density,
        length_km=body.length_km,
        hour=body.hour,
    )
    return seg.to_dict()


@router.get("/v1/traffic/segments", tags=["traffic"], summary="全セグメント一覧 — Sprint 76A")
def traffic_list_segments(city: Optional[str] = None):
    segs = _traffic_store.list_by_city(city) if city else _traffic_store.list_all()
    return {"segments": [s.to_dict() for s in segs], "count": len(segs)}


@router.get("/v1/traffic/hotspots/{city}", tags=["traffic"], summary="渋滞ホットスポット — Sprint 76A")
def traffic_hotspots(city: str, top_n: int = 5):
    hotspots = _traffic_analyzer.get_hotspots(city, top_n=top_n)
    return {"city": city, "hotspots": [h.to_dict() for h in hotspots]}


@router.get("/v1/traffic/forecast", tags=["traffic"], summary="24時間交通量予測 — Sprint 76A")
def traffic_forecast(base_speed_kmh: float = 50.0):
    forecasts = _traffic_analyzer.predict_by_hour(base_speed_kmh=base_speed_kmh)
    return {"forecasts": [f.to_dict() for f in forecasts]}


@router.get("/v1/traffic/summary/{city}", tags=["traffic"], summary="都市交通サマリー — Sprint 76A")
def traffic_city_summary(city: str):
    return _traffic_analyzer.city_summary(city)


# ── エネルギーモニタリング ────────────────────────────────────────

class _EnergyReadingIn(BaseModel):
    reading_id: str
    facility_id: str
    city: str
    energy_type: str
    value: float = Field(ge=0.0)
    hour: int = Field(default=12, ge=0, le=23)
    day: int = Field(default=1, ge=1, le=31)


@router.post("/v1/energy/readings", tags=["energy"], summary="エネルギー計測値登録 — Sprint 76B")
def energy_add_reading(body: _EnergyReadingIn):
    try:
        etype = _EnergyType(body.energy_type)
    except ValueError:
        raise HTTPException(400, f"Invalid energy_type: {body.energy_type}")
    reading = _energy_monitor.add_reading(
        reading_id=body.reading_id,
        facility_id=body.facility_id,
        city=body.city,
        energy_type=etype,
        value=body.value,
        hour=body.hour,
        day=body.day,
    )
    return reading.to_dict()


@router.get("/v1/energy/summary/{city}", tags=["energy"], summary="都市エネルギーサマリー — Sprint 76B")
def energy_city_summary(city: str, energy_type: str = "electricity"):
    try:
        etype = _EnergyType(energy_type)
    except ValueError:
        raise HTTPException(400, f"Invalid energy_type: {energy_type}")
    summary = _energy_monitor.summarize_city(city, etype)
    if summary is None:
        raise HTTPException(404, f"No readings for city={city} type={energy_type}")
    return summary.to_dict()


@router.get("/v1/energy/anomalies/{city}", tags=["energy"], summary="エネルギー異常検出 — Sprint 76B")
def energy_anomalies(city: str, energy_type: str = "electricity"):
    try:
        etype = _EnergyType(energy_type)
    except ValueError:
        raise HTTPException(400, f"Invalid energy_type: {energy_type}")
    anomalies = _energy_monitor.detect_anomalies(city, etype)
    return {"city": city, "energy_type": energy_type, "anomalies": [a.to_dict() for a in anomalies]}


@router.get("/v1/energy/profile/{city}", tags=["energy"], summary="時間帯別消費プロファイル — Sprint 76B")
def energy_hourly_profile(city: str, energy_type: str = "electricity"):
    try:
        etype = _EnergyType(energy_type)
    except ValueError:
        raise HTTPException(400, f"Invalid energy_type: {energy_type}")
    return {"city": city, "energy_type": energy_type, "profile": _energy_monitor.hourly_profile(city, etype)}


# ── 群衆予測 ──────────────────────────────────────────────────────

class _CrowdSnapshotIn(BaseModel):
    snapshot_id: str
    spot_name: str
    city: str
    count: int = Field(ge=0)
    hour: int = Field(default=12, ge=0, le=23)
    weather: str = "sunny"
    event: str = "none"


@router.post("/v1/crowd/snapshots", tags=["crowd"], summary="人流スナップショット登録 — Sprint 76C")
def crowd_add_snapshot(body: _CrowdSnapshotIn):
    try:
        weather = _WeatherCondition(body.weather)
        event = _EventType(body.event)
    except ValueError as e:
        raise HTTPException(400, str(e))
    snap = _crowd_predictor.add_snapshot(
        snapshot_id=body.snapshot_id,
        spot_name=body.spot_name,
        city=body.city,
        count=body.count,
        hour=body.hour,
        weather=weather,
        event=event,
    )
    return snap.to_dict()


@router.get("/v1/crowd/predict", tags=["crowd"], summary="群衆予測 — Sprint 76C")
def crowd_predict(
    spot_name: str,
    city: str,
    hour: int = 12,
    weather: str = "sunny",
    event: str = "none",
):
    try:
        weather_e = _WeatherCondition(weather)
        event_e = _EventType(event)
    except ValueError as e:
        raise HTTPException(400, str(e))
    result = _crowd_predictor.predict(
        spot_name=spot_name, city=city, hour=hour, weather=weather_e, event=event_e
    )
    return result.to_dict()


@router.get("/v1/crowd/heatmap/{city}", tags=["crowd"], summary="群衆ヒートマップ — Sprint 76C")
def crowd_heatmap(city: str):
    cells = _crowd_predictor.heatmap(city)
    return {"city": city, "heatmap": [c.to_dict() for c in cells]}


@router.get("/v1/crowd/forecast", tags=["crowd"], summary="24時間群衆予測 — Sprint 76C")
def crowd_daily_forecast(
    spot_name: str,
    city: str,
    weather: str = "sunny",
    event: str = "none",
):
    try:
        weather_e = _WeatherCondition(weather)
        event_e = _EventType(event)
    except ValueError as e:
        raise HTTPException(400, str(e))
    forecasts = _crowd_predictor.daily_forecast(
        spot_name=spot_name, city=city, weather=weather_e, event=event_e
    )
    return {"spot_name": spot_name, "city": city, "forecasts": [f.to_dict() for f in forecasts]}


# ---------------------------------------------------------------------------
# Sprint 77 — 災害アラート / 水質モニタリング / 騒音マッピング
# ---------------------------------------------------------------------------

from open_mythos.skills.disaster_alert import (
    DisasterAlertManager as _DisasterAlertManager,
    AlertStore as _DisasterAlertStore,
    DisasterType as _DisasterType,
    AlertLevel as _AlertLevel,
)
from open_mythos.skills.water_quality import (
    WaterQualityMonitor as _WQMonitor,
    WaterQualityStore as _WQStore,
    WaterParam as _WaterParam,
    SourceType as _SourceType,
)
from open_mythos.skills.noise_mapper import (
    NoiseMapper as _NoiseMapper,
    NoiseMeasurementStore as _NoiseStore,
    ZoneType as _ZoneType,
)

# Sprint 66 の異常検知用 _alert_store と衝突しないよう災害用は別名にする
_disaster_alert_store = _DisasterAlertStore()
_disaster_alert_manager = _DisasterAlertManager(store=_disaster_alert_store)

_wq_store = _WQStore()
_wq_monitor = _WQMonitor(store=_wq_store)

_noise_store = _NoiseStore()
_noise_mapper = _NoiseMapper(store=_noise_store)


# ── 災害アラート ──────────────────────────────────────────────────

class _AlertIn(BaseModel):
    alert_id: str
    disaster_type: str
    city: str
    level: Optional[str] = None
    magnitude: Optional[float] = None
    description: str = ""
    affected_areas: List[str] = []


@router.post("/v1/disaster/alerts", tags=["disaster"], summary="災害アラート発令 — Sprint 77A")
def disaster_issue(body: _AlertIn):
    try:
        dtype = _DisasterType(body.disaster_type)
        level = _AlertLevel(body.level) if body.level else None
    except ValueError as e:
        raise HTTPException(400, str(e))
    alert = _disaster_alert_manager.issue_alert(
        alert_id=body.alert_id,
        disaster_type=dtype,
        city=body.city,
        level=level,
        magnitude=body.magnitude,
        description=body.description,
        affected_areas=body.affected_areas,
    )
    return alert.to_dict()


@router.get("/v1/disaster/alerts", tags=["disaster"], summary="発令中アラート一覧 — Sprint 77A")
def disaster_active(city: Optional[str] = None):
    alerts = _disaster_alert_manager.get_active_alerts(city)
    return {"alerts": [a.to_dict() for a in alerts], "count": len(alerts)}


@router.patch("/v1/disaster/alerts/{alert_id}/resolve", tags=["disaster"], summary="アラート解除 — Sprint 77A")
def disaster_resolve(alert_id: str):
    ok = _disaster_alert_manager.resolve_alert(alert_id)
    if not ok:
        raise HTTPException(404, f"alert '{alert_id}' not found")
    return {"ok": True, "alert_id": alert_id, "status": "resolved"}


@router.get("/v1/disaster/summary/{city}", tags=["disaster"], summary="都市災害サマリー — Sprint 77A")
def disaster_city_summary(city: str):
    return _disaster_alert_manager.city_summary(city).to_dict()


# ── 水質モニタリング ──────────────────────────────────────────────

class _WaterReadingIn(BaseModel):
    reading_id: str
    station_id: str
    city: str
    source_type: str
    param: str
    value: float
    hour: int = Field(default=12, ge=0, le=23)
    day: int = Field(default=1, ge=1, le=31)


@router.post("/v1/water/readings", tags=["water"], summary="水質計測値登録 — Sprint 77B")
def water_add_reading(body: _WaterReadingIn):
    try:
        src = _SourceType(body.source_type)
        param = _WaterParam(body.param)
    except ValueError as e:
        raise HTTPException(400, str(e))
    r = _wq_monitor.add_reading(
        reading_id=body.reading_id,
        station_id=body.station_id,
        city=body.city,
        source_type=src,
        param=param,
        value=body.value,
        hour=body.hour,
        day=body.day,
    )
    return r.to_dict()


@router.get("/v1/water/unsafe/{city}", tags=["water"], summary="基準超過水質 — Sprint 77B")
def water_unsafe(city: str):
    readings = _wq_monitor.get_unsafe_readings(city)
    return {"city": city, "unsafe": [r.to_dict() for r in readings], "count": len(readings)}


@router.get("/v1/water/report/{city}", tags=["water"], summary="都市水質レポート — Sprint 77B")
def water_city_report(city: str):
    return _wq_monitor.city_report(city)


@router.get("/v1/water/station/{station_id}", tags=["water"], summary="観測所サマリー — Sprint 77B")
def water_station_summary(station_id: str, param: str = "ph"):
    try:
        p = _WaterParam(param)
    except ValueError:
        raise HTTPException(400, f"Invalid param: {param}")
    summary = _wq_monitor.station_summary(station_id, p)
    if summary is None:
        raise HTTPException(404, f"No readings for station={station_id} param={param}")
    return summary.to_dict()


# ── 騒音マッピング ────────────────────────────────────────────────

class _NoiseMeasurementIn(BaseModel):
    measurement_id: str
    location_name: str
    city: str
    zone_type: str
    db_level: float = Field(ge=0.0, le=200.0)
    hour: int = Field(default=12, ge=0, le=23)


@router.post("/v1/noise/measurements", tags=["noise"], summary="騒音計測値登録 — Sprint 77C")
def noise_add_measurement(body: _NoiseMeasurementIn):
    try:
        zone = _ZoneType(body.zone_type)
    except ValueError:
        raise HTTPException(400, f"Invalid zone_type: {body.zone_type}")
    m = _noise_mapper.add_measurement(
        measurement_id=body.measurement_id,
        location_name=body.location_name,
        city=body.city,
        zone_type=zone,
        db_level=body.db_level,
        hour=body.hour,
    )
    return m.to_dict()


@router.get("/v1/noise/violations", tags=["noise"], summary="騒音規制超過一覧 — Sprint 77C")
def noise_violations(city: Optional[str] = None):
    violations = _noise_mapper.get_violations(city)
    return {"violations": [m.to_dict() for m in violations], "count": len(violations)}


@router.get("/v1/noise/map/{city}", tags=["noise"], summary="騒音マップ — Sprint 77C")
def noise_map(city: str):
    cells = _noise_mapper.generate_map(city)
    return {"city": city, "map": [c.to_dict() for c in cells]}


@router.get("/v1/noise/report/{city}", tags=["noise"], summary="都市騒音レポート — Sprint 77C")
def noise_city_report(city: str):
    return _noise_mapper.city_report(city).to_dict()
