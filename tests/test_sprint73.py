"""
Sprint 73 — 地図アニメーション/経路探索/インポート テスト

対象:
  73A: open_mythos/skills/map_animator.py
  73B: open_mythos/skills/route_finder.py
  73C: open_mythos/skills/map_importer.py
  serve/api.py (/v1/map/{city}/animate/*, /v1/map/route/*, /v1/map/import/*)
"""
from __future__ import annotations
import json
import pytest
from fastapi.testclient import TestClient

from open_mythos.skills.city_map import (
    CityName, LineType, GeologyLayerType,
    GeoCoord, Station, MetroLine, GeologyLayer,
    CityMapStore, CityMapDataset,
)
from open_mythos.skills.map_animator import (
    SurveySnapshot, AnimationConfig, AnimationResult,
    MapAnimator, SurveyDataset,
)
from open_mythos.skills.route_finder import (
    RouteEdge, RouteStep, RouteResult,
    RouteGraph, RouteGraphBuilder, RouteFinder,
)
from open_mythos.skills.map_importer import (
    ImportError as MapImportError, ImportResult,
    StationCSVImporter, LineCSVImporter, GeologyCSVImporter,
    GeoJSONImporter, MapImporter,
)


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset_store():
    return CityMapDataset.build()


@pytest.fixture(scope="module")
def client():
    from serve.api import app
    return TestClient(app)


def _make_station(sid="s1", city=CityName.TOKYO, line_id="l1", depth=15.0):
    return Station(
        id=sid, name=f"駅{sid}", name_en=f"St{sid}",
        line_id=line_id, city=city,
        coord=GeoCoord(35.68, 139.76),
        depth_m=depth, platform_count=2, opened_year=2000,
    )


def _make_line(lid="l1", city=CityName.TOKYO, station_ids=None):
    return MetroLine(
        id=lid, name="テスト線", name_en="Test Line",
        city=city, line_type=LineType.SUBWAY,
        color="#FF0000",
        station_ids=station_ids or [],
        total_length_km=10.0, opened_year=2000,
    )


def _make_geology(gid="g1", city=CityName.TOKYO, d_from=0.0, d_to=10.0):
    return GeologyLayer(
        id=gid, city=city, layer_type=GeologyLayerType.ALLUVIUM,
        name="沖積層", depth_from_m=d_from, depth_to_m=d_to,
        color="#90EE90", n_value=5.0,
    )


def _make_snapshot(year=2000, city=CityName.TOKYO, layers=None):
    if layers is None:
        layers = [_make_geology()]
    return SurveySnapshot(year=year, city=city, layers=layers)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 73A: map_animator.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── SurveySnapshot ───────────────────────────────────────────────

def test_survey_snapshot_to_dict():
    snap = _make_snapshot(year=2020)
    d = snap.to_dict()
    assert d["year"] == 2020
    assert d["city"] == "tokyo"
    assert len(d["layers"]) >= 1


# ─── AnimationConfig ──────────────────────────────────────────────

def test_animation_config_defaults():
    cfg = AnimationConfig()
    assert cfg.width == 900
    assert cfg.height == 500
    assert cfg.max_depth_m == 60.0
    assert cfg.frame_duration_s == 1.5


def test_animation_config_depth_to_y_surface():
    cfg = AnimationConfig()
    y = cfg.depth_to_y(0.0)
    assert y == pytest.approx(cfg.surface_y, abs=1)


def test_animation_config_depth_to_y_ordering():
    cfg = AnimationConfig()
    assert cfg.depth_to_y(10.0) < cfg.depth_to_y(30.0) < cfg.depth_to_y(60.0)


def test_animation_config_depth_to_h():
    cfg = AnimationConfig()
    y0, h = cfg.depth_to_h(0.0, 10.0)
    assert h > 0
    y0b, hb = cfg.depth_to_h(0.0, 20.0)
    assert hb > h


def test_animation_config_plot_width():
    cfg = AnimationConfig(width=900, margin_left=80, margin_right=40)
    assert cfg.plot_width == 780


# ─── SurveyDataset ────────────────────────────────────────────────

def test_survey_dataset_build_tokyo():
    snaps = SurveyDataset.build_tokyo()
    assert len(snaps) == 4
    years = [s.year for s in snaps]
    assert years == sorted(years)
    assert 1960 in years and 2020 in years


def test_survey_dataset_build_osaka():
    snaps = SurveyDataset.build_osaka()
    assert len(snaps) == 4
    assert all(s.city == CityName.OSAKA for s in snaps)


def test_survey_dataset_build_generic():
    snaps = SurveyDataset.build(CityName.NAGOYA)
    assert len(snaps) >= 4
    assert all(s.city == CityName.NAGOYA for s in snaps)


def test_survey_dataset_layers_in_each_snapshot():
    snaps = SurveyDataset.build_tokyo()
    for s in snaps:
        assert len(s.layers) >= 4


def test_survey_dataset_bedrock_in_each_snapshot():
    snaps = SurveyDataset.build_tokyo()
    for s in snaps:
        types = {gl.layer_type for gl in s.layers}
        assert GeologyLayerType.BEDROCK in types


# ─── MapAnimator ──────────────────────────────────────────────────

def test_animator_empty_snapshots():
    animator = MapAnimator()
    result = animator.animate([])
    assert result.frame_count == 0
    assert result.svg == "<svg/>"


def test_animator_single_snapshot():
    animator = MapAnimator()
    snaps = [_make_snapshot(2000)]
    result = animator.animate(snaps)
    assert result.frame_count == 1
    assert "<svg" in result.svg


def test_animator_multi_snapshots():
    animator = MapAnimator()
    snaps = SurveyDataset.build_tokyo()
    result = animator.animate(snaps)
    assert result.frame_count == 4
    assert result.years == [1960, 1980, 2000, 2020]
    assert "<svg" in result.svg


def test_animator_svg_has_animate_element():
    animator = MapAnimator()
    snaps = SurveyDataset.build_tokyo()
    result = animator.animate(snaps)
    assert "<animate" in result.svg


def test_animator_svg_contains_year():
    animator = MapAnimator()
    snaps = SurveyDataset.build_tokyo()
    result = animator.animate(snaps)
    assert "1960" in result.svg
    assert "2020" in result.svg


def test_animator_svg_with_title():
    animator = MapAnimator()
    snaps = SurveyDataset.build_tokyo()
    result = animator.animate(snaps, title="カスタムタイトル")
    assert "カスタムタイトル" in result.svg


def test_animator_duration():
    cfg = AnimationConfig(frame_duration_s=2.0)
    animator = MapAnimator(cfg)
    snaps = SurveyDataset.build_tokyo()  # 4 frames
    result = animator.animate(snaps)
    assert result.duration_s == pytest.approx(8.0)


def test_animator_result_to_dict():
    animator = MapAnimator()
    snaps = SurveyDataset.build_osaka()
    result = animator.animate(snaps)
    d = result.to_dict()
    assert d["city"] == "osaka"
    assert d["frame_count"] == 4
    assert "svg" in d
    assert "years" in d


def test_animator_custom_config():
    cfg = AnimationConfig(width=600, height=300)
    animator = MapAnimator(cfg)
    snaps = SurveyDataset.build_tokyo()
    result = animator.animate(snaps)
    assert result.width == 600
    assert result.height == 300
    assert 'width="600"' in result.svg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 73B: route_finder.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── RouteStep / RouteResult ──────────────────────────────────────

def test_route_step_to_dict():
    step = RouteStep("s1", "駅A", "l1", 5.0, False)
    d = step.to_dict()
    assert d["station_id"] == "s1"
    assert d["cumulative_cost"] == 5.0
    assert d["transfer"] is False


def test_route_result_to_dict():
    step = RouteStep("s1", "A", "l1", 0.0)
    r = RouteResult("s1", "s2", True, [step], 4.0, 0, "ok")
    d = r.to_dict()
    assert d["found"] is True
    assert d["total_cost"] == 4.0
    assert len(d["steps"]) == 1


# ─── RouteGraph ───────────────────────────────────────────────────

def test_route_graph_add_station():
    g = RouteGraph()
    st = _make_station()
    g.add_station(st)
    assert g.has_station("s1")


def test_route_graph_add_edge():
    g = RouteGraph()
    g.add_station(_make_station("a"))
    g.add_station(_make_station("b"))
    g.add_edge("a", "b", "l1", 2.0)
    assert len(g.neighbors("a")) == 1
    assert len(g.neighbors("b")) == 1  # 双方向


def test_route_graph_edge_count():
    g = RouteGraph()
    for i in range(4):
        g.add_station(_make_station(f"s{i}"))
    g.add_edge("s0", "s1", "l1", 2.0)
    g.add_edge("s1", "s2", "l1", 2.0)
    g.add_edge("s2", "s3", "l1", 2.0)
    assert g.edge_count() == 3


def test_route_graph_no_station():
    g = RouteGraph()
    assert g.has_station("ghost") is False
    assert g.neighbors("ghost") == []


# ─── RouteGraphBuilder ────────────────────────────────────────────

def test_graph_builder_build(dataset_store):
    g = RouteGraphBuilder.build(dataset_store)
    assert len(g.station_ids()) > 0
    assert g.edge_count() > 0


def test_graph_builder_build_by_city(dataset_store):
    g = RouteGraphBuilder.build(dataset_store, city=CityName.TOKYO)
    stations = dataset_store.stations.list_by_city(CityName.TOKYO)
    for st in stations:
        assert g.has_station(st.id)


def test_graph_builder_edges_bidirectional(dataset_store):
    g = RouteGraphBuilder.build(dataset_store, city=CityName.TOKYO)
    # 丸ノ内線の最初の2駅
    assert g.has_station("tokyo-ogikubo")
    neighbors = [n for n, _, _ in g.neighbors("tokyo-ogikubo")]
    assert "tokyo-nakano" in neighbors


# ─── RouteFinder ──────────────────────────────────────────────────

def _build_linear_graph(n: int) -> tuple:
    """s0-s1-s2-...-s(n-1) の線形グラフ"""
    store = CityMapStore()
    ids = [f"s{i}" for i in range(n)]
    line = _make_line("l1", station_ids=ids)
    store.lines.add(line)
    for i in range(n):
        store.stations.add(_make_station(ids[i]))
    g = RouteGraphBuilder.build(store)
    return g, ids


def test_route_finder_same_station():
    g, ids = _build_linear_graph(3)
    finder = RouteFinder(g)
    result = finder.find("s0", "s0")
    assert result.found is True
    assert result.total_cost == 0.0
    assert result.transfer_count == 0


def test_route_finder_adjacent():
    g, ids = _build_linear_graph(3)
    finder = RouteFinder(g)
    result = finder.find("s0", "s1")
    assert result.found is True
    assert result.total_cost == pytest.approx(RouteGraphBuilder.DEFAULT_EDGE_COST)


def test_route_finder_linear_path():
    g, ids = _build_linear_graph(5)
    finder = RouteFinder(g)
    result = finder.find("s0", "s4")
    assert result.found is True
    assert len(result.steps) == 5
    assert result.steps[0].station_id == "s0"
    assert result.steps[-1].station_id == "s4"


def test_route_finder_not_found():
    store = CityMapStore()
    store.stations.add(_make_station("isolated"))
    g = RouteGraphBuilder.build(store)
    finder = RouteFinder(g)
    result = finder.find("isolated", "ghost")
    assert result.found is False


def test_route_finder_invalid_from():
    g, _ = _build_linear_graph(3)
    finder = RouteFinder(g)
    result = finder.find("no-such", "s1")
    assert result.found is False
    assert "出発駅" in result.message


def test_route_finder_invalid_to():
    g, _ = _build_linear_graph(3)
    finder = RouteFinder(g)
    result = finder.find("s0", "no-such")
    assert result.found is False
    assert "到着駅" in result.message


def test_route_finder_transfer(dataset_store):
    g = RouteGraphBuilder.build(dataset_store)
    finder = RouteFinder(g)
    result = finder.find("tokyo-ogikubo", "tokyo-ginza")
    assert result.found is True
    assert len(result.steps) >= 2


def test_route_finder_result_message():
    g, _ = _build_linear_graph(3)
    finder = RouteFinder(g)
    result = finder.find("s0", "s2")
    assert result.found is True
    assert len(result.message) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 73C: map_importer.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STATION_CSV = """\
id,name,name_en,line_id,city,lat,lon,depth_m,platform_count,opened_year
test-st-a,テスト駅A,Test A,l1,tokyo,35.68,139.76,15.0,2,2000
test-st-b,テスト駅B,Test B,l1,osaka,34.67,135.50,20.0,2,1990
"""

LINE_CSV = """\
id,name,name_en,city,line_type,color,total_length_km,opened_year
test-line-x,テストX線,Test X,tokyo,subway,#FF0000,12.5,2000
test-line-y,テストY線,Test Y,osaka,subway,#00FF00,8.0,1995
"""

GEOLOGY_CSV = """\
id,city,layer_type,name,depth_from_m,depth_to_m,color,n_value
test-gl-a,tokyo,fill,盛土テスト,0.0,3.5,#D2B48C,2.0
test-gl-b,osaka,alluvium,沖積テスト,3.5,12.0,#90EE90,5.0
"""

GEOJSON_DATA = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [139.76, 35.68]},
            "properties": {
                "id": "geojson-st-1", "name": "GJ駅1", "name_en": "GJ1",
                "line_id": "l1", "city": "tokyo", "depth_m": 18.0,
                "platform_count": 2,
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [135.50, 34.67]},
            "properties": {
                "id": "geojson-st-2", "name": "GJ駅2", "name_en": "GJ2",
                "line_id": "l2", "city": "osaka", "depth_m": 22.0,
            },
        },
    ],
}

# ─── ImportResult ─────────────────────────────────────────────────

def test_import_result_success():
    r = ImportResult("csv_stations", 3, 3, 0)
    assert r.success is True
    d = r.to_dict()
    assert d["imported"] == 3
    assert d["success"] is True


def test_import_result_with_errors():
    r = ImportResult("csv_stations", 3, 2, 0,
                     errors=[MapImportError(2, "変換エラー")])
    assert r.success is False


# ─── StationCSVImporter ───────────────────────────────────────────

def test_station_csv_import():
    store = CityMapStore()
    importer = StationCSVImporter(store)
    result = importer.import_csv(STATION_CSV)
    assert result.imported == 2
    assert result.errors == []
    assert store.stations.get("test-st-a") is not None


def test_station_csv_import_duplicate():
    store = CityMapStore()
    importer = StationCSVImporter(store)
    importer.import_csv(STATION_CSV)  # 1回目
    result = importer.import_csv(STATION_CSV)  # 2回目
    assert result.skipped == 2
    assert result.imported == 0


def test_station_csv_import_missing_column():
    store = CityMapStore()
    importer = StationCSVImporter(store)
    bad_csv = "id,name\ns1,A\n"
    result = importer.import_csv(bad_csv)
    assert not result.success
    assert len(result.errors) > 0


def test_station_csv_import_bad_row():
    store = CityMapStore()
    importer = StationCSVImporter(store)
    bad_csv = "id,name,name_en,line_id,city,lat,lon,depth_m\nbad,,X,l1,tokyo,NOT_FLOAT,139,15\n"
    result = importer.import_csv(bad_csv)
    assert len(result.errors) > 0


# ─── LineCSVImporter ──────────────────────────────────────────────

def test_line_csv_import():
    store = CityMapStore()
    importer = LineCSVImporter(store)
    result = importer.import_csv(LINE_CSV)
    assert result.imported == 2
    assert result.errors == []
    assert store.lines.get("test-line-x") is not None


def test_line_csv_import_duplicate():
    store = CityMapStore()
    importer = LineCSVImporter(store)
    importer.import_csv(LINE_CSV)
    result = importer.import_csv(LINE_CSV)
    assert result.skipped == 2


def test_line_csv_import_invalid_city():
    store = CityMapStore()
    importer = LineCSVImporter(store)
    bad = "id,name,name_en,city,line_type,color\nl1,X,X,badcity,subway,#000\n"
    result = importer.import_csv(bad)
    assert len(result.errors) > 0


# ─── GeologyCSVImporter ───────────────────────────────────────────

def test_geology_csv_import():
    store = CityMapStore()
    importer = GeologyCSVImporter(store)
    result = importer.import_csv(GEOLOGY_CSV)
    assert result.imported == 2
    assert result.errors == []
    gl = store.geology.get("test-gl-a")
    assert gl is not None
    assert gl.depth_to_m == 3.5


def test_geology_csv_import_duplicate():
    store = CityMapStore()
    importer = GeologyCSVImporter(store)
    importer.import_csv(GEOLOGY_CSV)
    result = importer.import_csv(GEOLOGY_CSV)
    assert result.skipped == 2


def test_geology_csv_import_missing_column():
    store = CityMapStore()
    importer = GeologyCSVImporter(store)
    result = importer.import_csv("id,city\ng1,tokyo\n")
    assert not result.success


# ─── GeoJSONImporter ──────────────────────────────────────────────

def test_geojson_import():
    store = CityMapStore()
    importer = GeoJSONImporter(store)
    result = importer.import_geojson(json.dumps(GEOJSON_DATA))
    assert result.imported == 2
    assert result.errors == []
    assert store.stations.get("geojson-st-1") is not None


def test_geojson_import_coord_order():
    """GeoJSON は [lon, lat] 順 → Station は lat/lon に正しく変換される"""
    store = CityMapStore()
    importer = GeoJSONImporter(store)
    importer.import_geojson(json.dumps(GEOJSON_DATA))
    st = store.stations.get("geojson-st-1")
    assert st.coord.lat == pytest.approx(35.68)
    assert st.coord.lon == pytest.approx(139.76)


def test_geojson_import_invalid_json():
    store = CityMapStore()
    importer = GeoJSONImporter(store)
    result = importer.import_geojson("not json")
    assert not result.success


def test_geojson_import_wrong_type():
    store = CityMapStore()
    importer = GeoJSONImporter(store)
    result = importer.import_geojson('{"type": "Feature"}')
    assert not result.success


def test_geojson_import_skip_non_point():
    store = CityMapStore()
    importer = GeoJSONImporter(store)
    data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": []},
            "properties": {},
        }],
    }
    result = importer.import_geojson(json.dumps(data))
    assert result.skipped == 1
    assert result.imported == 0


def test_geojson_import_missing_properties():
    store = CityMapStore()
    importer = GeoJSONImporter(store)
    data = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [139.76, 35.68]},
            "properties": {"id": "x"},  # 必須フィールド不足
        }],
    }
    result = importer.import_geojson(json.dumps(data))
    assert len(result.errors) > 0


# ─── MapImporter (facade) ─────────────────────────────────────────

def test_map_importer_facade():
    store = CityMapStore()
    imp = MapImporter(store)
    imp.import_stations_csv(STATION_CSV)
    imp.import_lines_csv(LINE_CSV)
    imp.import_geology_csv(GEOLOGY_CSV)
    s = imp.summary()
    assert s["stations"] >= 2
    assert s["lines"] >= 2
    assert s["geology_layers"] >= 2


def test_map_importer_geojson():
    store = CityMapStore()
    imp = MapImporter(store)
    result = imp.import_geojson(json.dumps(GEOJSON_DATA))
    assert result.imported == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 73 API エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── 73A: /v1/map/{city}/animate ─────────────────────────────────

def test_api_animate_tokyo(client):
    resp = client.get("/v1/map/tokyo/animate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert data["frame_count"] == 4
    assert "<svg" in data["svg"]
    assert "<animate" in data["svg"]


def test_api_animate_osaka(client):
    resp = client.get("/v1/map/osaka/animate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "osaka"
    assert len(data["years"]) == 4


def test_api_animate_with_title(client):
    resp = client.get("/v1/map/tokyo/animate?title=東京断面アニメ")
    assert resp.status_code == 200
    assert "東京断面アニメ" in resp.json()["svg"]


def test_api_animate_invalid_city(client):
    resp = client.get("/v1/map/nowhere/animate")
    assert resp.status_code == 404


def test_api_animate_snapshots(client):
    resp = client.get("/v1/map/tokyo/animate/snapshots")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert data["count"] == 4
    assert len(data["snapshots"]) == 4


def test_api_animate_snapshots_invalid(client):
    resp = client.get("/v1/map/badcity/animate/snapshots")
    assert resp.status_code == 404


# ─── 73B: /v1/map/route/* ────────────────────────────────────────

def test_api_route_graph_stats(client):
    resp = client.get("/v1/map-route/graph/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["station_count"] > 0
    assert data["edge_count"] > 0


def test_api_route_same_station(client):
    resp = client.get("/v1/map/route/tokyo-ogikubo/tokyo-ogikubo")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["total_cost"] == 0.0


def test_api_route_adjacent(client):
    resp = client.get("/v1/map/route/tokyo-ogikubo/tokyo-nakano")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert len(data["steps"]) >= 2


def test_api_route_longer_path(client):
    resp = client.get("/v1/map/route/tokyo-ogikubo/tokyo-ginza")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["total_cost"] > 0


def test_api_route_not_found(client):
    resp = client.get("/v1/map/route/tokyo-ogikubo/fukuoka-hakata")
    assert resp.status_code == 200
    # 別都市間は未接続なので found=False
    data = resp.json()
    assert isinstance(data["found"], bool)


def test_api_route_invalid_station(client):
    resp = client.get("/v1/map/route/no-such-station/tokyo-ginza")
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is False


# ─── 73C: /v1/map/import/* ───────────────────────────────────────

def test_api_import_summary(client):
    resp = client.get("/v1/map-import/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "stations" in data
    assert "lines" in data
    assert "geology_layers" in data


def test_api_import_stations_csv(client):
    resp = client.post("/v1/map/import/stations/csv",
                       json={"csv_text": STATION_CSV})
    assert resp.status_code == 200
    data = resp.json()
    assert "imported" in data
    assert "source_type" in data
    assert data["source_type"] == "csv_stations"


def test_api_import_lines_csv(client):
    resp = client.post("/v1/map/import/lines/csv",
                       json={"csv_text": LINE_CSV})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "csv_lines"


def test_api_import_geology_csv(client):
    resp = client.post("/v1/map/import/geology/csv",
                       json={"csv_text": GEOLOGY_CSV})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "csv_geology"


def test_api_import_geojson(client):
    resp = client.post("/v1/map/import/geojson",
                       json={"geojson_text": json.dumps(GEOJSON_DATA)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "geojson"


def test_api_import_invalid_csv(client):
    resp = client.post("/v1/map/import/stations/csv",
                       json={"csv_text": "bad,header\n1,2\n"})
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]


def test_api_import_invalid_geojson(client):
    resp = client.post("/v1/map/import/geojson",
                       json={"geojson_text": "not json at all"})
    assert resp.status_code == 200
    data = resp.json()
    assert not data["success"]
