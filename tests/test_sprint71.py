"""
Sprint 71 — 主要都市地図ビジュアライザ テスト

対象:
  71A: open_mythos/skills/city_map.py
  71B: open_mythos/skills/map_renderer.py
  71C: serve/api.py (/v1/map/*)
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from open_mythos.skills.city_map import (
    CityName, LineType, GeologyLayerType,
    GeoCoord, Station, MetroLine, GeologyLayer, CityMapData,
    StationStore, MetroLineStore, GeologyStore, CityMapStore,
    CityMapDataset,
)
from open_mythos.skills.map_renderer import (
    CrossSectionConfig,
    CrossSectionResult,
    SVGCrossSectionRenderer,
    CrossSectionEngine,
)


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def dataset_store():
    return CityMapDataset.build()


@pytest.fixture
def client(dataset_store):
    from serve.api import app
    return TestClient(app)


def _make_station(sid="s1", city=CityName.TOKYO, line_id="l1", depth=15.0):
    return Station(
        id=sid, name=f"駅{sid}", name_en=f"Station {sid}",
        line_id=line_id, city=city,
        coord=GeoCoord(35.68, 139.76),
        depth_m=depth, platform_count=2, opened_year=2000,
    )


def _make_line(lid="l1", city=CityName.TOKYO):
    return MetroLine(
        id=lid, name="テスト線", name_en="Test Line",
        city=city, line_type=LineType.SUBWAY,
        color="#FF0000", station_ids=["s1", "s2"],
        total_length_km=10.0, opened_year=2000,
    )


def _make_geology(gid="g1", city=CityName.TOKYO, depth_from=0.0, depth_to=10.0):
    return GeologyLayer(
        id=gid, city=city,
        layer_type=GeologyLayerType.ALLUVIUM,
        name="沖積層", depth_from_m=depth_from, depth_to_m=depth_to,
        color="#90EE90", n_value=5.0,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 71A: city_map.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── Enums ────────────────────────────────────────────────────────

def test_city_name_values():
    assert CityName.TOKYO.value == "tokyo"
    assert CityName.OSAKA.value == "osaka"
    assert CityName.NAGOYA.value == "nagoya"
    assert CityName.YOKOHAMA.value == "yokohama"
    assert CityName.FUKUOKA.value == "fukuoka"


def test_line_type_values():
    assert LineType.SUBWAY.value == "subway"
    assert LineType.JR.value == "jr"
    assert LineType.PRIVATE.value == "private"


def test_geology_layer_type_values():
    types = [t.value for t in GeologyLayerType]
    assert "fill" in types
    assert "alluvium" in types
    assert "bedrock" in types


# ─── GeoCoord ─────────────────────────────────────────────────────

def test_geocoord_to_dict():
    c = GeoCoord(35.68, 139.76)
    d = c.to_dict()
    assert d["lat"] == 35.68
    assert d["lon"] == 139.76


# ─── Station ──────────────────────────────────────────────────────

def test_station_to_dict():
    st = _make_station()
    d = st.to_dict()
    assert d["id"] == "s1"
    assert d["city"] == "tokyo"
    assert d["depth_m"] == 15.0
    assert "coord" in d


def test_station_coord_in_dict():
    st = _make_station()
    d = st.to_dict()
    assert "lat" in d["coord"]
    assert "lon" in d["coord"]


# ─── MetroLine ────────────────────────────────────────────────────

def test_metro_line_to_dict():
    ln = _make_line()
    d = ln.to_dict()
    assert d["id"] == "l1"
    assert d["line_type"] == "subway"
    assert d["color"] == "#FF0000"
    assert d["station_ids"] == ["s1", "s2"]


def test_metro_line_city():
    ln = _make_line(city=CityName.OSAKA)
    assert ln.city == CityName.OSAKA
    assert ln.to_dict()["city"] == "osaka"


# ─── GeologyLayer ─────────────────────────────────────────────────

def test_geology_layer_thickness():
    gl = _make_geology(depth_from=0.0, depth_to=10.0)
    assert gl.thickness_m == 10.0


def test_geology_layer_to_dict():
    gl = _make_geology()
    d = gl.to_dict()
    assert d["layer_type"] == "alluvium"
    assert d["depth_from_m"] == 0.0
    assert d["depth_to_m"] == 10.0
    assert d["thickness_m"] == 10.0
    assert d["color"] == "#90EE90"


def test_geology_layer_n_value():
    gl = _make_geology()
    assert gl.n_value == 5.0
    d = gl.to_dict()
    assert d["n_value"] == 5.0


# ─── StationStore ─────────────────────────────────────────────────

def test_station_store_add_get():
    store = StationStore()
    st = _make_station()
    store.add(st)
    assert store.get("s1") is st


def test_station_store_list_by_city():
    store = StationStore()
    store.add(_make_station("s1", CityName.TOKYO))
    store.add(_make_station("s2", CityName.OSAKA))
    tokyo = store.list_by_city(CityName.TOKYO)
    assert len(tokyo) == 1
    assert tokyo[0].id == "s1"


def test_station_store_list_by_line():
    store = StationStore()
    store.add(_make_station("s1", line_id="lineA"))
    store.add(_make_station("s2", line_id="lineB"))
    result = store.list_by_line("lineA")
    assert len(result) == 1


def test_station_store_len():
    store = StationStore()
    store.add(_make_station("s1"))
    store.add(_make_station("s2"))
    assert len(store) == 2


def test_station_store_all():
    store = StationStore()
    store.add(_make_station("s1"))
    store.add(_make_station("s2"))
    assert len(store.all()) == 2


# ─── MetroLineStore ───────────────────────────────────────────────

def test_metro_line_store_add_get():
    store = MetroLineStore()
    ln = _make_line()
    store.add(ln)
    assert store.get("l1") is ln


def test_metro_line_store_list_by_city():
    store = MetroLineStore()
    store.add(_make_line("l1", CityName.TOKYO))
    store.add(_make_line("l2", CityName.OSAKA))
    osaka = store.list_by_city(CityName.OSAKA)
    assert len(osaka) == 1
    assert osaka[0].id == "l2"


def test_metro_line_store_len():
    store = MetroLineStore()
    store.add(_make_line("l1"))
    store.add(_make_line("l2"))
    assert len(store) == 2


# ─── GeologyStore ─────────────────────────────────────────────────

def test_geology_store_add_get():
    store = GeologyStore()
    gl = _make_geology()
    store.add(gl)
    assert store.get("g1") is gl


def test_geology_store_list_by_city_sorted():
    store = GeologyStore()
    store.add(_make_geology("g2", depth_from=10.0, depth_to=20.0))
    store.add(_make_geology("g1", depth_from=0.0, depth_to=10.0))
    result = store.list_by_city(CityName.TOKYO)
    assert result[0].id == "g1"
    assert result[1].id == "g2"


def test_geology_store_len():
    store = GeologyStore()
    store.add(_make_geology("g1"))
    assert len(store) == 1


# ─── CityMapStore ─────────────────────────────────────────────────

def test_city_map_store_get_city_data():
    store = CityMapStore()
    store.lines.add(_make_line())
    store.stations.add(_make_station())
    store.geology.add(_make_geology())
    data = store.get_city_data(CityName.TOKYO)
    assert isinstance(data, CityMapData)
    assert len(data.lines) == 1
    assert len(data.stations) == 1
    assert len(data.geology_layers) == 1


def test_city_map_store_cities():
    store = CityMapStore()
    store.stations.add(_make_station("s1", CityName.TOKYO))
    store.stations.add(_make_station("s2", CityName.OSAKA))
    cities = store.cities()
    assert "tokyo" in cities
    assert "osaka" in cities


# ─── CityMapData ──────────────────────────────────────────────────

def test_city_map_data_to_dict():
    data = CityMapData(
        city=CityName.TOKYO,
        lines=[_make_line()],
        stations=[_make_station()],
        geology_layers=[_make_geology()],
    )
    d = data.to_dict()
    assert d["city"] == "tokyo"
    assert len(d["lines"]) == 1
    assert len(d["stations"]) == 1
    assert len(d["geology_layers"]) == 1


def test_city_map_data_to_geojson():
    data = CityMapData(
        city=CityName.TOKYO,
        lines=[],
        stations=[_make_station()],
        geology_layers=[],
    )
    gj = data.to_geojson()
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 1
    feat = gj["features"][0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Point"
    assert len(feat["geometry"]["coordinates"]) == 2


# ─── CityMapDataset (プリセットデータ) ───────────────────────────

def test_dataset_build_returns_store(dataset_store):
    assert isinstance(dataset_store, CityMapStore)


def test_dataset_has_all_cities(dataset_store):
    cities = dataset_store.cities()
    for city in ["tokyo", "osaka", "nagoya", "yokohama", "fukuoka"]:
        assert city in cities


def test_dataset_tokyo_marunouchi_line(dataset_store):
    line = dataset_store.lines.get("tokyo-marunouchi")
    assert line is not None
    assert line.name == "丸ノ内線"
    assert line.city == CityName.TOKYO


def test_dataset_tokyo_station_count(dataset_store):
    stations = dataset_store.stations.list_by_city(CityName.TOKYO)
    assert len(stations) >= 5


def test_dataset_tokyo_geology(dataset_store):
    layers = dataset_store.geology.list_by_city(CityName.TOKYO)
    assert len(layers) >= 4
    assert layers[0].depth_from_m < layers[-1].depth_from_m


def test_dataset_osaka_station(dataset_store):
    stations = dataset_store.stations.list_by_city(CityName.OSAKA)
    assert len(stations) >= 3


def test_dataset_nagoya_geology(dataset_store):
    layers = dataset_store.geology.list_by_city(CityName.NAGOYA)
    assert len(layers) >= 3


def test_dataset_yokohama_line(dataset_store):
    line = dataset_store.lines.get("yokohama-blue")
    assert line is not None
    assert line.name == "ブルーライン"


def test_dataset_fukuoka_station(dataset_store):
    stations = dataset_store.stations.list_by_city(CityName.FUKUOKA)
    assert len(stations) >= 3


def test_dataset_all_stations_have_depth(dataset_store):
    for st in dataset_store.stations.all():
        assert st.depth_m > 0


def test_dataset_geology_bedrock_deepest(dataset_store):
    for city in CityName:
        layers = dataset_store.geology.list_by_city(city)
        if layers:
            deepest = layers[-1]
            assert deepest.layer_type == GeologyLayerType.BEDROCK


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 71B: map_renderer.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── CrossSectionConfig ───────────────────────────────────────────

def test_config_defaults():
    cfg = CrossSectionConfig()
    assert cfg.width == 1200
    assert cfg.height == 600
    assert cfg.max_depth_m == 60.0


def test_config_plot_width():
    cfg = CrossSectionConfig(width=1200, margin_left=80, margin_right=40)
    assert cfg.plot_width == 1080


def test_config_depth_to_y_surface():
    cfg = CrossSectionConfig()
    y = cfg.depth_to_y(0.0)
    assert y == pytest.approx(cfg.surface_y, abs=1)


def test_config_depth_to_y_max():
    cfg = CrossSectionConfig()
    y = cfg.depth_to_y(cfg.max_depth_m)
    assert y < cfg.height


def test_config_depth_to_y_ordering():
    cfg = CrossSectionConfig()
    y10 = cfg.depth_to_y(10.0)
    y30 = cfg.depth_to_y(30.0)
    assert y30 > y10  # 深いほど Y が大きい


# ─── SVGCrossSectionRenderer ──────────────────────────────────────

def test_renderer_returns_svg_string():
    renderer = SVGCrossSectionRenderer()
    line = _make_line()
    stations = [_make_station("s1", depth=15.0), _make_station("s2", depth=20.0)]
    geology = [_make_geology()]
    svg = renderer.render(line, stations, geology)
    assert isinstance(svg, str)
    assert "<svg" in svg
    assert "</svg>" in svg


def test_renderer_svg_has_xml_declaration():
    renderer = SVGCrossSectionRenderer()
    svg = renderer.render(_make_line(), [], [])
    assert svg.startswith('<?xml')


def test_renderer_svg_contains_rect():
    renderer = SVGCrossSectionRenderer()
    svg = renderer.render(_make_line(), [_make_station()], [_make_geology()])
    assert "<rect" in svg


def test_renderer_svg_contains_circle_for_station():
    renderer = SVGCrossSectionRenderer()
    svg = renderer.render(_make_line(), [_make_station()], [])
    assert "<circle" in svg


def test_renderer_svg_with_title():
    renderer = SVGCrossSectionRenderer()
    svg = renderer.render(_make_line(), [], [], title="テストタイトル")
    assert "テストタイトル" in svg


def test_renderer_custom_config():
    cfg = CrossSectionConfig(width=800, height=400)
    renderer = SVGCrossSectionRenderer(cfg)
    svg = renderer.render(_make_line(), [], [])
    assert 'width="800"' in svg
    assert 'height="400"' in svg


def test_renderer_multiple_stations():
    renderer = SVGCrossSectionRenderer()
    stations = [_make_station(f"s{i}", depth=10.0 + i * 2) for i in range(5)]
    svg = renderer.render(_make_line(), stations, [])
    assert svg.count("<circle") == 5


def test_renderer_no_geology_no_error():
    renderer = SVGCrossSectionRenderer()
    svg = renderer.render(_make_line(), [_make_station()], [])
    assert "<svg" in svg


# ─── CrossSectionResult ───────────────────────────────────────────

def test_cross_section_result_to_dict():
    result = CrossSectionResult(
        city="tokyo", line_id="l1", svg="<svg/>",
        station_count=5, geology_count=6,
        width=1200, height=600,
    )
    d = result.to_dict()
    assert d["city"] == "tokyo"
    assert d["line_id"] == "l1"
    assert d["format"] == "svg"
    assert d["station_count"] == 5


# ─── CrossSectionEngine ───────────────────────────────────────────

def test_engine_generate(dataset_store):
    engine = CrossSectionEngine(dataset_store)
    result = engine.generate(CityName.TOKYO, "tokyo-marunouchi")
    assert result is not None
    assert result.city == "tokyo"
    assert result.station_count > 0
    assert "<svg" in result.svg


def test_engine_generate_invalid_city(dataset_store):
    engine = CrossSectionEngine(dataset_store)
    result = engine.generate(CityName.OSAKA, "tokyo-marunouchi")
    assert result is None  # city mismatch


def test_engine_generate_invalid_line(dataset_store):
    engine = CrossSectionEngine(dataset_store)
    result = engine.generate(CityName.TOKYO, "no-such-line")
    assert result is None


def test_engine_generate_with_title(dataset_store):
    engine = CrossSectionEngine(dataset_store)
    result = engine.generate(CityName.TOKYO, "tokyo-marunouchi", title="丸ノ内線断面図")
    assert result is not None
    assert "丸ノ内線断面図" in result.svg


def test_engine_generate_all_lines(dataset_store):
    engine = CrossSectionEngine(dataset_store)
    results = engine.generate_all_lines(CityName.TOKYO)
    assert len(results) >= 1
    for r in results:
        assert r.city == "tokyo"
        assert r.svg


def test_engine_osaka(dataset_store):
    engine = CrossSectionEngine(dataset_store)
    result = engine.generate(CityName.OSAKA, "osaka-midosuji")
    assert result is not None
    assert result.geology_count > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 71C: /v1/map/* API エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_api_map_cities(client):
    resp = client.get("/v1/map/cities")
    assert resp.status_code == 200
    data = resp.json()
    assert "cities" in data
    assert "count" in data
    assert len(data["cities"]) >= 5


def test_api_map_cities_includes_all(client):
    resp = client.get("/v1/map/cities")
    cities = resp.json()["cities"]
    for city in ["tokyo", "osaka", "nagoya", "yokohama", "fukuoka"]:
        assert city in cities


def test_api_map_city_lines_tokyo(client):
    resp = client.get("/v1/map/tokyo/lines")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert data["count"] >= 1
    assert len(data["lines"]) >= 1


def test_api_map_city_lines_osaka(client):
    resp = client.get("/v1/map/osaka/lines")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "osaka"
    assert data["count"] >= 1


def test_api_map_city_lines_invalid(client):
    resp = client.get("/v1/map/invalid_city/lines")
    assert resp.status_code == 404


def test_api_map_city_stations_tokyo(client):
    resp = client.get("/v1/map/tokyo/stations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert data["count"] >= 5


def test_api_map_city_stations_fields(client):
    resp = client.get("/v1/map/tokyo/stations")
    stations = resp.json()["stations"]
    st = stations[0]
    assert "id" in st
    assert "name" in st
    assert "depth_m" in st
    assert "coord" in st


def test_api_map_city_geology_tokyo(client):
    resp = client.get("/v1/map/tokyo/geology")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert data["count"] >= 4


def test_api_map_city_geology_layers_sorted(client):
    resp = client.get("/v1/map/tokyo/geology")
    layers = resp.json()["geology_layers"]
    depths = [ln["depth_from_m"] for ln in layers]
    assert depths == sorted(depths)


def test_api_map_city_geology_invalid(client):
    resp = client.get("/v1/map/badcity/geology")
    assert resp.status_code == 404


def test_api_map_city_geojson(client):
    resp = client.get("/v1/map/tokyo/geojson")
    assert resp.status_code == 200
    gj = resp.json()
    assert gj["type"] == "FeatureCollection"
    assert "features" in gj
    assert len(gj["features"]) >= 5


def test_api_map_geojson_feature_structure(client):
    resp = client.get("/v1/map/osaka/geojson")
    features = resp.json()["features"]
    feat = features[0]
    assert feat["type"] == "Feature"
    assert feat["geometry"]["type"] == "Point"
    coords = feat["geometry"]["coordinates"]
    assert len(coords) == 2


def test_api_map_cross_section_tokyo(client):
    resp = client.get("/v1/map/tokyo/tokyo-marunouchi/cross-section")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert data["line_id"] == "tokyo-marunouchi"
    assert "<svg" in data["svg"]
    assert data["station_count"] > 0
    assert data["format"] == "svg"


def test_api_map_cross_section_osaka(client):
    resp = client.get("/v1/map/osaka/osaka-midosuji/cross-section")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "osaka"
    assert data["geology_count"] > 0


def test_api_map_cross_section_with_title(client):
    resp = client.get("/v1/map/tokyo/tokyo-marunouchi/cross-section?title=テスト断面図")
    assert resp.status_code == 200
    assert "テスト断面図" in resp.json()["svg"]


def test_api_map_cross_section_invalid_city(client):
    resp = client.get("/v1/map/unknown/tokyo-marunouchi/cross-section")
    assert resp.status_code == 404


def test_api_map_cross_section_invalid_line(client):
    resp = client.get("/v1/map/tokyo/no-such-line/cross-section")
    assert resp.status_code == 404


def test_api_map_city_summary_tokyo(client):
    resp = client.get("/v1/map/tokyo/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert data["lines"] >= 1
    assert data["stations"] >= 5
    assert data["geology_layers"] >= 4
    assert data["max_station_depth_m"] > 0
    assert data["avg_station_depth_m"] > 0


def test_api_map_city_summary_all_cities(client):
    for city in ["tokyo", "osaka", "nagoya", "yokohama", "fukuoka"]:
        resp = client.get(f"/v1/map/{city}/summary")
        assert resp.status_code == 200, f"Failed for city: {city}"


def test_api_map_city_summary_invalid(client):
    resp = client.get("/v1/map/nowhere/summary")
    assert resp.status_code == 404
