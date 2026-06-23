"""
Sprint 71 — 主要都市地図ビジュアライザ テスト

対象:
  - open_mythos/skills/city_map.py
      GeoPoint, Station, GeologyLayer, Line, City, CrossSection,
      SampleCityDataSource, GTFSCityDataSource, GeologyModel,
      CrossSectionBuilder, CityMapStore, CityMapFactory, haversine_m
  - open_mythos/skills/map_renderer.py
      SvgStyle, CrossSectionSvgRenderer, FrontViewSvgRenderer, MapRenderer
  - serve/map_router.py
      GET  /v1/map/cities
      GET  /v1/map/{city}/lines
      GET  /v1/map/{city}/{line}/cross-section
      GET  /v1/map/{city}/{line}/front-view
      POST /v1/map/cross-section

テスト構成:
  Section A: データクラス (to_dict / from_dict / 整合性)
  Section B: SampleCityDataSource (13都市)
  Section C: GTFSCityDataSource (取り込み + フォールバック)
  Section D: GeologyModel (層順序・深度)
  Section E: CrossSectionBuilder (統合・例外)
  Section F: SVGレンダラー
  Section G: MapRenderer / PNG フォールバック
  Section H: CityMapStore / Factory
  Section I: API エンドポイント (/v1/map/*)
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# transformers モック (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def mock_transformers():
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock
    yield mock
    sys.modules.pop("transformers", None)


# ---------------------------------------------------------------------------
# 共通インポート
# ---------------------------------------------------------------------------

from open_mythos.skills.city_map import (
    GeoPoint,
    Station,
    GeologyLayer,
    Line,
    City,
    CrossSection,
    SampleCityDataSource,
    GTFSCityDataSource,
    GeologyModel,
    CrossSectionBuilder,
    CityMapStore,
    CityMapFactory,
    haversine_m,
)
from open_mythos.skills.map_renderer import (
    SvgStyle,
    CrossSectionSvgRenderer,
    FrontViewSvgRenderer,
    MapRenderer,
)


# ---------------------------------------------------------------------------
# ヘルパ: 最小 GTFS zip を生成
# ---------------------------------------------------------------------------

def _make_gtfs_zip(
    *,
    with_required: bool = True,
    n_stops: int = 4,
) -> bytes:
    """テスト用の最小 GTFS zip バイト列を生成する。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if with_required:
            # routes.txt
            rout = io.StringIO()
            w = csv.writer(rout)
            w.writerow(["route_id", "route_short_name", "route_color"])
            w.writerow(["R1", "テスト線", "FF0000"])
            zf.writestr("routes.txt", rout.getvalue())
            # stops.txt
            stp = io.StringIO()
            w = csv.writer(stp)
            w.writerow(["stop_id", "stop_name", "stop_lat", "stop_lon"])
            for i in range(n_stops):
                w.writerow(["S%d" % i, "駅%d" % i, 35.0 + i * 0.01, 139.0 + i * 0.01])
            zf.writestr("stops.txt", stp.getvalue())
        else:
            zf.writestr("other.txt", "no required files")
    return buf.getvalue()


# ===========================================================================
# Section A: データクラス
# ===========================================================================

class TestDataModels:
    def test_geopoint_roundtrip(self):
        p = GeoPoint(35.6812, 139.7671)
        d = p.to_dict()
        assert d["lat"] == 35.6812 and d["lon"] == 139.7671
        assert GeoPoint.from_dict(d).lat == p.lat

    def test_station_to_from_dict(self):
        st = Station("S1", "東京", GeoPoint(35.68, 139.76), "L1", 0, depth_m=18.0, name_kana="トウキョウ")
        d = st.to_dict()
        assert d["station_id"] == "S1"
        assert d["depth_m"] == 18.0
        back = Station.from_dict(d)
        assert back.name == "東京" and back.order == 0 and back.depth_m == 18.0

    def test_station_depth_none(self):
        st = Station("S1", "X", GeoPoint(0, 0), "L1", 0)
        assert st.depth_m is None
        assert st.to_dict()["depth_m"] is None

    def test_geology_layer_thickness(self):
        g = GeologyLayer("L0", "沖積層", 0.0, 12.0, "#D7C49E", "clay")
        assert g.thickness_m == 12.0
        assert g.top_m < g.bottom_m
        assert g.to_dict()["thickness_m"] == 12.0

    def test_line_roundtrip(self):
        ln = Line("L1", "丸ノ内線", "tokyo", "#F62E36", stations=[
            Station("S0", "A", GeoPoint(0, 0), "L1", 0),
        ], operator="東京メトロ")
        d = ln.to_dict()
        assert d["station_count"] == 1
        assert Line.from_dict(d).operator == "東京メトロ"

    def test_city_get_line(self):
        ln = Line("L1", "X", "c1", "#000")
        city = City("c1", "都市", "City", GeoPoint(0, 0), lines=[ln])
        assert city.get_line("L1") is ln
        assert city.get_line("none") is None
        assert city.line_count == 1

    def test_cross_section_to_dict(self):
        cs = CrossSection("tokyo", "L1", "丸ノ内線", [], [], 0.0, 30.0, "sample")
        d = cs.to_dict()
        assert d["generated_by"] == "sample"
        assert d["max_depth_m"] == 30.0
        assert d["station_count"] == 0

    def test_haversine_positive(self):
        a = GeoPoint(35.0, 139.0)
        b = GeoPoint(35.1, 139.0)
        assert haversine_m(a, b) > 0
        assert haversine_m(a, a) == 0


# ===========================================================================
# Section B: SampleCityDataSource (13都市)
# ===========================================================================

class TestSampleSource:
    @pytest.fixture(scope="class")
    def src(self):
        return SampleCityDataSource()

    def test_13_cities(self, src):
        cities = src.load_cities()
        assert len(cities) == 13

    def test_required_cities_present(self, src):
        ids = {c.city_id for c in src.load_cities()}
        for cid in ("tokyo", "yokohama", "osaka", "nagoya", "sapporo",
                    "fukuoka", "kobe", "kawasaki", "kyoto", "saitama",
                    "hiroshima", "sendai", "chiba"):
            assert cid in ids

    def test_each_city_has_line(self, src):
        for c in src.load_cities():
            assert c.line_count >= 1, c.city_id

    def test_station_order_sequential(self, src):
        lines = src.load_lines("tokyo")
        for ln in lines:
            orders = [s.order for s in ln.stations]
            assert orders == list(range(len(orders)))

    def test_unknown_city_empty(self, src):
        assert src.load_lines("atlantis") == []

    def test_load_city(self, src):
        assert src.load_city("tokyo").name == "東京"
        assert src.load_city("nope") is None

    def test_source_kind(self, src):
        assert src.source_kind == "sample"


# ===========================================================================
# Section C: GTFSCityDataSource (取り込み + フォールバック)
# ===========================================================================

class TestGTFSSource:
    def test_valid_zip_parsed(self):
        zb = _make_gtfs_zip(n_stops=5)
        src = GTFSCityDataSource("tokyo", _zip_bytes=zb)
        lines = src.load_lines("tokyo")
        assert src.source_kind == "gtfs"
        assert len(lines) >= 1
        assert lines[0].station_count == 5

    def test_invalid_zip_falls_back(self):
        zb = _make_gtfs_zip(with_required=False)
        src = GTFSCityDataSource("tokyo", _zip_bytes=zb)
        lines = src.load_lines("tokyo")
        assert src.source_kind == "sample(fallback)"
        # サンプルの丸ノ内線(6駅)が返る
        assert any(ln.line_id == "marunouchi" for ln in lines)

    def test_no_url_no_bytes_falls_back(self):
        src = GTFSCityDataSource("osaka")  # url も bytes も無し
        lines = src.load_lines("osaka")
        assert src.source_kind == "sample(fallback)"
        assert len(lines) >= 1

    def test_fetch_failure_falls_back(self, monkeypatch):
        def _boom(*a, **k):
            raise OSError("network down")
        monkeypatch.setattr("urllib.request.urlopen", _boom)
        src = GTFSCityDataSource("tokyo", gtfs_url="http://example.invalid/gtfs.zip")
        lines = src.load_lines("tokyo")
        assert src.source_kind == "sample(fallback)"
        assert len(lines) >= 1

    def test_other_city_uses_sample(self):
        zb = _make_gtfs_zip()
        src = GTFSCityDataSource("tokyo", _zip_bytes=zb)
        # 対象外の都市は常にサンプル
        osaka = src.load_lines("osaka")
        assert any(ln.line_id == "midosuji" for ln in osaka)

    def test_load_cities_swaps_target(self):
        zb = _make_gtfs_zip(n_stops=3)
        src = GTFSCityDataSource("tokyo", _zip_bytes=zb)
        cities = src.load_cities()
        tokyo = next(c for c in cities if c.city_id == "tokyo")
        assert tokyo.lines[0].station_count == 3


# ===========================================================================
# Section D: GeologyModel
# ===========================================================================

class TestGeologyModel:
    @pytest.fixture(scope="class")
    def gm(self):
        return GeologyModel()

    def test_layers_ordered(self, gm):
        layers = gm.estimate_layers("tokyo")
        assert len(layers) >= 3
        for a, b in zip(layers, layers[1:]):
            assert a.bottom_m <= b.top_m + 1e-9  # 連続・順序

    def test_layers_start_at_zero(self, gm):
        layers = gm.estimate_layers("tokyo")
        assert layers[0].top_m == 0.0

    def test_tokyo_has_alluvial(self, gm):
        names = [g.name for g in gm.estimate_layers("tokyo")]
        assert "沖積層" in names

    def test_default_profile_for_unknown(self, gm):
        layers = gm.estimate_layers("unknown_city")
        assert len(layers) >= 3

    def test_underground_depth_positive(self, gm):
        ln = Line("marunouchi", "丸ノ内線", "tokyo", "#F62E36", stations=[
            Station("S0", "A", GeoPoint(0, 0), "marunouchi", 0),
            Station("S1", "B", GeoPoint(0, 0), "marunouchi", 1),
        ])
        d = gm.estimate_depth(ln.stations[1], ln)
        assert d > 0

    def test_monorail_above_ground(self, gm):
        ln = Line("monorail", "千葉都市モノレール1号線", "chiba", "#E60012", stations=[
            Station("C0", "千葉", GeoPoint(0, 0), "monorail", 0),
        ])
        d = gm.estimate_depth(ln.stations[0], ln)
        assert d < 0  # 高架 = 負


# ===========================================================================
# Section E: CrossSectionBuilder
# ===========================================================================

class TestCrossSectionBuilder:
    @pytest.fixture(scope="class")
    def src(self):
        return SampleCityDataSource()

    def test_build_tokyo(self, src):
        cs = CrossSectionBuilder().build("tokyo", "marunouchi", src)
        assert cs.city_id == "tokyo"
        assert cs.line_name == "丸ノ内線"
        assert len(cs.stations) == 6
        assert len(cs.layers) >= 3

    def test_all_stations_have_depth(self, src):
        cs = CrossSectionBuilder().build("tokyo", "marunouchi", src)
        assert all(s.depth_m is not None for s in cs.stations)

    def test_total_distance_positive(self, src):
        cs = CrossSectionBuilder().build("osaka", "midosuji", src)
        assert cs.total_distance_m > 0

    def test_max_depth_consistent(self, src):
        cs = CrossSectionBuilder().build("tokyo", "marunouchi", src)
        assert cs.max_depth_m >= max(s.depth_m for s in cs.stations)

    def test_generated_by_sample(self, src):
        cs = CrossSectionBuilder().build("tokyo", "ginza", src)
        assert cs.generated_by == "sample"

    def test_unknown_line_raises(self, src):
        with pytest.raises(ValueError):
            CrossSectionBuilder().build("tokyo", "nonexistent", src)

    def test_unknown_city_raises(self, src):
        with pytest.raises(ValueError):
            CrossSectionBuilder().build("atlantis", "x", src)


# ===========================================================================
# Section F: SVGレンダラー
# ===========================================================================

class TestSvgRenderer:
    @pytest.fixture(scope="class")
    def cs(self):
        return CrossSectionBuilder().build("tokyo", "marunouchi", SampleCityDataSource())

    def test_cross_section_is_svg(self, cs):
        svg = CrossSectionSvgRenderer().render(cs)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")

    def test_contains_layers_rects(self, cs):
        svg = CrossSectionSvgRenderer().render(cs)
        assert "<rect" in svg

    def test_contains_station_circles(self, cs):
        svg = CrossSectionSvgRenderer().render(cs)
        assert "<circle" in svg

    def test_contains_line_polyline(self, cs):
        svg = CrossSectionSvgRenderer().render(cs)
        assert "<polyline" in svg

    def test_contains_legend(self, cs):
        svg = CrossSectionSvgRenderer().render(cs)
        for layer in cs.layers:
            assert layer.name in svg

    def test_empty_cross_section_ok(self):
        empty = CrossSection("x", "y", "空路線", [], [], 0.0, 0.0, "sample")
        svg = CrossSectionSvgRenderer().render(empty)
        assert svg.startswith("<svg")

    def test_custom_style(self, cs):
        style = SvgStyle(width=1200, height=600)
        svg = CrossSectionSvgRenderer(style).render(cs)
        assert 'width="1200"' in svg

    def test_front_view_svg(self):
        line = SampleCityDataSource().load_lines("tokyo")[0]
        svg = FrontViewSvgRenderer().render(line)
        assert svg.startswith("<svg")
        assert "<circle" in svg

    def test_escaping(self):
        cs = CrossSection("x", "y", "A<b>&\"", [], [], 0.0, 0.0, "sample")
        svg = CrossSectionSvgRenderer().render(cs)
        assert "<b>" not in svg  # エスケープされている
        assert "&lt;b&gt;" in svg


# ===========================================================================
# Section G: MapRenderer / PNG フォールバック
# ===========================================================================

class TestMapRenderer:
    @pytest.fixture(scope="class")
    def cs(self):
        return CrossSectionBuilder().build("tokyo", "marunouchi", SampleCityDataSource())

    def test_cross_section_svg(self, cs):
        svg = MapRenderer().cross_section_svg(cs)
        assert svg.startswith("<svg")

    def test_front_view_svg(self):
        line = SampleCityDataSource().load_lines("osaka")[0]
        svg = MapRenderer().front_view_svg(line)
        assert svg.startswith("<svg")

    def test_to_png_returns_bytes_or_none(self, cs):
        svg = MapRenderer().cross_section_svg(cs)
        png = MapRenderer().to_png(svg)
        # cairosvg 不在環境では None。在れば bytes。どちらも許容
        assert png is None or isinstance(png, bytes)

    def test_to_png_invalid_svg_none(self):
        # 不正な SVG でも例外を投げず None
        png = MapRenderer().to_png("not-svg")
        assert png is None or isinstance(png, bytes)


# ===========================================================================
# Section H: CityMapStore / CityMapFactory
# ===========================================================================

class TestStoreFactory:
    def test_store_save_get(self):
        store = CityMapStore()
        cs = CrossSectionBuilder().build("tokyo", "marunouchi", SampleCityDataSource())
        store.save(cs)
        assert store.get("tokyo", "marunouchi") is cs
        assert store.get("tokyo", "none") is None
        assert store.count() == 1

    def test_store_list_by_city(self):
        store = CityMapStore()
        src = SampleCityDataSource()
        store.save(CrossSectionBuilder().build("tokyo", "marunouchi", src))
        store.save(CrossSectionBuilder().build("tokyo", "ginza", src))
        store.save(CrossSectionBuilder().build("osaka", "midosuji", src))
        assert len(store.list_by_city("tokyo")) == 2
        assert len(store.list_by_city("osaka")) == 1

    def test_store_clear(self):
        store = CityMapStore()
        store.save(CrossSectionBuilder().build("tokyo", "ginza", SampleCityDataSource()))
        store.clear()
        assert store.count() == 0

    def test_factory_from_sample(self):
        assert isinstance(CityMapFactory.from_sample(), SampleCityDataSource)

    def test_factory_from_gtfs(self):
        src = CityMapFactory.from_gtfs("tokyo", "http://x/gtfs.zip")
        assert isinstance(src, GTFSCityDataSource)

    def test_factory_available_cities(self):
        cities = CityMapFactory.available_cities()
        assert len(cities) == 13
        assert all("city_id" in c and "line_count" in c for c in cities)


# ===========================================================================
# Section I: API エンドポイント
# ===========================================================================

@pytest.fixture(scope="module")
def client():
    import serve.api as api_module
    api_module.state.model = MagicMock()
    api_module.state.tokenizer = MagicMock()
    from fastapi.testclient import TestClient
    return TestClient(api_module.app, raise_server_exceptions=False)


class TestMapAPI:
    def test_cities(self, client):
        r = client.get("/v1/map/cities")
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 13
        assert len(body["cities"]) == 13

    def test_lines(self, client):
        r = client.get("/v1/map/tokyo/lines")
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_lines_unknown_city_404(self, client):
        r = client.get("/v1/map/atlantis/lines")
        assert r.status_code == 404

    def test_cross_section_svg(self, client):
        r = client.get("/v1/map/tokyo/marunouchi/cross-section")
        assert r.status_code == 200
        body = r.json()
        assert body["svg"] and body["svg"].startswith("<svg")
        assert body["cross_section"]["line_name"] == "丸ノ内線"

    def test_cross_section_json(self, client):
        r = client.get("/v1/map/tokyo/marunouchi/cross-section?format=json")
        assert r.status_code == 200
        body = r.json()
        assert body["svg"] is None
        assert body["cross_section"]["max_depth_m"] > 0

    def test_cross_section_png_fallback(self, client):
        # cairosvg 不在環境では SVG にフォールバック (image/svg+xml)、在れば image/png
        r = client.get("/v1/map/tokyo/marunouchi/cross-section?format=png")
        assert r.status_code == 200
        assert r.headers["content-type"] in ("image/png", "image/svg+xml")

    def test_cross_section_unknown_line_404(self, client):
        r = client.get("/v1/map/tokyo/nonexistent/cross-section")
        assert r.status_code == 404

    def test_front_view(self, client):
        r = client.get("/v1/map/tokyo/marunouchi/front-view")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/svg+xml")
        assert r.text.startswith("<svg")

    def test_front_view_404(self, client):
        r = client.get("/v1/map/tokyo/nope/front-view")
        assert r.status_code == 404

    def test_post_cross_section_sample(self, client):
        r = client.post("/v1/map/cross-section", json={"city": "osaka", "line": "midosuji"})
        assert r.status_code == 200
        assert r.json()["cross_section"]["generated_by"] == "sample"

    def test_post_cross_section_gtfs_no_url_falls_back(self, client):
        r = client.post("/v1/map/cross-section",
                        json={"city": "tokyo", "line": "marunouchi", "source": "gtfs"})
        assert r.status_code == 200
        assert r.json()["cross_section"]["generated_by"] == "sample(fallback)"

    def test_post_cross_section_unknown_404(self, client):
        r = client.post("/v1/map/cross-section", json={"city": "atlantis", "line": "x"})
        assert r.status_code == 404

    def test_post_cross_section_json_format(self, client):
        r = client.post("/v1/map/cross-section",
                        json={"city": "tokyo", "line": "ginza", "format": "json"})
        assert r.status_code == 200
        assert r.json()["svg"] is None

    def test_map_ui(self, client):
        r = client.get("/map")
        assert r.status_code == 200
        assert "地下断面" in r.text
