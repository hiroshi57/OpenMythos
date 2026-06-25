"""
Sprint 78 テスト — 都市マップビジュアライゼーション
"""
import pytest

from open_mythos.skills.city_map_viz import (
    CityMapBuilder, CityMapStore, CityMapData, DistrictData,
    MapLayer, generate_html, TOKYO_DISTRICTS,
    _district_color, _legend_html,
)


def _make_builder():
    store = CityMapStore()
    builder = CityMapBuilder(store=store)
    return builder, store


def _sample_districts():
    return [
        DistrictData("新宿", x=5, y=5, traffic_level="congested", noise_status="violation",
                     crowd_level="packed", energy_status="high", disaster_level="warning"),
        DistrictData("渋谷", x=25, y=5, traffic_level="clear", noise_status="compliant",
                     crowd_level="sparse", energy_status="normal", disaster_level=None),
        DistrictData("品川", x=45, y=5, traffic_level="moderate", noise_status="near_limit",
                     crowd_level="normal", energy_status="normal", disaster_level=None),
    ]


class TestDistrictData:
    def test_to_dict_keys(self):
        d = DistrictData("新宿", x=5, y=5, traffic_level="congested")
        dd = d.to_dict()
        for key in ["name", "x", "y", "width", "height", "traffic_level",
                    "noise_status", "crowd_level", "energy_status", "disaster_level"]:
            assert key in dd

    def test_defaults(self):
        d = DistrictData("駅前", x=10, y=10)
        assert d.traffic_level is None
        assert d.width == 18.0
        assert d.height == 14.0


class TestCityMapData:
    def test_to_dict(self):
        districts = _sample_districts()
        data = CityMapData(city="Tokyo", districts=districts)
        dd = data.to_dict()
        assert dd["city"] == "Tokyo"
        assert len(dd["districts"]) == 3
        assert "active_layer" in dd

    def test_default_layer(self):
        data = CityMapData(city="Tokyo", districts=[])
        assert data.active_layer == MapLayer.TRAFFIC


class TestDistrictColor:
    def test_traffic_congested(self):
        d = DistrictData("A", x=0, y=0, traffic_level="congested")
        assert _district_color(d, MapLayer.TRAFFIC) == "#ea8600"   # Google orange

    def test_traffic_clear(self):
        d = DistrictData("A", x=0, y=0, traffic_level="clear")
        assert _district_color(d, MapLayer.TRAFFIC) == "#34a853"   # Google green

    def test_noise_violation(self):
        d = DistrictData("A", x=0, y=0, noise_status="violation")
        assert _district_color(d, MapLayer.NOISE) == "#d93025"     # Google red

    def test_noise_compliant(self):
        d = DistrictData("A", x=0, y=0, noise_status="compliant")
        assert _district_color(d, MapLayer.NOISE) == "#34a853"

    def test_crowd_packed(self):
        d = DistrictData("A", x=0, y=0, crowd_level="packed")
        assert _district_color(d, MapLayer.CROWD) == "#d93025"

    def test_crowd_sparse(self):
        d = DistrictData("A", x=0, y=0, crowd_level="sparse")
        assert _district_color(d, MapLayer.CROWD) == "#e8f0fe"     # Google light blue

    def test_energy_critical(self):
        d = DistrictData("A", x=0, y=0, energy_status="critical")
        assert _district_color(d, MapLayer.ENERGY) == "#d93025"

    def test_disaster_critical(self):
        d = DistrictData("A", x=0, y=0, disaster_level="critical")
        assert _district_color(d, MapLayer.DISASTER) == "#d93025"

    def test_disaster_none(self):
        d = DistrictData("A", x=0, y=0, disaster_level=None)
        color = _district_color(d, MapLayer.DISASTER)
        assert color == "#e8f5e9"

    def test_missing_status_fallback(self):
        d = DistrictData("A", x=0, y=0)
        color = _district_color(d, MapLayer.TRAFFIC)
        assert color == "#e8e0d0"   # map land neutral color


class TestLegendHtml:
    def test_traffic_legend(self):
        html = _legend_html(MapLayer.TRAFFIC)
        assert "通常走行" in html
        assert "渋滞" in html

    def test_noise_legend(self):
        html = _legend_html(MapLayer.NOISE)
        assert "規制内" in html
        assert "規制超過" in html

    def test_crowd_legend(self):
        html = _legend_html(MapLayer.CROWD)
        assert "閑散" in html
        assert "超混雑" in html

    def test_disaster_legend(self):
        html = _legend_html(MapLayer.DISASTER)
        assert "緊急" in html

    def test_energy_legend(self):
        html = _legend_html(MapLayer.ENERGY)
        assert "通常" in html


class TestGenerateHtml:
    def test_returns_string(self):
        data = CityMapData("Tokyo", _sample_districts(), MapLayer.TRAFFIC)
        html = generate_html(data)
        assert isinstance(html, str)

    def test_contains_city_name(self):
        data = CityMapData("Tokyo", _sample_districts(), MapLayer.TRAFFIC)
        html = generate_html(data)
        assert "Tokyo" in html

    def test_contains_district_names(self):
        data = CityMapData("Tokyo", _sample_districts(), MapLayer.TRAFFIC)
        html = generate_html(data)
        assert "新宿" in html
        assert "渋谷" in html

    def test_contains_svg(self):
        data = CityMapData("Tokyo", _sample_districts(), MapLayer.TRAFFIC)
        html = generate_html(data)
        assert "<svg" in html

    def test_layer_traffic_html(self):
        data = CityMapData("Tokyo", _sample_districts(), MapLayer.TRAFFIC)
        html = generate_html(data)
        assert "交通量" in html

    def test_layer_noise_html(self):
        data = CityMapData("Tokyo", _sample_districts(), MapLayer.NOISE)
        html = generate_html(data)
        assert "騒音" in html

    def test_layer_disaster_html(self):
        data = CityMapData("Tokyo", _sample_districts(), MapLayer.DISASTER)
        html = generate_html(data)
        assert "災害" in html

    def test_empty_districts(self):
        data = CityMapData("Empty", [], MapLayer.TRAFFIC)
        html = generate_html(data)
        assert isinstance(html, str)
        assert "Empty" in html


class TestCityMapStore:
    def test_set_and_get(self):
        _, store = _make_builder()
        data = CityMapData("Tokyo", _sample_districts())
        store.set("Tokyo", data)
        assert store.get("Tokyo") is not None

    def test_get_nonexistent(self):
        _, store = _make_builder()
        assert store.get("NoCity") is None

    def test_list_cities(self):
        _, store = _make_builder()
        store.set("Tokyo", CityMapData("Tokyo", []))
        store.set("Osaka", CityMapData("Osaka", []))
        cities = store.list_cities()
        assert "Tokyo" in cities
        assert "Osaka" in cities

    def test_count(self):
        _, store = _make_builder()
        store.set("Tokyo", CityMapData("Tokyo", []))
        store.set("Osaka", CityMapData("Osaka", []))
        assert store.count() == 2


class TestCityMapBuilder:
    def test_build_stores_data(self):
        builder, store = _make_builder()
        builder.build("Tokyo", _sample_districts())
        assert store.get("Tokyo") is not None

    def test_build_district_count(self):
        builder, _ = _make_builder()
        data = builder.build("Tokyo", _sample_districts())
        assert len(data.districts) == 3

    def test_get_html_returns_string(self):
        builder, _ = _make_builder()
        builder.build("Tokyo", _sample_districts())
        html = builder.get_html("Tokyo")
        assert isinstance(html, str)
        assert "<html" in html

    def test_get_html_with_layer(self):
        builder, _ = _make_builder()
        builder.build("Tokyo", _sample_districts())
        html = builder.get_html("Tokyo", MapLayer.NOISE)
        assert "騒音" in html

    def test_get_html_nonexistent_city(self):
        builder, _ = _make_builder()
        assert builder.get_html("NoCity") is None

    def test_build_overwrites(self):
        builder, store = _make_builder()
        builder.build("Tokyo", _sample_districts())
        builder.build("Tokyo", [DistrictData("只今", x=0, y=0)])
        assert len(store.get("Tokyo").districts) == 1


class TestTokyoPreset:
    def test_preset_count(self):
        assert len(TOKYO_DISTRICTS) == 20

    def test_preset_has_shinjuku(self):
        names = [d.name for d in TOKYO_DISTRICTS]
        assert "新宿" in names

    def test_preset_all_have_coords(self):
        for d in TOKYO_DISTRICTS:
            assert 0 <= d.x <= 100
            assert 0 <= d.y <= 100

    def test_preset_generate_html(self):
        builder, _ = _make_builder()
        data = builder.build("Tokyo", TOKYO_DISTRICTS, active_layer=MapLayer.TRAFFIC)
        html = generate_html(data)
        assert "新宿" in html
        assert "渋谷" in html
        assert len(html) > 1000
