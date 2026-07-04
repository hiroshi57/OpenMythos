"""
Sprint 72 — 地図拡張 (比較・編集・レポート) テスト

対象:
  72A: open_mythos/skills/map_comparator.py
  72B: open_mythos/skills/map_editor.py
  72C: open_mythos/skills/map_report.py
  serve/api.py (/v1/map/compare/*, /v1/map-editor/*, /v1/map/*/report/*)
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from open_mythos.skills.city_map import (
    CityName, LineType, GeologyLayerType,
    GeoCoord, Station, MetroLine, GeologyLayer,
    CityMapStore, CityMapDataset,
)
from open_mythos.skills.map_comparator import (
    ComparisonConfig, DepthStats, ComparisonResult, MapComparator,
)
from open_mythos.skills.map_editor import (
    EditAction, EditRecord, EditResult, MapEditor,
)
from open_mythos.skills.map_report import (
    ReportSection, CityMapReport, MultiCityReport, MapReportEngine,
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


def _make_line(lid="l1", city=CityName.TOKYO):
    return MetroLine(
        id=lid, name="テスト線", name_en="Test Line",
        city=city, line_type=LineType.SUBWAY,
        color="#FF0000", station_ids=[], total_length_km=10.0, opened_year=2000,
    )


def _make_geology(gid="g1", city=CityName.TOKYO, d_from=0.0, d_to=10.0,
                  layer_type=GeologyLayerType.ALLUVIUM):
    return GeologyLayer(
        id=gid, city=city, layer_type=layer_type,
        name="テスト層", depth_from_m=d_from, depth_to_m=d_to,
        color="#90EE90", n_value=5.0,
    )


def _fresh_store():
    """各テスト用の独立ストア (小規模)"""
    store = CityMapStore()
    store.lines.add(_make_line("l1", CityName.TOKYO))
    store.lines.add(_make_line("l2", CityName.OSAKA))
    store.stations.add(_make_station("s1", CityName.TOKYO, depth=15.0))
    store.stations.add(_make_station("s2", CityName.TOKYO, depth=25.0))
    store.stations.add(_make_station("s3", CityName.OSAKA, depth=20.0))
    store.geology.add(_make_geology("g1", CityName.TOKYO, 0.0, 10.0))
    store.geology.add(_make_geology("g2", CityName.OSAKA, 0.0, 15.0))
    return store


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 72A: map_comparator.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── ComparisonConfig ─────────────────────────────────────────────

def test_comparison_config_defaults():
    cfg = ComparisonConfig()
    assert cfg.width == 1400
    assert cfg.height == 600
    assert cfg.max_depth_m == 60.0


def test_comparison_config_panel_width():
    cfg = ComparisonConfig(width=1400, margin=40)
    assert cfg.panel_width > 0
    assert cfg.panel_width * 2 + cfg.margin * 3 <= cfg.width


def test_comparison_config_depth_to_y():
    cfg = ComparisonConfig()
    y0 = cfg.depth_to_y(0.0, 100)
    y30 = cfg.depth_to_y(30.0, 100)
    y60 = cfg.depth_to_y(60.0, 100)
    assert y0 == 100
    assert y30 > y0
    assert y60 > y30


def test_comparison_config_depth_clamped():
    cfg = ComparisonConfig(max_depth_m=60.0)
    y_over = cfg.depth_to_y(100.0, 0)
    y_max = cfg.depth_to_y(60.0, 0)
    assert y_over == y_max  # クランプ


# ─── DepthStats ───────────────────────────────────────────────────

def test_depth_stats_to_dict():
    stats = DepthStats("tokyo", 10.0, 25.0, 17.5, 4)
    d = stats.to_dict()
    assert d["city"] == "tokyo"
    assert d["min_depth_m"] == 10.0
    assert d["max_depth_m"] == 25.0
    assert d["avg_depth_m"] == 17.5
    assert d["station_count"] == 4


# ─── ComparisonResult ─────────────────────────────────────────────

def test_comparison_result_to_dict():
    stats = DepthStats("tokyo", 10.0, 25.0, 17.5, 2)
    result = ComparisonResult(
        city_a="tokyo", city_b="osaka",
        svg="<svg/>",
        stats_a=stats, stats_b=stats,
        deeper_city="tokyo",
        geology_diff=[],
    )
    d = result.to_dict()
    assert d["city_a"] == "tokyo"
    assert d["city_b"] == "osaka"
    assert d["deeper_city"] == "tokyo"
    assert "svg" in d
    assert "stats_a" in d
    assert "geology_diff" in d


# ─── MapComparator ────────────────────────────────────────────────

def test_comparator_compare_returns_result(dataset_store):
    comp = MapComparator(dataset_store)
    result = comp.compare(CityName.TOKYO, CityName.OSAKA)
    assert isinstance(result, ComparisonResult)
    assert result.city_a == "tokyo"
    assert result.city_b == "osaka"


def test_comparator_svg_generated(dataset_store):
    comp = MapComparator(dataset_store)
    result = comp.compare(CityName.TOKYO, CityName.OSAKA)
    assert "<svg" in result.svg
    assert "</svg>" in result.svg


def test_comparator_stats_populated(dataset_store):
    comp = MapComparator(dataset_store)
    result = comp.compare(CityName.TOKYO, CityName.OSAKA)
    assert result.stats_a.station_count > 0
    assert result.stats_b.station_count > 0
    assert result.stats_a.max_depth_m > 0


def test_comparator_deeper_city(dataset_store):
    comp = MapComparator(dataset_store)
    result = comp.compare(CityName.TOKYO, CityName.FUKUOKA)
    assert result.deeper_city in {"tokyo", "fukuoka"}


def test_comparator_geology_diff(dataset_store):
    comp = MapComparator(dataset_store)
    result = comp.compare(CityName.TOKYO, CityName.OSAKA)
    assert isinstance(result.geology_diff, list)
    assert len(result.geology_diff) > 0
    assert "layer_type" in result.geology_diff[0]


def test_comparator_geology_diff_has_both_cities(dataset_store):
    comp = MapComparator(dataset_store)
    result = comp.compare(CityName.TOKYO, CityName.OSAKA)
    diff = result.geology_diff[0]
    assert "tokyo" in diff
    assert "osaka" in diff


def test_comparator_same_city(dataset_store):
    comp = MapComparator(dataset_store)
    result = comp.compare(CityName.TOKYO, CityName.TOKYO)
    assert result.city_a == result.city_b == "tokyo"
    assert result.deeper_city == "tokyo"


def test_comparator_small_store():
    store = _fresh_store()
    comp = MapComparator(store)
    result = comp.compare(CityName.TOKYO, CityName.OSAKA)
    assert result.stats_a.station_count == 2
    assert result.stats_b.station_count == 1


def test_comparator_svg_contains_city_names(dataset_store):
    comp = MapComparator(dataset_store)
    result = comp.compare(CityName.TOKYO, CityName.NAGOYA)
    assert "TOKYO" in result.svg
    assert "NAGOYA" in result.svg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 72B: map_editor.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── EditResult ───────────────────────────────────────────────────

def test_edit_result_to_dict():
    r = EditResult(True, EditAction.ADD_LINE, "l1", "Line added.")
    d = r.to_dict()
    assert d["success"] is True
    assert d["action"] == "add_line"
    assert d["target_id"] == "l1"


# ─── EditRecord ───────────────────────────────────────────────────

def test_edit_record_to_dict():
    rec = EditRecord(EditAction.ADD_STATION, "s1", None, {"id": "s1"})
    d = rec.to_dict()
    assert d["action"] == "add_station"
    assert d["snapshot_before"] is None
    assert d["snapshot_after"] == {"id": "s1"}


# ─── MapEditor — Line CRUD ────────────────────────────────────────

def test_editor_add_line():
    store = _fresh_store()
    editor = MapEditor(store)
    new_line = _make_line("new-line", CityName.NAGOYA)
    result = editor.add_line(new_line)
    assert result.success is True
    assert result.action == EditAction.ADD_LINE
    assert store.lines.get("new-line") is not None


def test_editor_add_line_duplicate():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.add_line(_make_line("l1"))  # already exists
    assert result.success is False


def test_editor_update_line():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.update_line("l1", name="更新線", color="#00FF00")
    assert result.success is True
    assert store.lines.get("l1").name == "更新線"
    assert store.lines.get("l1").color == "#00FF00"


def test_editor_update_line_not_found():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.update_line("no-such-line", name="X")
    assert result.success is False


def test_editor_delete_line():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.delete_line("l1")
    assert result.success is True
    assert store.lines.get("l1") is None


def test_editor_delete_line_not_found():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.delete_line("ghost")
    assert result.success is False


# ─── MapEditor — Station CRUD ────────────────────────────────────

def test_editor_add_station():
    store = _fresh_store()
    editor = MapEditor(store)
    new_st = _make_station("new-st", depth=30.0)
    result = editor.add_station(new_st)
    assert result.success is True
    assert store.stations.get("new-st") is not None


def test_editor_add_station_duplicate():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.add_station(_make_station("s1"))  # duplicate
    assert result.success is False


def test_editor_update_station():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.update_station("s1", name="新駅名", depth_m=99.0)
    assert result.success is True
    assert store.stations.get("s1").depth_m == 99.0


def test_editor_update_station_not_found():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.update_station("ghost", name="X")
    assert result.success is False


def test_editor_delete_station():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.delete_station("s1")
    assert result.success is True
    assert store.stations.get("s1") is None


def test_editor_delete_station_not_found():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.delete_station("ghost")
    assert result.success is False


# ─── MapEditor — Geology CRUD ────────────────────────────────────

def test_editor_add_geology():
    store = _fresh_store()
    editor = MapEditor(store)
    new_gl = _make_geology("new-gl", d_from=50.0, d_to=70.0)
    result = editor.add_geology(new_gl)
    assert result.success is True
    assert store.geology.get("new-gl") is not None


def test_editor_add_geology_duplicate():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.add_geology(_make_geology("g1"))  # duplicate
    assert result.success is False


def test_editor_update_geology():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.update_geology("g1", depth_to_m=20.0, n_value=8.0)
    assert result.success is True
    assert store.geology.get("g1").depth_to_m == 20.0
    assert store.geology.get("g1").n_value == 8.0


def test_editor_update_geology_not_found():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.update_geology("ghost", name="X")
    assert result.success is False


def test_editor_delete_geology():
    store = _fresh_store()
    editor = MapEditor(store)
    result = editor.delete_geology("g1")
    assert result.success is True
    assert store.geology.get("g1") is None


# ─── MapEditor — history / summary ───────────────────────────────

def test_editor_history_tracks_changes():
    store = _fresh_store()
    editor = MapEditor(store)
    editor.add_line(_make_line("lx"))
    editor.update_station("s1", depth_m=50.0)
    editor.delete_geology("g1")
    assert len(editor.history) == 3


def test_editor_history_dicts():
    store = _fresh_store()
    editor = MapEditor(store)
    editor.add_line(_make_line("lx"))
    h = editor.history_dicts()
    assert isinstance(h, list)
    assert h[0]["action"] == "add_line"


def test_editor_history_failed_not_recorded():
    store = _fresh_store()
    editor = MapEditor(store)
    editor.delete_line("no-exist")  # 失敗
    assert len(editor.history) == 0


def test_editor_summary():
    store = _fresh_store()
    editor = MapEditor(store)
    s = editor.summary()
    assert "lines" in s
    assert "stations" in s
    assert "geology_layers" in s
    assert "history_count" in s
    assert s["history_count"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 72C: map_report.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── ReportSection ────────────────────────────────────────────────

def test_report_section_to_markdown():
    sec = ReportSection("テスト", "内容です")
    md = sec.to_markdown()
    assert md.startswith("## テスト")
    assert "内容です" in md


# ─── CityMapReport ────────────────────────────────────────────────

def test_city_map_report_to_markdown():
    report = CityMapReport(
        city="tokyo",
        title="東京レポート",
        sections=[ReportSection("概要", "東京の地下鉄")],
    )
    md = report.to_markdown()
    assert "# 東京レポート" in md
    assert "## 概要" in md
    assert "東京の地下鉄" in md


def test_city_map_report_to_dict():
    report = CityMapReport(
        city="tokyo",
        title="東京レポート",
        sections=[ReportSection("概要", "内容")],
    )
    d = report.to_dict()
    assert d["city"] == "tokyo"
    assert d["title"] == "東京レポート"
    assert "markdown" in d
    assert len(d["sections"]) == 1


# ─── MultiCityReport ──────────────────────────────────────────────

def test_multi_city_report_to_markdown():
    report = MultiCityReport(
        title="比較レポート",
        cities=["tokyo", "osaka"],
        sections=[ReportSection("比較", "内容")],
    )
    md = report.to_markdown()
    assert "# 比較レポート" in md
    assert "## 比較" in md


def test_multi_city_report_to_dict():
    report = MultiCityReport(
        title="比較", cities=["tokyo", "osaka"],
        sections=[ReportSection("x", "y")],
    )
    d = report.to_dict()
    assert d["cities"] == ["tokyo", "osaka"]
    assert "markdown" in d


# ─── MapReportEngine ──────────────────────────────────────────────

def test_report_engine_city_report(dataset_store):
    engine = MapReportEngine(dataset_store)
    report = engine.generate_city_report(CityName.TOKYO)
    assert isinstance(report, CityMapReport)
    assert report.city == "tokyo"
    assert len(report.sections) >= 4


def test_report_engine_city_report_has_geology(dataset_store):
    engine = MapReportEngine(dataset_store)
    report = engine.generate_city_report(CityName.TOKYO)
    titles = [s.title for s in report.sections]
    assert any("地質" in t for t in titles)


def test_report_engine_city_report_has_stations(dataset_store):
    engine = MapReportEngine(dataset_store)
    report = engine.generate_city_report(CityName.OSAKA)
    md = report.to_markdown()
    assert "Namba" in md or "Shinsaibashi" in md


def test_report_engine_city_report_markdown(dataset_store):
    engine = MapReportEngine(dataset_store)
    report = engine.generate_city_report(CityName.NAGOYA)
    md = report.to_markdown()
    assert "# " in md
    assert "## " in md


def test_report_engine_depth_stats(dataset_store):
    engine = MapReportEngine(dataset_store)
    report = engine.generate_city_report(CityName.TOKYO)
    md = report.to_markdown()
    assert "最深" in md
    assert "平均" in md


def test_report_engine_multi_city(dataset_store):
    engine = MapReportEngine(dataset_store)
    cities = [CityName.TOKYO, CityName.OSAKA, CityName.FUKUOKA]
    report = engine.generate_multi_city_report(cities)
    assert isinstance(report, MultiCityReport)
    assert len(report.cities) == 3


def test_report_engine_multi_city_markdown(dataset_store):
    engine = MapReportEngine(dataset_store)
    report = engine.generate_multi_city_report(list(CityName))
    md = report.to_markdown()
    assert "TOKYO" in md
    assert "OSAKA" in md


def test_report_engine_all_cities(dataset_store):
    engine = MapReportEngine(dataset_store)
    for city in CityName:
        report = engine.generate_city_report(city)
        assert report.city == city.value


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 72 API エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── 72A: /v1/map/compare/* ──────────────────────────────────────

def test_api_compare_tokyo_osaka(client):
    resp = client.get("/v1/map/compare/tokyo/osaka")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city_a"] == "tokyo"
    assert data["city_b"] == "osaka"
    assert "<svg" in data["svg"]


def test_api_compare_stats(client):
    resp = client.get("/v1/map/compare/tokyo/fukuoka/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "city_a" in data
    assert "city_b" in data
    assert "deeper_city" in data
    assert "geology_diff" in data


def test_api_compare_geology_diff(client):
    resp = client.get("/v1/map/compare/tokyo/osaka/stats")
    data = resp.json()
    assert len(data["geology_diff"]) > 0
    assert "layer_type" in data["geology_diff"][0]


def test_api_compare_invalid_city(client):
    resp = client.get("/v1/map/compare/invalid/osaka")
    assert resp.status_code == 404


def test_api_compare_invalid_city_b(client):
    resp = client.get("/v1/map/compare/tokyo/invalid")
    assert resp.status_code == 404


# ─── 72B: /v1/map-editor/* ───────────────────────────────────────

def test_api_editor_summary(client):
    resp = client.get("/v1/map-editor/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "lines" in data
    assert "stations" in data
    assert "history_count" in data


def test_api_editor_add_line(client):
    resp = client.post("/v1/map-editor/lines", json={
        "id": "test-new-line-72", "name": "テスト72線",
        "name_en": "Test72 Line", "city": "nagoya",
        "line_type": "subway", "color": "#123456",
        "total_length_km": 5.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["action"] == "add_line"


def test_api_editor_add_line_duplicate(client):
    # 2回目は失敗
    client.post("/v1/map-editor/lines", json={
        "id": "dup-line-72", "name": "X", "name_en": "X",
        "city": "tokyo", "line_type": "subway", "color": "#000",
    })
    resp = client.post("/v1/map-editor/lines", json={
        "id": "dup-line-72", "name": "X", "name_en": "X",
        "city": "tokyo", "line_type": "subway", "color": "#000",
    })
    data = resp.json()
    assert data["success"] is False


def test_api_editor_update_line(client):
    # まず追加
    client.post("/v1/map-editor/lines", json={
        "id": "upd-line-72", "name": "更新前", "name_en": "Before",
        "city": "tokyo", "line_type": "subway", "color": "#AAA",
    })
    resp = client.patch("/v1/map-editor/lines/upd-line-72", json={"color": "#FF9900"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True


def test_api_editor_delete_line(client):
    client.post("/v1/map-editor/lines", json={
        "id": "del-line-72", "name": "削除線", "name_en": "DelLine",
        "city": "osaka", "line_type": "subway", "color": "#BBB",
    })
    resp = client.delete("/v1/map-editor/lines/del-line-72")
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_api_editor_add_station(client):
    resp = client.post("/v1/map-editor/stations", json={
        "id": "test-st-72", "name": "テスト72駅",
        "name_en": "Test72 St", "line_id": "tokyo-marunouchi",
        "city": "tokyo", "lat": 35.68, "lon": 139.76,
        "depth_m": 18.5, "platform_count": 2,
    })
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_api_editor_history(client):
    resp = client.get("/v1/map-editor/history")
    assert resp.status_code == 200
    data = resp.json()
    assert "history" in data
    assert "count" in data
    assert data["count"] >= 0


# ─── 72C: /v1/map/*/report/md ────────────────────────────────────

def test_api_report_md_tokyo(client):
    resp = client.get("/v1/map/tokyo/report/md")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert "markdown" in data
    assert "# " in data["markdown"]


def test_api_report_md_sections(client):
    resp = client.get("/v1/map/osaka/report/md")
    data = resp.json()
    assert len(data["sections"]) >= 4


def test_api_report_md_contains_geology(client):
    resp = client.get("/v1/map/nagoya/report/md")
    md = resp.json()["markdown"]
    assert "地質" in md


def test_api_report_md_invalid_city(client):
    resp = client.get("/v1/map/nowhere/report/md")
    assert resp.status_code == 404


def test_api_report_compare_all(client):
    resp = client.get("/v1/map/report/compare")
    assert resp.status_code == 200
    data = resp.json()
    assert "markdown" in data
    assert len(data["cities"]) == 5


def test_api_report_compare_subset(client):
    resp = client.get("/v1/map/report/compare?cities=tokyo,osaka")
    assert resp.status_code == 200
    data = resp.json()
    assert data["cities"] == ["tokyo", "osaka"]


def test_api_report_compare_markdown_has_table(client):
    resp = client.get("/v1/map/report/compare")
    md = resp.json()["markdown"]
    assert "|" in md  # Markdown テーブル


def test_api_report_compare_invalid_city(client):
    resp = client.get("/v1/map/report/compare?cities=tokyo,badcity")
    assert resp.status_code == 422
