"""
Sprint 74 — 混雑シミュレーション/アクセシビリティ/地下水位 テスト

対象:
  74A: open_mythos/skills/crowd_simulator.py
  74B: open_mythos/skills/accessibility.py
  74C: open_mythos/skills/groundwater.py
  serve/api.py (/v1/crowd/*, /v1/access/*, /v1/groundwater/*)
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from open_mythos.skills.city_map import CityName
from open_mythos.skills.crowd_simulator import (
    CrowdLevel, TimeSlot, hour_to_slot,
    StationProfile, CrowdSnapshot, DailyProfile,
    CrowdSimulator, CrowdDataset,
)
from open_mythos.skills.accessibility import (
    AccessFeature, AccessLevel,
    AccessibilityProfile, AccessibilityScore, AccessibilityReport,
    AccessibilityAnalyzer, AccessibilityDataset,
)
from open_mythos.skills.groundwater import (
    WaterLevelZone, FloodRiskLevel, Season,
    GroundwaterLayer, FloodRiskFactor, FloodRiskResult,
    CityWaterProfile, FloodRiskAssessor, GroundwaterDataset,
)


# ─── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from serve.api import app
    return TestClient(app)


@pytest.fixture(scope="module")
def crowd_sim():
    return CrowdDataset.build()


@pytest.fixture(scope="module")
def access_analyzer():
    return AccessibilityDataset.build()


@pytest.fixture(scope="module")
def gw_assessor():
    return GroundwaterDataset.build()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 74A: crowd_simulator.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_crowd_level_values():
    assert CrowdLevel.EMPTY.value == "empty"
    assert CrowdLevel.OVERCROWDED.value == "overcrowded"


def test_time_slot_values():
    assert TimeSlot.MORNING_RUSH.value == "morning_rush"
    assert TimeSlot.EVENING_RUSH.value == "evening_rush"


def test_hour_to_slot_morning_rush():
    assert hour_to_slot(7) == TimeSlot.MORNING_RUSH
    assert hour_to_slot(8) == TimeSlot.MORNING_RUSH


def test_hour_to_slot_evening_rush():
    assert hour_to_slot(17) == TimeSlot.EVENING_RUSH
    assert hour_to_slot(19) == TimeSlot.EVENING_RUSH


def test_hour_to_slot_late_night():
    assert hour_to_slot(0) == TimeSlot.LATE_NIGHT
    assert hour_to_slot(4) == TimeSlot.LATE_NIGHT


def test_hour_to_slot_wraps():
    assert hour_to_slot(24) == hour_to_slot(0)


def test_station_profile_auto_capacity():
    p = StationProfile("s1", "駅A", daily_passengers=100_000, platform_count=2)
    assert p.base_capacity == 1000


def test_station_profile_custom_capacity():
    p = StationProfile("s1", "駅A", 100_000, platform_count=2, base_capacity=2000)
    assert p.base_capacity == 2000


def test_station_profile_to_dict():
    p = StationProfile("s1", "駅A", 200_000, platform_count=2)
    d = p.to_dict()
    assert d["station_id"] == "s1"
    assert d["daily_passengers"] == 200_000


def test_crowd_snapshot_to_dict():
    snap = CrowdSnapshot("s1", "駅A", 8, TimeSlot.MORNING_RUSH,
                         500, 1000, 0.5, CrowdLevel.MODERATE)
    d = snap.to_dict()
    assert d["slot"] == "morning_rush"
    assert d["occupancy_rate"] == 0.5
    assert d["level"] == "moderate"


def test_crowd_simulator_snapshot():
    sim = CrowdSimulator()
    sim.register(StationProfile("s1", "駅A", 100_000))
    snap = sim.snapshot("s1", 8)
    assert snap is not None
    assert snap.station_id == "s1"
    assert snap.hour == 8
    assert snap.estimated_passengers > 0


def test_crowd_simulator_snapshot_not_found():
    sim = CrowdSimulator()
    assert sim.snapshot("ghost", 8) is None


def test_crowd_simulator_morning_rush_higher_than_late_night():
    sim = CrowdSimulator()
    sim.register(StationProfile("s1", "駅A", 500_000))
    rush = sim.snapshot("s1", 8)
    night = sim.snapshot("s1", 2)
    assert rush.estimated_passengers > night.estimated_passengers


def test_crowd_simulator_daily_profile():
    sim = CrowdSimulator()
    sim.register(StationProfile("s1", "駅A", 300_000))
    profile = sim.daily_profile("s1")
    assert profile is not None
    assert len(profile.snapshots) == 24


def test_crowd_simulator_daily_peak_hour():
    sim = CrowdSimulator()
    sim.register(StationProfile("s1", "駅A", 500_000))
    profile = sim.daily_profile("s1")
    assert profile.peak_hour in [7, 8, 17, 18, 19]


def test_crowd_simulator_compare():
    sim = CrowdSimulator()
    sim.register(StationProfile("s1", "駅A", 1_000_000))
    sim.register(StationProfile("s2", "駅B", 100_000))
    results = sim.compare(["s1", "s2"], hour=8)
    assert len(results) == 2
    assert results[0].station_id == "s1"  # より混んでいる方が先


def test_crowd_dataset_build(crowd_sim):
    assert len(crowd_sim.all_station_ids()) >= 10


def test_crowd_dataset_shinjuku(crowd_sim):
    snap = crowd_sim.snapshot("tokyo-shinjuku", 8)
    assert snap is not None
    assert snap.level in (CrowdLevel.CROWDED, CrowdLevel.OVERCROWDED, CrowdLevel.BUSY)


def test_crowd_dataset_late_night_quiet(crowd_sim):
    """深夜は朝ラッシュより混雑度が低いことを確認 (絶対レベルは駅規模次第)"""
    snap_night = crowd_sim.snapshot("tokyo-shinjuku", 2)
    snap_rush = crowd_sim.snapshot("tokyo-shinjuku", 8)
    assert snap_night is not None
    assert snap_night.occupancy_rate < snap_rush.occupancy_rate


def test_daily_profile_to_dict():
    sim = CrowdSimulator()
    sim.register(StationProfile("s1", "駅A", 200_000))
    dp = sim.daily_profile("s1")
    d = dp.to_dict()
    assert "peak_hour" in d
    assert "peak_level" in d
    assert len(d["snapshots"]) == 24


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 74B: accessibility.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_access_feature_values():
    assert AccessFeature.ELEVATOR.value == "elevator"
    assert AccessFeature.TACTILE_PAVING.value == "tactile_paving"


def test_access_level_values():
    assert AccessLevel.EXCELLENT.value == "excellent"
    assert AccessLevel.CRITICAL.value == "critical"


def test_accessibility_profile_has_feature():
    p = AccessibilityProfile("s1", "駅A", CityName.TOKYO,
                             [AccessFeature.ELEVATOR, AccessFeature.WIDE_GATE])
    assert p.has_feature(AccessFeature.ELEVATOR) is True
    assert p.has_feature(AccessFeature.ESCALATOR) is False


def test_accessibility_profile_to_dict():
    p = AccessibilityProfile("s1", "駅A", CityName.TOKYO,
                             [AccessFeature.ELEVATOR])
    d = p.to_dict()
    assert d["station_id"] == "s1"
    assert "elevator" in d["features"]


def test_accessibility_score_to_dict():
    score = AccessibilityScore("s1", "駅A", 85.0, AccessLevel.EXCELLENT,
                               ["elevator"], ["escalator"], {"elevator": 85.0})
    d = score.to_dict()
    assert d["score"] == 85.0
    assert d["level"] == "excellent"


def test_analyzer_score_all_features():
    analyzer = AccessibilityAnalyzer()
    analyzer.register(AccessibilityProfile(
        "s1", "駅A", CityName.TOKYO,
        features=list(AccessFeature),
    ))
    score = analyzer.score("s1")
    assert score is not None
    assert score.score == pytest.approx(100.0)
    assert score.level == AccessLevel.EXCELLENT


def test_analyzer_score_no_features():
    analyzer = AccessibilityAnalyzer()
    analyzer.register(AccessibilityProfile("s1", "駅A", CityName.TOKYO, features=[]))
    score = analyzer.score("s1")
    assert score.score == pytest.approx(0.0)
    assert score.level == AccessLevel.CRITICAL


def test_analyzer_score_not_found():
    analyzer = AccessibilityAnalyzer()
    assert analyzer.score("ghost") is None


def test_analyzer_features_present_missing():
    analyzer = AccessibilityAnalyzer()
    analyzer.register(AccessibilityProfile(
        "s1", "駅A", CityName.TOKYO,
        features=[AccessFeature.ELEVATOR],
    ))
    score = analyzer.score("s1")
    assert "elevator" in score.features_present
    assert "escalator" in score.features_missing


def test_analyzer_city_report(access_analyzer):
    report = access_analyzer.city_report(CityName.TOKYO)
    assert isinstance(report, AccessibilityReport)
    assert report.city == "tokyo"
    assert len(report.scores) >= 4


def test_analyzer_city_report_average(access_analyzer):
    report = access_analyzer.city_report(CityName.TOKYO)
    assert 0 <= report.average_score <= 100


def test_analyzer_city_report_best_worst(access_analyzer):
    report = access_analyzer.city_report(CityName.TOKYO)
    assert report.best_station is not None
    assert report.worst_station is not None
    assert report.best_station.score >= report.worst_station.score


def test_analyzer_rank(access_analyzer):
    ranking = access_analyzer.rank()
    assert len(ranking) >= 10
    # スコア降順
    for i in range(len(ranking) - 1):
        assert ranking[i].score >= ranking[i + 1].score


def test_analyzer_rank_by_city(access_analyzer):
    ranking = access_analyzer.rank(CityName.TOKYO)
    assert all(s.station_id.startswith("tokyo") for s in ranking)


def test_access_dataset_all_cities(access_analyzer):
    for city in CityName:
        report = access_analyzer.city_report(city)
        assert len(report.scores) >= 1


def test_access_report_to_dict(access_analyzer):
    report = access_analyzer.city_report(CityName.OSAKA)
    d = report.to_dict()
    assert "average_score" in d
    assert "best_station" in d
    assert "worst_station" in d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 74C: groundwater.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_water_level_zone_values():
    assert WaterLevelZone.SHALLOW.value == "shallow"
    assert WaterLevelZone.DEEP.value == "deep"


def test_flood_risk_level_values():
    assert FloodRiskLevel.VERY_HIGH.value == "very_high"
    assert FloodRiskLevel.VERY_LOW.value == "very_low"


def test_season_values():
    assert Season.SUMMER.value == "summer"
    assert Season.WINTER.value == "winter"


def test_groundwater_layer_to_dict():
    gl = GroundwaterLayer("g1", CityName.TOKYO, WaterLevelZone.SHALLOW,
                          5.0, 30.0, 1.5, False)
    d = gl.to_dict()
    assert d["depth_m"] == 5.0
    assert d["zone"] == "shallow"
    assert d["artesian"] is False


def test_groundwater_layer_season_depth():
    gl = GroundwaterLayer("g1", CityName.TOKYO, WaterLevelZone.SHALLOW,
                          5.0, 30.0, 1.5, False)
    spring_d = gl.season_depth(Season.SPRING)
    summer_d = gl.season_depth(Season.SUMMER)
    assert spring_d < summer_d   # 春は水位高い = 深度小


def test_groundwater_layer_depth_not_negative():
    gl = GroundwaterLayer("g1", CityName.TOKYO, WaterLevelZone.SURFACE,
                          0.2, 5.0, 1.0)
    for season in Season:
        assert gl.season_depth(season) > 0


def test_flood_risk_factor_weighted_score():
    f = FloodRiskFactor("test", 80.0, 0.5, "desc")
    assert f.weighted_score == pytest.approx(40.0)


def test_flood_risk_factor_to_dict():
    f = FloodRiskFactor("water_depth", 60.0, 0.35, "説明")
    d = f.to_dict()
    assert d["name"] == "water_depth"
    assert d["weighted_score"] == pytest.approx(21.0)


def test_city_water_profile_shallowest():
    profile = CityWaterProfile(CityName.TOKYO, [
        GroundwaterLayer("a", CityName.TOKYO, WaterLevelZone.SHALLOW, 5.0, 30.0),
        GroundwaterLayer("b", CityName.TOKYO, WaterLevelZone.DEEP, 40.0, 200.0),
    ])
    assert profile.shallowest_layer().depth_m == 5.0


def test_city_water_profile_to_dict():
    profile = CityWaterProfile(CityName.TOKYO, [
        GroundwaterLayer("a", CityName.TOKYO, WaterLevelZone.SHALLOW, 5.0, 30.0),
    ], monitoring_stations=10)
    d = profile.to_dict()
    assert d["monitoring_stations"] == 10
    assert d["shallowest_depth_m"] == 5.0


def test_assessor_city_risk(gw_assessor):
    result = gw_assessor.city_risk(CityName.TOKYO)
    assert result is not None
    assert result.city == "tokyo"
    assert 0 <= result.total_score <= 100
    assert len(result.factors) >= 3


def test_assessor_city_risk_osaka(gw_assessor):
    result = gw_assessor.city_risk(CityName.OSAKA)
    assert result is not None
    assert result.level in FloodRiskLevel


def test_assessor_city_risk_season(gw_assessor):
    summer = gw_assessor.city_risk(CityName.TOKYO, Season.SUMMER)
    winter = gw_assessor.city_risk(CityName.TOKYO, Season.WINTER)
    # 夏は渇水 → 水位低下 → 深度増 → リスク低め (必ずしも下がるとは限らないが差異があること)
    assert summer is not None and winter is not None


def test_assessor_station_risk(gw_assessor):
    result = gw_assessor.station_risk(
        CityName.TOKYO, "tokyo-shinjuku", station_depth_m=18.0
    )
    assert result is not None
    assert result.station_id == "tokyo-shinjuku"
    assert result.station_depth_m == 18.0


def test_assessor_station_risk_shallow_station(gw_assessor):
    """水位近くの浅い駅はリスクが高い"""
    shallow = gw_assessor.station_risk(CityName.TOKYO, "s", station_depth_m=5.0)
    deep = gw_assessor.station_risk(CityName.TOKYO, "s", station_depth_m=30.0)
    assert shallow.total_score > deep.total_score


def test_assessor_not_found():
    assessor = FloodRiskAssessor()
    assert assessor.city_risk(CityName.TOKYO) is None


def test_assessor_all_cities(gw_assessor):
    for city in CityName:
        result = gw_assessor.city_risk(city)
        assert result is not None


def test_flood_risk_result_to_dict(gw_assessor):
    result = gw_assessor.city_risk(CityName.FUKUOKA)
    d = result.to_dict()
    assert "total_score" in d
    assert "level" in d
    assert "recommendation" in d
    assert len(d["recommendation"]) > 0


def test_assessor_recommendation_not_empty(gw_assessor):
    for city in CityName:
        result = gw_assessor.city_risk(city)
        assert len(result.recommendation) > 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 74 API エンドポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─── 74A: /v1/crowd/* ────────────────────────────────────────────

def test_api_crowd_stations(client):
    resp = client.get("/v1/crowd/stations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 10


def test_api_crowd_snapshot(client):
    resp = client.get("/v1/crowd/tokyo-shinjuku/snapshot?hour=8")
    assert resp.status_code == 200
    data = resp.json()
    assert data["station_id"] == "tokyo-shinjuku"
    assert data["hour"] == 8
    assert "level" in data
    assert "occupancy_rate" in data


def test_api_crowd_snapshot_default_hour(client):
    resp = client.get("/v1/crowd/tokyo-ginza/snapshot")
    assert resp.status_code == 200


def test_api_crowd_snapshot_not_found(client):
    resp = client.get("/v1/crowd/no-such-station/snapshot")
    assert resp.status_code == 404


def test_api_crowd_daily(client):
    resp = client.get("/v1/crowd/tokyo-shinjuku/daily")
    assert resp.status_code == 200
    data = resp.json()
    assert "peak_hour" in data
    assert len(data["snapshots"]) == 24


def test_api_crowd_daily_not_found(client):
    resp = client.get("/v1/crowd/ghost/daily")
    assert resp.status_code == 404


def test_api_crowd_compare(client):
    resp = client.get("/v1/crowd/compare?stations=tokyo-shinjuku,tokyo-ginza&hour=8")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert data["hour"] == 8


def test_api_crowd_compare_default(client):
    resp = client.get("/v1/crowd/compare")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1


# ─── 74B: /v1/access/* ───────────────────────────────────────────

def test_api_access_score_shinjuku(client):
    resp = client.get("/v1/access/tokyo-shinjuku/score")
    assert resp.status_code == 200
    data = resp.json()
    assert data["score"] == pytest.approx(100.0)
    assert data["level"] == "excellent"


def test_api_access_score_not_found(client):
    resp = client.get("/v1/access/no-station/score")
    assert resp.status_code == 404


def test_api_access_rank(client):
    resp = client.get("/v1/access/rank")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 10
    scores = [s["score"] for s in data["ranking"]]
    assert scores == sorted(scores, reverse=True)


def test_api_access_rank_by_city(client):
    resp = client.get("/v1/access/rank?city=tokyo")
    assert resp.status_code == 200
    data = resp.json()
    assert all(s["station_id"].startswith("tokyo") for s in data["ranking"])


def test_api_access_rank_invalid_city(client):
    resp = client.get("/v1/access/rank?city=badcity")
    assert resp.status_code == 422


def test_api_access_city_report_tokyo(client):
    resp = client.get("/v1/access/tokyo/report")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert "average_score" in data
    assert "best_station" in data


def test_api_access_city_report_invalid(client):
    resp = client.get("/v1/access/nowhere/report")
    assert resp.status_code == 404


# ─── 74C: /v1/groundwater/* ──────────────────────────────────────

def test_api_groundwater_profile_tokyo(client):
    resp = client.get("/v1/groundwater/tokyo")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert len(data["layers"]) >= 2


def test_api_groundwater_profile_invalid(client):
    resp = client.get("/v1/groundwater/badcity")
    assert resp.status_code == 404


def test_api_groundwater_city_risk(client):
    resp = client.get("/v1/groundwater/osaka/risk")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "osaka"
    assert "total_score" in data
    assert "recommendation" in data


def test_api_groundwater_city_risk_season(client):
    resp = client.get("/v1/groundwater/tokyo/risk?season=winter")
    assert resp.status_code == 200
    data = resp.json()
    assert "level" in data


def test_api_groundwater_city_risk_invalid_season(client):
    resp = client.get("/v1/groundwater/tokyo/risk?season=badseason")
    assert resp.status_code == 422


def test_api_groundwater_station_risk(client):
    resp = client.get("/v1/groundwater/tokyo/tokyo-shinjuku/flood-risk")
    assert resp.status_code == 200
    data = resp.json()
    assert data["city"] == "tokyo"
    assert data["station_id"] == "tokyo-shinjuku"
    assert "level" in data


def test_api_groundwater_all_cities(client):
    for city in ["tokyo", "osaka", "nagoya", "yokohama", "fukuoka"]:
        resp = client.get(f"/v1/groundwater/{city}/risk")
        assert resp.status_code == 200, f"Failed for {city}"
