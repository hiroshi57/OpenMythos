"""
Sprint 76 テスト — 交通量分析 / エネルギーモニタリング / 群衆予測
"""
import pytest

# ─── 76A: TrafficAnalyzer ─────────────────────────────────────────

from open_mythos.skills.traffic_analyzer import (
    TrafficAnalyzer, TrafficStore, TrafficLevel, TimeOfDay,
    _classify_level, _get_time_of_day,
)


def _make_traffic():
    store = TrafficStore()
    analyzer = TrafficAnalyzer(store=store)
    return analyzer, store


class TestTrafficLevel:
    def test_classify_clear(self):
        assert _classify_level(70.0) == TrafficLevel.CLEAR

    def test_classify_moderate(self):
        assert _classify_level(50.0) == TrafficLevel.MODERATE

    def test_classify_congested(self):
        assert _classify_level(30.0) == TrafficLevel.CONGESTED

    def test_classify_gridlock(self):
        assert _classify_level(10.0) == TrafficLevel.GRIDLOCK

    def test_classify_boundary_clear(self):
        assert _classify_level(60.0) == TrafficLevel.CLEAR

    def test_classify_boundary_moderate(self):
        assert _classify_level(40.0) == TrafficLevel.MODERATE


class TestTimeOfDay:
    def test_early_morning(self):
        assert _get_time_of_day(3) == TimeOfDay.EARLY_MORNING

    def test_morning_rush(self):
        assert _get_time_of_day(8) == TimeOfDay.MORNING_RUSH

    def test_midday(self):
        assert _get_time_of_day(13) == TimeOfDay.MIDDAY

    def test_evening_rush(self):
        assert _get_time_of_day(18) == TimeOfDay.EVENING_RUSH

    def test_night(self):
        assert _get_time_of_day(22) == TimeOfDay.NIGHT


class TestTrafficStore:
    def test_add_and_get(self):
        analyzer, store = _make_traffic()
        seg = analyzer.add_segment("s1", "国道1号", "Tokyo", 1200, 45.0, 15.0)
        assert store.get("s1") is not None
        assert store.get("s1").road_name == "国道1号"

    def test_count(self):
        analyzer, store = _make_traffic()
        analyzer.add_segment("s1", "R1", "Tokyo", 1000, 50.0, 12.0)
        analyzer.add_segment("s2", "R2", "Tokyo", 800, 35.0, 10.0)
        assert store.count() == 2

    def test_list_by_city(self):
        analyzer, store = _make_traffic()
        analyzer.add_segment("s1", "R1", "Tokyo", 1000, 50.0, 12.0)
        analyzer.add_segment("s2", "R2", "Osaka", 900, 48.0, 11.0)
        tokyo = store.list_by_city("Tokyo")
        assert len(tokyo) == 1
        assert tokyo[0].city == "Tokyo"

    def test_delete(self):
        analyzer, store = _make_traffic()
        analyzer.add_segment("s1", "R1", "Tokyo", 1000, 50.0, 12.0)
        assert store.delete("s1") is True
        assert store.count() == 0

    def test_delete_nonexistent(self):
        _, store = _make_traffic()
        assert store.delete("nonexistent") is False


class TestTrafficSegment:
    def test_level_property(self):
        analyzer, _ = _make_traffic()
        seg = analyzer.add_segment("s1", "R1", "Tokyo", 500, 25.0, 8.0)
        assert seg.level == TrafficLevel.CONGESTED

    def test_congestion_score_range(self):
        analyzer, _ = _make_traffic()
        seg = analyzer.add_segment("s1", "R1", "Tokyo", 500, 0.0, 20.0)
        assert 0.0 <= seg.congestion_score <= 1.0

    def test_congestion_score_clear(self):
        analyzer, _ = _make_traffic()
        seg = analyzer.add_segment("s1", "R1", "Tokyo", 500, 80.0, 6.0)
        assert seg.congestion_score == 0.0

    def test_to_dict_keys(self):
        analyzer, _ = _make_traffic()
        seg = analyzer.add_segment("s1", "R1", "Tokyo", 500, 50.0, 10.0)
        d = seg.to_dict()
        assert "segment_id" in d
        assert "level" in d
        assert "congestion_score" in d
        assert "time_of_day" in d


class TestTrafficAnalyzer:
    def test_get_hotspots(self):
        analyzer, _ = _make_traffic()
        analyzer.add_segment("s1", "R1", "Tokyo", 1200, 15.0, 18.0)
        analyzer.add_segment("s2", "R2", "Tokyo", 900, 55.0, 10.0)
        hotspots = analyzer.get_hotspots("Tokyo", top_n=2)
        assert len(hotspots) >= 1
        assert hotspots[0].road_name == "R1"  # 低速 → 高渋滞

    def test_get_hotspots_empty_city(self):
        analyzer, _ = _make_traffic()
        hotspots = analyzer.get_hotspots("NoCity")
        assert hotspots == []

    def test_predict_by_hour_length(self):
        analyzer, _ = _make_traffic()
        forecasts = analyzer.predict_by_hour()
        assert len(forecasts) == 24

    def test_predict_by_hour_rush_slower(self):
        analyzer, _ = _make_traffic()
        forecasts = analyzer.predict_by_hour(base_speed_kmh=60.0)
        morning = [f for f in forecasts if f.time_of_day == TimeOfDay.MORNING_RUSH]
        night = [f for f in forecasts if f.time_of_day == TimeOfDay.NIGHT]
        # ラッシュ時は scale が大きい → 速度が低い → 渋滞スコアが高い
        assert morning[0].scale_factor > night[0].scale_factor

    def test_city_summary_empty(self):
        analyzer, _ = _make_traffic()
        summary = analyzer.city_summary("NoCity")
        assert summary["segment_count"] == 0

    def test_city_summary(self):
        analyzer, _ = _make_traffic()
        analyzer.add_segment("s1", "R1", "Tokyo", 1000, 40.0, 12.0)
        analyzer.add_segment("s2", "R2", "Tokyo", 800, 60.0, 8.0)
        summary = analyzer.city_summary("Tokyo")
        assert summary["segment_count"] == 2
        assert summary["avg_speed_kmh"] == 50.0


# ─── 76B: EnergyMonitor ──────────────────────────────────────────

from open_mythos.skills.energy_monitor import (
    EnergyMonitor, EnergyStore, EnergyType, AnomalyLevel,
)


def _make_energy():
    store = EnergyStore()
    monitor = EnergyMonitor(store=store)
    return monitor, store


class TestEnergyStore:
    def test_add_and_get(self):
        monitor, store = _make_energy()
        r = monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.ELECTRICITY, 120.0)
        assert store.get("r1") is not None
        assert store.get("r1").value == 120.0

    def test_list_by_facility(self):
        monitor, store = _make_energy()
        monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.ELECTRICITY, 100.0)
        monitor.add_reading("r2", "fac2", "Tokyo", EnergyType.GAS, 50.0)
        fac1 = store.list_by_facility("fac1")
        assert len(fac1) == 1

    def test_list_by_city(self):
        monitor, store = _make_energy()
        monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.ELECTRICITY, 100.0)
        monitor.add_reading("r2", "fac1", "Osaka", EnergyType.ELECTRICITY, 80.0)
        tokyo = store.list_by_city("Tokyo")
        assert len(tokyo) == 1

    def test_delete(self):
        monitor, store = _make_energy()
        monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.ELECTRICITY, 100.0)
        assert store.delete("r1") is True
        assert store.count() == 0

    def test_count(self):
        monitor, store = _make_energy()
        monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.ELECTRICITY, 100.0)
        monitor.add_reading("r2", "fac1", "Tokyo", EnergyType.GAS, 50.0)
        assert store.count() == 2


class TestEnergyReading:
    def test_unit_electricity(self):
        monitor, _ = _make_energy()
        r = monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.ELECTRICITY, 100.0)
        assert r.unit == "kWh"

    def test_unit_gas(self):
        monitor, _ = _make_energy()
        r = monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.GAS, 50.0)
        assert r.unit == "m³"

    def test_to_dict_keys(self):
        monitor, _ = _make_energy()
        r = monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.ELECTRICITY, 100.0)
        d = r.to_dict()
        for key in ["reading_id", "facility_id", "energy_type", "value", "unit"]:
            assert key in d


class TestEnergyMonitor:
    def test_summarize_facility(self):
        monitor, _ = _make_energy()
        for i in range(3):
            monitor.add_reading(f"r{i}", "fac1", "Tokyo", EnergyType.ELECTRICITY, float(100 + i * 10))
        s = monitor.summarize_facility("fac1", EnergyType.ELECTRICITY)
        assert s is not None
        assert s.reading_count == 3
        assert s.total == 330.0

    def test_summarize_city(self):
        monitor, _ = _make_energy()
        monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.ELECTRICITY, 100.0)
        monitor.add_reading("r2", "fac2", "Tokyo", EnergyType.ELECTRICITY, 200.0)
        s = monitor.summarize_city("Tokyo", EnergyType.ELECTRICITY)
        assert s is not None
        assert s.total == 300.0

    def test_summarize_returns_none_on_empty(self):
        monitor, _ = _make_energy()
        s = monitor.summarize_city("NoCity", EnergyType.ELECTRICITY)
        assert s is None

    def test_detect_anomalies_critical(self):
        monitor, _ = _make_energy()
        # baseline ≈ 100, critical = 100 * 2.5 = 250
        for i in range(5):
            monitor.add_reading(f"r{i}", f"fac{i}", "Tokyo", EnergyType.ELECTRICITY, 100.0)
        monitor.add_reading("r_high", "fac_big", "Tokyo", EnergyType.ELECTRICITY, 400.0)
        anomalies = monitor.detect_anomalies("Tokyo", EnergyType.ELECTRICITY)
        assert len(anomalies) >= 1
        assert anomalies[0].level == AnomalyLevel.CRITICAL

    def test_detect_anomalies_high(self):
        monitor, _ = _make_energy()
        for i in range(5):
            monitor.add_reading(f"r{i}", f"fac{i}", "Tokyo", EnergyType.ELECTRICITY, 100.0)
        monitor.add_reading("r_mid", "fac_mid", "Tokyo", EnergyType.ELECTRICITY, 180.0)
        anomalies = monitor.detect_anomalies("Tokyo", EnergyType.ELECTRICITY)
        levels = {a.level for a in anomalies}
        assert AnomalyLevel.HIGH in levels

    def test_detect_no_anomalies(self):
        monitor, _ = _make_energy()
        for i in range(5):
            monitor.add_reading(f"r{i}", f"fac{i}", "Tokyo", EnergyType.ELECTRICITY, 100.0)
        anomalies = monitor.detect_anomalies("Tokyo", EnergyType.ELECTRICITY)
        assert anomalies == []

    def test_hourly_profile_length(self):
        monitor, _ = _make_energy()
        monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.ELECTRICITY, 100.0, hour=9)
        profile = monitor.hourly_profile("Tokyo", EnergyType.ELECTRICITY)
        assert len(profile) == 24

    def test_hourly_profile_values(self):
        monitor, _ = _make_energy()
        monitor.add_reading("r1", "fac1", "Tokyo", EnergyType.ELECTRICITY, 150.0, hour=9)
        profile = monitor.hourly_profile("Tokyo", EnergyType.ELECTRICITY)
        hour9 = next(p for p in profile if p["hour"] == 9)
        assert hour9["avg_value"] == 150.0


# ─── 76C: CrowdPredictor ─────────────────────────────────────────

from open_mythos.skills.crowd_predictor import (
    CrowdPredictor, CrowdStore, CrowdLevel, WeatherCondition, EventType,
    _classify_crowd,
)


def _make_crowd():
    store = CrowdStore()
    predictor = CrowdPredictor(store=store)
    return predictor, store


class TestCrowdLevel:
    def test_sparse(self):
        assert _classify_crowd(100.0) == CrowdLevel.SPARSE

    def test_normal(self):
        assert _classify_crowd(700.0) == CrowdLevel.NORMAL

    def test_crowded(self):
        assert _classify_crowd(2000.0) == CrowdLevel.CROWDED

    def test_packed(self):
        assert _classify_crowd(4000.0) == CrowdLevel.PACKED


class TestCrowdStore:
    def test_add_and_get(self):
        predictor, store = _make_crowd()
        snap = predictor.add_snapshot("sn1", "新宿駅", "Tokyo", 3000)
        assert store.get("sn1") is not None
        assert store.get("sn1").count == 3000

    def test_list_by_city(self):
        predictor, store = _make_crowd()
        predictor.add_snapshot("sn1", "新宿駅", "Tokyo", 3000)
        predictor.add_snapshot("sn2", "梅田駅", "Osaka", 2000)
        tokyo = store.list_by_city("Tokyo")
        assert len(tokyo) == 1

    def test_list_by_spot(self):
        predictor, store = _make_crowd()
        predictor.add_snapshot("sn1", "新宿駅", "Tokyo", 3000, hour=8)
        predictor.add_snapshot("sn2", "新宿駅", "Tokyo", 1000, hour=14)
        snaps = store.list_by_spot("新宿駅", "Tokyo")
        assert len(snaps) == 2

    def test_delete(self):
        predictor, store = _make_crowd()
        predictor.add_snapshot("sn1", "新宿駅", "Tokyo", 3000)
        assert store.delete("sn1") is True
        assert store.count() == 0

    def test_count(self):
        predictor, store = _make_crowd()
        predictor.add_snapshot("sn1", "駅A", "Tokyo", 1000)
        predictor.add_snapshot("sn2", "駅B", "Tokyo", 2000)
        assert store.count() == 2


class TestCrowdSnapshot:
    def test_level_crowded(self):
        predictor, _ = _make_crowd()
        snap = predictor.add_snapshot("sn1", "新宿駅", "Tokyo", 2000)
        assert snap.level == CrowdLevel.CROWDED

    def test_to_dict_keys(self):
        predictor, _ = _make_crowd()
        snap = predictor.add_snapshot("sn1", "新宿駅", "Tokyo", 1000)
        d = snap.to_dict()
        for key in ["snapshot_id", "spot_name", "city", "count", "level"]:
            assert key in d


class TestCrowdPredictor:
    def test_predict_returns_result(self):
        predictor, _ = _make_crowd()
        result = predictor.predict("新宿駅", "Tokyo", hour=8)
        assert result.predicted_count >= 0
        assert result.predicted_level in list(CrowdLevel)

    def test_predict_event_boosts_count(self):
        predictor, _ = _make_crowd()
        base = predictor.predict("渋谷", "Tokyo", hour=14, event=EventType.NONE)
        event = predictor.predict("渋谷", "Tokyo", hour=14, event=EventType.CONCERT)
        assert event.predicted_count > base.predicted_count

    def test_predict_rainy_reduces_count(self):
        predictor, _ = _make_crowd()
        sunny = predictor.predict("渋谷", "Tokyo", hour=14, weather=WeatherCondition.SUNNY)
        rainy = predictor.predict("渋谷", "Tokyo", hour=14, weather=WeatherCondition.RAINY)
        assert rainy.predicted_count < sunny.predicted_count

    def test_predict_uses_existing_snapshots(self):
        predictor, _ = _make_crowd()
        predictor.add_snapshot("sn1", "新宿駅", "Tokyo", 5000, hour=8)
        result = predictor.predict("新宿駅", "Tokyo", hour=8)
        # base は 5000 なので予測値も高くなるはず
        assert result.predicted_count > 1000

    def test_predict_to_dict(self):
        predictor, _ = _make_crowd()
        result = predictor.predict("新宿駅", "Tokyo", hour=12)
        d = result.to_dict()
        assert "predicted_count" in d
        assert "predicted_level" in d
        assert "factors" in d

    def test_heatmap_empty(self):
        predictor, _ = _make_crowd()
        cells = predictor.heatmap("NoCity")
        assert cells == []

    def test_heatmap_sorted_by_count(self):
        predictor, _ = _make_crowd()
        predictor.add_snapshot("sn1", "新宿駅", "Tokyo", 4000)
        predictor.add_snapshot("sn2", "渋谷駅", "Tokyo", 1000)
        cells = predictor.heatmap("Tokyo")
        assert len(cells) == 2
        assert cells[0].avg_count > cells[1].avg_count  # 降順

    def test_daily_forecast_length(self):
        predictor, _ = _make_crowd()
        forecasts = predictor.daily_forecast("新宿駅", "Tokyo")
        assert len(forecasts) == 24

    def test_daily_forecast_rush_packed(self):
        predictor, _ = _make_crowd()
        predictor.add_snapshot("sn1", "新宿駅", "Tokyo", 8000, hour=8)
        forecasts = predictor.daily_forecast(
            "新宿駅", "Tokyo",
            event=EventType.COMMUTE,
        )
        rush = [f for f in forecasts if f.hour in (7, 8, 9)]
        assert any(f.predicted_level in (CrowdLevel.CROWDED, CrowdLevel.PACKED) for f in rush)
