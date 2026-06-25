"""
Sprint 77 テスト — 災害アラート / 水質モニタリング / 騒音マッピング
"""
import pytest

# ─── 77A: DisasterAlertManager ───────────────────────────────────

from open_mythos.skills.disaster_alert import (
    DisasterAlertManager, AlertStore, DisasterType, AlertLevel, AlertStatus,
    _recommend_actions,
)


def _make_alert():
    store = AlertStore()
    manager = DisasterAlertManager(store=store)
    return manager, store


class TestAlertStore:
    def test_add_and_get(self):
        manager, store = _make_alert()
        a = manager.issue_alert("a1", DisasterType.EARTHQUAKE, "Tokyo")
        assert store.get("a1") is not None

    def test_count(self):
        manager, store = _make_alert()
        manager.issue_alert("a1", DisasterType.FLOOD, "Tokyo")
        manager.issue_alert("a2", DisasterType.FIRE, "Osaka")
        assert store.count() == 2

    def test_list_by_city(self):
        manager, store = _make_alert()
        manager.issue_alert("a1", DisasterType.FLOOD, "Tokyo")
        manager.issue_alert("a2", DisasterType.FIRE, "Osaka")
        tokyo = store.list_by_city("Tokyo")
        assert len(tokyo) == 1

    def test_list_active(self):
        manager, store = _make_alert()
        manager.issue_alert("a1", DisasterType.FLOOD, "Tokyo")
        manager.issue_alert("a2", DisasterType.FIRE, "Tokyo")
        store.resolve("a1")
        active = store.list_active("Tokyo")
        assert len(active) == 1

    def test_delete(self):
        manager, store = _make_alert()
        manager.issue_alert("a1", DisasterType.EARTHQUAKE, "Tokyo")
        assert store.delete("a1") is True
        assert store.count() == 0

    def test_resolve(self):
        manager, store = _make_alert()
        manager.issue_alert("a1", DisasterType.FLOOD, "Tokyo")
        ok = store.resolve("a1")
        assert ok is True
        assert store.get("a1").status == AlertStatus.RESOLVED

    def test_resolve_nonexistent(self):
        _, store = _make_alert()
        assert store.resolve("nonexistent") is False


class TestDisasterAlert:
    def test_default_level_tsunami(self):
        manager, _ = _make_alert()
        a = manager.issue_alert("a1", DisasterType.TSUNAMI, "Tokyo")
        assert a.level == AlertLevel.CRITICAL

    def test_default_level_fire(self):
        manager, _ = _make_alert()
        a = manager.issue_alert("a1", DisasterType.FIRE, "Tokyo")
        assert a.level == AlertLevel.WATCH

    def test_custom_level(self):
        manager, _ = _make_alert()
        a = manager.issue_alert("a1", DisasterType.FLOOD, "Tokyo", level=AlertLevel.INFO)
        assert a.level == AlertLevel.INFO

    def test_is_active(self):
        manager, _ = _make_alert()
        a = manager.issue_alert("a1", DisasterType.EARTHQUAKE, "Tokyo")
        assert a.is_active is True

    def test_recommended_actions_critical(self):
        actions = _recommend_actions(AlertLevel.CRITICAL)
        assert len(actions) > 0
        assert any("避難" in act for act in actions)

    def test_to_dict_keys(self):
        manager, _ = _make_alert()
        a = manager.issue_alert("a1", DisasterType.EARTHQUAKE, "Tokyo")
        d = a.to_dict()
        for key in ["alert_id", "disaster_type", "city", "level", "status",
                    "recommended_actions", "is_active"]:
            assert key in d


class TestDisasterAlertManager:
    def test_get_active_sorted_by_level(self):
        manager, _ = _make_alert()
        manager.issue_alert("a1", DisasterType.FIRE, "Tokyo", level=AlertLevel.WATCH)
        manager.issue_alert("a2", DisasterType.TSUNAMI, "Tokyo", level=AlertLevel.CRITICAL)
        manager.issue_alert("a3", DisasterType.FLOOD, "Tokyo", level=AlertLevel.WARNING)
        alerts = manager.get_active_alerts("Tokyo")
        assert alerts[0].level == AlertLevel.CRITICAL

    def test_resolve_alert(self):
        manager, store = _make_alert()
        manager.issue_alert("a1", DisasterType.FLOOD, "Tokyo")
        assert manager.resolve_alert("a1") is True
        assert store.get("a1").status == AlertStatus.RESOLVED

    def test_resolve_nonexistent(self):
        manager, _ = _make_alert()
        assert manager.resolve_alert("none") is False

    def test_city_summary(self):
        manager, _ = _make_alert()
        manager.issue_alert("a1", DisasterType.FLOOD, "Tokyo", level=AlertLevel.WARNING)
        manager.issue_alert("a2", DisasterType.FIRE, "Tokyo", level=AlertLevel.WATCH)
        summary = manager.city_summary("Tokyo")
        assert summary.total == 2
        assert summary.active == 2
        assert summary.highest_level == AlertLevel.WARNING.value

    def test_city_summary_after_resolve(self):
        manager, _ = _make_alert()
        manager.issue_alert("a1", DisasterType.FLOOD, "Tokyo")
        manager.resolve_alert("a1")
        summary = manager.city_summary("Tokyo")
        assert summary.active == 0
        assert summary.highest_level is None

    def test_city_summary_empty(self):
        manager, _ = _make_alert()
        summary = manager.city_summary("NoCity")
        assert summary.total == 0


# ─── 77B: WaterQualityMonitor ────────────────────────────────────

from open_mythos.skills.water_quality import (
    WaterQualityMonitor, WaterQualityStore, WaterParam, SourceType,
    WaterQualityStatus, _assess_status,
)


def _make_wq():
    store = WaterQualityStore()
    monitor = WaterQualityMonitor(store=store)
    return monitor, store


class TestWaterQualityStatus:
    def test_ph_safe(self):
        assert _assess_status(WaterParam.PH, 7.0) == WaterQualityStatus.SAFE

    def test_ph_caution_low(self):
        assert _assess_status(WaterParam.PH, 6.2) == WaterQualityStatus.CAUTION

    def test_ph_unsafe_low(self):
        assert _assess_status(WaterParam.PH, 5.0) == WaterQualityStatus.UNSAFE

    def test_turbidity_safe(self):
        assert _assess_status(WaterParam.TURBIDITY, 0.5) == WaterQualityStatus.SAFE

    def test_turbidity_unsafe(self):
        assert _assess_status(WaterParam.TURBIDITY, 5.0) == WaterQualityStatus.UNSAFE

    def test_chlorine_safe(self):
        assert _assess_status(WaterParam.CHLORINE, 0.5) == WaterQualityStatus.SAFE

    def test_do_safe(self):
        assert _assess_status(WaterParam.DO, 8.0) == WaterQualityStatus.SAFE

    def test_do_unsafe(self):
        assert _assess_status(WaterParam.DO, 3.0) == WaterQualityStatus.UNSAFE


class TestWaterQualityStore:
    def test_add_and_get(self):
        monitor, store = _make_wq()
        r = monitor.add_reading("r1", "st1", "Tokyo", SourceType.RIVER, WaterParam.PH, 7.2)
        assert store.get("r1") is not None

    def test_list_by_city(self):
        monitor, store = _make_wq()
        monitor.add_reading("r1", "st1", "Tokyo", SourceType.RIVER, WaterParam.PH, 7.0)
        monitor.add_reading("r2", "st2", "Osaka", SourceType.TAP, WaterParam.PH, 7.5)
        assert len(store.list_by_city("Tokyo")) == 1

    def test_list_by_station(self):
        monitor, store = _make_wq()
        monitor.add_reading("r1", "st1", "Tokyo", SourceType.RIVER, WaterParam.PH, 7.0)
        monitor.add_reading("r2", "st1", "Tokyo", SourceType.RIVER, WaterParam.TURBIDITY, 0.8)
        assert len(store.list_by_station("st1")) == 2

    def test_list_unsafe(self):
        monitor, store = _make_wq()
        monitor.add_reading("r1", "st1", "Tokyo", SourceType.RIVER, WaterParam.PH, 7.0)
        monitor.add_reading("r2", "st1", "Tokyo", SourceType.RIVER, WaterParam.TURBIDITY, 10.0)
        unsafe = store.list_unsafe("Tokyo")
        assert len(unsafe) == 1

    def test_delete(self):
        monitor, store = _make_wq()
        monitor.add_reading("r1", "st1", "Tokyo", SourceType.TAP, WaterParam.PH, 7.0)
        assert store.delete("r1") is True
        assert store.count() == 0


class TestWaterReading:
    def test_status_safe(self):
        monitor, _ = _make_wq()
        r = monitor.add_reading("r1", "st1", "Tokyo", SourceType.RIVER, WaterParam.PH, 7.0)
        assert r.status == WaterQualityStatus.SAFE

    def test_status_unsafe(self):
        monitor, _ = _make_wq()
        r = monitor.add_reading("r1", "st1", "Tokyo", SourceType.RIVER, WaterParam.TURBIDITY, 10.0)
        assert r.status == WaterQualityStatus.UNSAFE

    def test_unit(self):
        monitor, _ = _make_wq()
        r = monitor.add_reading("r1", "st1", "Tokyo", SourceType.TAP, WaterParam.PH, 7.0)
        assert r.unit == "pH"

    def test_to_dict_keys(self):
        monitor, _ = _make_wq()
        r = monitor.add_reading("r1", "st1", "Tokyo", SourceType.TAP, WaterParam.PH, 7.0)
        d = r.to_dict()
        for key in ["reading_id", "station_id", "param", "value", "status", "unit"]:
            assert key in d


class TestWaterQualityMonitor:
    def test_station_summary(self):
        monitor, _ = _make_wq()
        for i, v in enumerate([7.0, 7.2, 6.8]):
            monitor.add_reading(f"r{i}", "st1", "Tokyo", SourceType.RIVER, WaterParam.PH, v)
        s = monitor.station_summary("st1", WaterParam.PH)
        assert s is not None
        assert s.reading_count == 3
        assert s.overall_status == WaterQualityStatus.SAFE

    def test_station_summary_with_unsafe(self):
        monitor, _ = _make_wq()
        monitor.add_reading("r1", "st1", "Tokyo", SourceType.RIVER, WaterParam.PH, 7.0)
        monitor.add_reading("r2", "st1", "Tokyo", SourceType.RIVER, WaterParam.PH, 4.0)
        s = monitor.station_summary("st1", WaterParam.PH)
        assert s.overall_status == WaterQualityStatus.UNSAFE
        assert s.unsafe_count == 1

    def test_station_summary_none_on_empty(self):
        monitor, _ = _make_wq()
        assert monitor.station_summary("nostation", WaterParam.PH) is None

    def test_get_unsafe_readings(self):
        monitor, _ = _make_wq()
        monitor.add_reading("r1", "st1", "Tokyo", SourceType.RIVER, WaterParam.TURBIDITY, 10.0)
        monitor.add_reading("r2", "st1", "Tokyo", SourceType.RIVER, WaterParam.PH, 7.0)
        unsafe = monitor.get_unsafe_readings("Tokyo")
        assert len(unsafe) == 1

    def test_city_report(self):
        monitor, _ = _make_wq()
        monitor.add_reading("r1", "st1", "Tokyo", SourceType.TAP, WaterParam.PH, 7.0)
        monitor.add_reading("r2", "st2", "Tokyo", SourceType.TAP, WaterParam.TURBIDITY, 0.5)
        report = monitor.city_report("Tokyo")
        assert report["reading_count"] == 2
        assert "ph" in report["params"]

    def test_city_report_empty(self):
        monitor, _ = _make_wq()
        report = monitor.city_report("NoCity")
        assert report["reading_count"] == 0


# ─── 77C: NoiseMapper ────────────────────────────────────────────

from open_mythos.skills.noise_mapper import (
    NoiseMapper, NoiseMeasurementStore, ZoneType, NoiseStatus, TimeSlot,
    _assess_noise, _get_time_slot,
)


def _make_noise():
    store = NoiseMeasurementStore()
    mapper = NoiseMapper(store=store)
    return mapper, store


class TestNoiseAssessment:
    def test_residential_daytime_compliant(self):
        # 住宅昼間 limit=55dB, near_limit=50dB → 45dB は COMPLIANT
        assert _assess_noise(ZoneType.RESIDENTIAL, 45.0, 12) == NoiseStatus.COMPLIANT

    def test_residential_daytime_violation(self):
        assert _assess_noise(ZoneType.RESIDENTIAL, 60.0, 12) == NoiseStatus.VIOLATION

    def test_residential_nighttime_violation(self):
        assert _assess_noise(ZoneType.RESIDENTIAL, 50.0, 23) == NoiseStatus.VIOLATION

    def test_industrial_daytime_compliant(self):
        # 工業昼間 limit=70dB, near_limit=65dB → 60dB は COMPLIANT
        assert _assess_noise(ZoneType.INDUSTRIAL, 60.0, 10) == NoiseStatus.COMPLIANT

    def test_quiet_zone_near_limit(self):
        # 限界 45dB, near = 40–45dB
        assert _assess_noise(ZoneType.QUIET_ZONE, 42.0, 12) == NoiseStatus.NEAR_LIMIT

    def test_time_slot_day(self):
        assert _get_time_slot(12) == TimeSlot.DAYTIME

    def test_time_slot_night(self):
        assert _get_time_slot(23) == TimeSlot.NIGHTTIME


class TestNoiseMeasurementStore:
    def test_add_and_get(self):
        mapper, store = _make_noise()
        m = mapper.add_measurement("m1", "新宿交差点", "Tokyo", ZoneType.COMMERCIAL, 62.0)
        assert store.get("m1") is not None

    def test_list_by_city(self):
        mapper, store = _make_noise()
        mapper.add_measurement("m1", "地点A", "Tokyo", ZoneType.RESIDENTIAL, 50.0)
        mapper.add_measurement("m2", "地点B", "Osaka", ZoneType.COMMERCIAL, 60.0)
        assert len(store.list_by_city("Tokyo")) == 1

    def test_list_violations(self):
        mapper, store = _make_noise()
        mapper.add_measurement("m1", "地点A", "Tokyo", ZoneType.RESIDENTIAL, 70.0, hour=12)
        mapper.add_measurement("m2", "地点B", "Tokyo", ZoneType.RESIDENTIAL, 40.0, hour=12)
        violations = store.list_violations("Tokyo")
        assert len(violations) == 1

    def test_delete(self):
        mapper, store = _make_noise()
        mapper.add_measurement("m1", "地点A", "Tokyo", ZoneType.COMMERCIAL, 55.0)
        assert store.delete("m1") is True
        assert store.count() == 0


class TestNoiseMeasurement:
    def test_status_compliant(self):
        mapper, _ = _make_noise()
        # 商業昼間 limit=65dB, near_limit=60dB → 55dB は COMPLIANT
        m = mapper.add_measurement("m1", "地点A", "Tokyo", ZoneType.COMMERCIAL, 55.0, hour=12)
        assert m.status == NoiseStatus.COMPLIANT

    def test_status_violation(self):
        mapper, _ = _make_noise()
        m = mapper.add_measurement("m1", "地点A", "Tokyo", ZoneType.RESIDENTIAL, 70.0, hour=12)
        assert m.status == NoiseStatus.VIOLATION

    def test_excess_db(self):
        mapper, _ = _make_noise()
        m = mapper.add_measurement("m1", "地点A", "Tokyo", ZoneType.RESIDENTIAL, 60.0, hour=12)
        assert m.excess_db == 5.0  # 60 - 55 = 5

    def test_excess_db_compliant(self):
        mapper, _ = _make_noise()
        m = mapper.add_measurement("m1", "地点A", "Tokyo", ZoneType.RESIDENTIAL, 40.0, hour=12)
        assert m.excess_db == 0.0

    def test_to_dict_keys(self):
        mapper, _ = _make_noise()
        m = mapper.add_measurement("m1", "地点A", "Tokyo", ZoneType.COMMERCIAL, 55.0)
        d = m.to_dict()
        for key in ["measurement_id", "location_name", "db_level", "status",
                    "limit_db", "excess_db", "time_slot"]:
            assert key in d


class TestNoiseMapper:
    def test_get_violations_sorted(self):
        mapper, _ = _make_noise()
        mapper.add_measurement("m1", "地点A", "Tokyo", ZoneType.RESIDENTIAL, 65.0, hour=12)
        mapper.add_measurement("m2", "地点B", "Tokyo", ZoneType.RESIDENTIAL, 70.0, hour=12)
        violations = mapper.get_violations("Tokyo")
        assert len(violations) == 2
        assert violations[0].excess_db >= violations[1].excess_db

    def test_get_violations_empty(self):
        mapper, _ = _make_noise()
        mapper.add_measurement("m1", "静かな場所", "Tokyo", ZoneType.RESIDENTIAL, 30.0)
        assert mapper.get_violations("Tokyo") == []

    def test_generate_map(self):
        mapper, _ = _make_noise()
        mapper.add_measurement("m1", "新宿", "Tokyo", ZoneType.COMMERCIAL, 68.0, hour=12)
        mapper.add_measurement("m2", "新宿", "Tokyo", ZoneType.COMMERCIAL, 72.0, hour=20)
        mapper.add_measurement("m3", "渋谷", "Tokyo", ZoneType.COMMERCIAL, 60.0, hour=12)
        cells = mapper.generate_map("Tokyo")
        assert len(cells) == 2
        # 新宿は平均高 → 先頭
        assert cells[0].location_name == "新宿"

    def test_generate_map_empty(self):
        mapper, _ = _make_noise()
        assert mapper.generate_map("NoCity") == []

    def test_city_report(self):
        mapper, _ = _make_noise()
        mapper.add_measurement("m1", "地点A", "Tokyo", ZoneType.RESIDENTIAL, 70.0, hour=12)
        mapper.add_measurement("m2", "地点B", "Tokyo", ZoneType.RESIDENTIAL, 40.0, hour=12)
        report = mapper.city_report("Tokyo")
        assert report.total_measurements == 2
        assert report.violations == 1
        assert report.worst_location == "地点A"

    def test_city_report_compliance_rate(self):
        mapper, _ = _make_noise()
        # 商業昼間 limit=65dB, near_limit=60dB → 55dB は COMPLIANT
        for i in range(4):
            mapper.add_measurement(f"m{i}", f"loc{i}", "Tokyo",
                                   ZoneType.COMMERCIAL, 55.0, hour=12)
        report = mapper.city_report("Tokyo")
        assert report.to_dict()["compliance_rate"] == 1.0

    def test_city_report_empty(self):
        mapper, _ = _make_noise()
        report = mapper.city_report("NoCity")
        assert report.total_measurements == 0
