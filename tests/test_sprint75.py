"""
Sprint 75 テスト

75A: 駅環境センサー統合
75B: 乗り換え最適化
75C: 都市インフラダッシュボード
"""
import pytest
from fastapi.testclient import TestClient

# ─── 共通フィクスチャ ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    from serve.api import app
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════
# Sprint 75A — 駅環境センサー
# ═══════════════════════════════════════════════════════════════════

from open_mythos.skills.env_sensor import (
    SensorType, SensorStatus, SensorReading,
    StationEnvironment, EnvSensorAnalyzer, EnvSensorDataset,
    _classify_status, _THRESHOLDS,
)
from open_mythos.skills.city_map import CityName


class TestSensorReading:
    """SensorReading.create() のステータス分類"""

    # 気温
    def test_temp_normal(self):
        r = SensorReading.create(SensorType.TEMPERATURE, 27.9)
        assert r.status == SensorStatus.NORMAL

    def test_temp_warning_boundary(self):
        r = SensorReading.create(SensorType.TEMPERATURE, 28.0)
        assert r.status == SensorStatus.WARNING

    def test_temp_critical_boundary(self):
        r = SensorReading.create(SensorType.TEMPERATURE, 35.0)
        assert r.status == SensorStatus.CRITICAL

    # 湿度
    def test_humidity_normal(self):
        r = SensorReading.create(SensorType.HUMIDITY, 69.9)
        assert r.status == SensorStatus.NORMAL

    def test_humidity_warning(self):
        r = SensorReading.create(SensorType.HUMIDITY, 75.0)
        assert r.status == SensorStatus.WARNING

    def test_humidity_critical(self):
        r = SensorReading.create(SensorType.HUMIDITY, 80.0)
        assert r.status == SensorStatus.CRITICAL

    # CO2
    def test_co2_normal(self):
        r = SensorReading.create(SensorType.CO2, 999.9)
        assert r.status == SensorStatus.NORMAL

    def test_co2_warning(self):
        r = SensorReading.create(SensorType.CO2, 1500.0)
        assert r.status == SensorStatus.WARNING

    def test_co2_critical(self):
        r = SensorReading.create(SensorType.CO2, 2000.0)
        assert r.status == SensorStatus.CRITICAL

    # 騒音
    def test_noise_normal(self):
        r = SensorReading.create(SensorType.NOISE, 74.9)
        assert r.status == SensorStatus.NORMAL

    def test_noise_warning(self):
        r = SensorReading.create(SensorType.NOISE, 80.0)
        assert r.status == SensorStatus.WARNING

    def test_noise_critical(self):
        r = SensorReading.create(SensorType.NOISE, 90.0)
        assert r.status == SensorStatus.CRITICAL

    def test_unit_attached(self):
        r = SensorReading.create(SensorType.CO2, 1500.0)
        assert r.unit == "ppm"

    def test_to_dict_keys(self):
        r = SensorReading.create(SensorType.TEMPERATURE, 30.0)
        d = r.to_dict()
        assert {"sensor_type", "value", "unit", "status"} <= d.keys()


class TestStationEnvironment:
    """StationEnvironment.overall_status() の最悪値伝播"""

    def _make_env(self, statuses):
        """statuses: list of SensorStatus"""
        readings = []
        for i, st in enumerate(statuses):
            sensor = list(SensorType)[i % 4]
            # status を直接注入
            r = SensorReading(sensor_type=sensor, value=0.0, unit="", status=st)
            readings.append(r)
        return StationEnvironment(
            station_id="test-s",
            station_name="Test",
            city=CityName.TOKYO,
            readings=readings,
        )

    def test_all_normal_returns_normal(self):
        env = self._make_env([SensorStatus.NORMAL, SensorStatus.NORMAL])
        assert env.overall_status() == SensorStatus.NORMAL

    def test_one_warning_returns_warning(self):
        env = self._make_env([SensorStatus.NORMAL, SensorStatus.WARNING])
        assert env.overall_status() == SensorStatus.WARNING

    def test_one_critical_returns_critical(self):
        env = self._make_env([SensorStatus.NORMAL, SensorStatus.WARNING, SensorStatus.CRITICAL])
        assert env.overall_status() == SensorStatus.CRITICAL

    def test_empty_readings_returns_normal(self):
        env = StationEnvironment("x", "X", CityName.TOKYO)
        assert env.overall_status() == SensorStatus.NORMAL

    def test_get_reading_found(self):
        env = self._make_env([SensorStatus.NORMAL])
        r = env.get_reading(list(SensorType)[0])
        assert r is not None

    def test_get_reading_missing(self):
        env = StationEnvironment("x", "X", CityName.TOKYO)
        assert env.get_reading(SensorType.CO2) is None

    def test_to_dict_has_overall_status(self):
        env = self._make_env([SensorStatus.WARNING])
        d = env.to_dict()
        assert "overall_status" in d
        assert d["overall_status"] == "warning"


class TestEnvSensorAnalyzer:
    """EnvSensorAnalyzer の操作"""

    @pytest.fixture
    def analyzer(self):
        return EnvSensorDataset.build()

    def test_snapshot_known_station(self, analyzer):
        env = analyzer.snapshot("tokyo-shinjuku")
        assert env is not None
        assert env.station_id == "tokyo-shinjuku"

    def test_snapshot_unknown_station(self, analyzer):
        assert analyzer.snapshot("unknown-xxx") is None

    def test_compare_returns_result(self, analyzer):
        result = analyzer.compare(["tokyo-shinjuku", "tokyo-tokyo"])
        assert result.worst_station_id in ["tokyo-shinjuku", "tokyo-tokyo"]

    def test_compare_empty_list(self, analyzer):
        result = analyzer.compare([])
        assert result.worst_station_id is None

    def test_alert_stations_warning(self, analyzer):
        # データセット中に WARNING 以上の駅が存在する
        alerts = analyzer.alert_stations(min_status=SensorStatus.WARNING)
        assert len(alerts) >= 1

    def test_alert_stations_city_filter(self, analyzer):
        alerts = analyzer.alert_stations(city=CityName.TOKYO, min_status=SensorStatus.WARNING)
        for e in alerts:
            assert e.city == CityName.TOKYO

    def test_alert_stations_sorted_worst_first(self, analyzer):
        priority = {SensorStatus.CRITICAL: 2, SensorStatus.WARNING: 1, SensorStatus.NORMAL: 0}
        alerts = analyzer.alert_stations(min_status=SensorStatus.NORMAL)
        statuses = [priority[e.overall_status()] for e in alerts]
        assert statuses == sorted(statuses, reverse=True)


class TestEnvSensorDataset:
    """プリセットデータ検証"""

    @pytest.fixture(scope="class")
    def analyzer(self):
        return EnvSensorDataset.build()

    def test_16_stations_registered(self, analyzer):
        assert len(analyzer.all_station_ids()) == 16

    def test_hakata_co2_critical(self, analyzer):
        # 福岡-博多: co2=2100.0 → CRITICAL
        env = analyzer.snapshot("fukuoka-hakata")
        r = env.get_reading(SensorType.CO2)
        assert r.status == SensorStatus.CRITICAL

    def test_kasumigaseki_temp_normal(self, analyzer):
        # 霞ケ関: temp=25.5 → NORMAL
        env = analyzer.snapshot("tokyo-kasumigaseki")
        r = env.get_reading(SensorType.TEMPERATURE)
        assert r.status == SensorStatus.NORMAL

    def test_all_stations_have_4_readings(self, analyzer):
        for sid in analyzer.all_station_ids():
            env = analyzer.snapshot(sid)
            assert len(env.readings) == 4


# Sprint 75A API テスト

class TestEnvAPI:
    def test_env_snapshot_ok(self, client):
        r = client.get("/v1/env/tokyo-shinjuku/snapshot")
        assert r.status_code == 200
        d = r.json()
        assert d["station_id"] == "tokyo-shinjuku"
        assert "readings" in d

    def test_env_snapshot_not_found(self, client):
        r = client.get("/v1/env/unknown-xxx/snapshot")
        assert r.status_code == 404

    def test_env_compare_ok(self, client):
        r = client.get("/v1/env/compare", params={"stations": "tokyo-shinjuku,tokyo-tokyo"})
        assert r.status_code == 200
        d = r.json()
        assert "worst_station_id" in d

    def test_env_alerts_city_ok(self, client):
        r = client.get("/v1/env/tokyo/alerts")
        assert r.status_code == 200
        d = r.json()
        assert "alerts" in d

    def test_env_alerts_city_not_found(self, client):
        r = client.get("/v1/env/nowhere/alerts")
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════════
# Sprint 75B — 乗り換え最適化
# ═══════════════════════════════════════════════════════════════════

from open_mythos.skills.transfer_optimizer import (
    OptimizationWeight, TransferOptimizer, OptimizationDataset,
    WEIGHT_TIME_FIRST, WEIGHT_BALANCED, WEIGHT_ACCESSIBILITY,
    WEIGHT_CROWD_AVOIDANCE, DEFAULT_WEIGHT_PRESETS,
)


class TestOptimizationWeight:
    def test_default_sums_to_one(self):
        w = OptimizationWeight()
        assert abs(w.crowd_w + w.access_w + w.time_w - 1.0) < 1e-9

    def test_custom_valid(self):
        w = OptimizationWeight(0.5, 0.3, 0.2)
        assert abs(w.crowd_w + w.access_w + w.time_w - 1.0) < 1e-9

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError):
            OptimizationWeight(0.5, 0.5, 0.5)  # 合計1.5

    def test_presets_all_valid(self):
        for p in DEFAULT_WEIGHT_PRESETS:
            total = p.crowd_w + p.access_w + p.time_w
            assert abs(total - 1.0) < 1e-9

    def test_time_first_label(self):
        assert WEIGHT_TIME_FIRST.label == "time_first"

    def test_to_dict_keys(self):
        d = WEIGHT_BALANCED.to_dict()
        assert {"label", "crowd_w", "access_w", "time_w"} <= d.keys()


class TestTransferOptimizer:
    @pytest.fixture(scope="class")
    def dataset(self):
        return OptimizationDataset.build()

    @pytest.fixture(scope="class")
    def optimizer(self, dataset):
        return dataset.optimizer()

    def test_optimize_returns_list(self, optimizer):
        # 東京シリーズ内の経路
        opts = optimizer.optimize("tokyo-shinjuku", "tokyo-ginza", hour=8)
        assert isinstance(opts, list)

    def test_optimize_found_route_returns_options(self, optimizer):
        opts = optimizer.optimize("tokyo-shinjuku", "tokyo-tokyo", hour=8)
        # found なら DEFAULT_WEIGHT_PRESETS 数のオプション
        if opts:
            assert len(opts) == len(DEFAULT_WEIGHT_PRESETS)

    def test_optimize_sorted_by_total_cost(self, optimizer):
        opts = optimizer.optimize("tokyo-shinjuku", "tokyo-ginza", hour=8)
        if len(opts) >= 2:
            costs = [o.total_cost for o in opts]
            assert costs == sorted(costs)

    def test_optimize_costs_in_range(self, optimizer):
        opts = optimizer.optimize("tokyo-shinjuku", "tokyo-tokyo", hour=8)
        for o in opts:
            assert 0.0 <= o.crowd_cost <= 1.0
            assert 0.0 <= o.access_cost <= 1.0
            assert 0.0 <= o.time_cost <= 1.0

    def test_optimize_unknown_stations_returns_empty(self, optimizer):
        opts = optimizer.optimize("nowhere-a", "nowhere-b", hour=8)
        assert opts == []

    def test_score_route_custom_weight(self, optimizer):
        w = OptimizationWeight(0.6, 0.2, 0.2, "heavy_crowd")
        result = optimizer.score_route("tokyo-shinjuku", "tokyo-tokyo", hour=8, weight=w)
        # found なら結果あり、なければ None
        if result is not None:
            assert result.label == "heavy_crowd"
            assert abs(result.total_cost -
                       (0.6 * result.crowd_cost + 0.2 * result.access_cost + 0.2 * result.time_cost)
                       ) < 1e-9

    def test_to_dict_keys(self, optimizer):
        opts = optimizer.optimize("tokyo-shinjuku", "tokyo-ginza", hour=8)
        if opts:
            d = opts[0].to_dict()
            assert {"label", "weight", "path_ids", "crowd_cost", "access_cost",
                    "time_cost", "total_cost"} <= d.keys()


# Sprint 75B API テスト

class TestTransferAPI:
    def test_optimize_ok(self, client):
        r = client.get("/v1/transfer/optimize", params={
            "from_id": "tokyo-shinjuku", "to_id": "tokyo-ginza", "hour": 8
        })
        assert r.status_code == 200
        d = r.json()
        assert "options" in d

    def test_optimize_unknown_route(self, client):
        r = client.get("/v1/transfer/optimize", params={
            "from_id": "nowhere-a", "to_id": "nowhere-b", "hour": 8
        })
        assert r.status_code == 200
        assert r.json()["options"] == []

    def test_score_ok(self, client):
        r = client.get("/v1/transfer/score", params={
            "from_id": "tokyo-shinjuku", "to_id": "tokyo-tokyo",
            "hour": 8, "crowd_w": 0.5, "access_w": 0.3, "time_w": 0.2
        })
        assert r.status_code == 200

    def test_score_invalid_weight(self, client):
        r = client.get("/v1/transfer/score", params={
            "from_id": "tokyo-shinjuku", "to_id": "tokyo-tokyo",
            "hour": 8, "crowd_w": 0.9, "access_w": 0.9, "time_w": 0.9
        })
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════
# Sprint 75C — 都市インフラダッシュボード
# ═══════════════════════════════════════════════════════════════════

from open_mythos.skills.infra_dashboard import (
    MetricStatus, DashboardMetric, StationPanel, CityDashboard,
    InfraDashboard, DashboardDataset,
    _crowd_status, _access_status, _flood_status,
)
from open_mythos.skills.groundwater import FloodRiskLevel


class TestMetricStatus:
    """ステータス変換ヘルパー"""

    def test_crowd_good(self):
        assert _crowd_status(0.8) == MetricStatus.GOOD

    def test_crowd_warn(self):
        assert _crowd_status(1.2) == MetricStatus.WARN

    def test_crowd_alert(self):
        assert _crowd_status(1.6) == MetricStatus.ALERT

    def test_access_good(self):
        assert _access_status(80.0) == MetricStatus.GOOD

    def test_access_warn(self):
        assert _access_status(45.0) == MetricStatus.WARN

    def test_access_alert(self):
        assert _access_status(20.0) == MetricStatus.ALERT

    def test_flood_good(self):
        assert _flood_status(FloodRiskLevel.LOW) == MetricStatus.GOOD

    def test_flood_warn(self):
        assert _flood_status(FloodRiskLevel.MODERATE) == MetricStatus.WARN

    def test_flood_alert(self):
        assert _flood_status(FloodRiskLevel.HIGH) == MetricStatus.ALERT

    def test_flood_very_high_alert(self):
        assert _flood_status(FloodRiskLevel.VERY_HIGH) == MetricStatus.ALERT


class TestStationPanel:
    def _panel(self, statuses):
        metrics = [
            DashboardMetric("m", 0, "", st) for st in statuses
        ]
        return StationPanel("s1", metrics)

    def test_worst_status_all_good(self):
        p = self._panel([MetricStatus.GOOD, MetricStatus.GOOD])
        assert p.worst_status == MetricStatus.GOOD

    def test_worst_status_with_warn(self):
        p = self._panel([MetricStatus.GOOD, MetricStatus.WARN])
        assert p.worst_status == MetricStatus.WARN

    def test_worst_status_with_alert(self):
        p = self._panel([MetricStatus.GOOD, MetricStatus.WARN, MetricStatus.ALERT])
        assert p.worst_status == MetricStatus.ALERT

    def test_to_dict_keys(self):
        p = self._panel([MetricStatus.GOOD])
        d = p.to_dict()
        assert {"station_id", "worst_status", "metrics"} <= d.keys()


class TestCityDashboard:
    def _make_dashboard(self, worst_statuses):
        panels = []
        for i, st in enumerate(worst_statuses):
            m = DashboardMetric("m", 0, "", st)
            panels.append(StationPanel(f"s{i}", [m]))
        return CityDashboard("tokyo", panels, "2026-01-01T00:00:00+00:00")

    def test_summary_counts_correctly(self):
        db = self._make_dashboard([
            MetricStatus.ALERT, MetricStatus.WARN, MetricStatus.GOOD, MetricStatus.GOOD
        ])
        s = db.summary()
        assert s["alert_count"] == 1
        assert s["warn_count"] == 1
        assert s["ok_count"] == 2
        assert s["total"] == 4

    def test_summary_all_ok(self):
        db = self._make_dashboard([MetricStatus.GOOD] * 5)
        s = db.summary()
        assert s["alert_count"] == 0
        assert s["ok_count"] == 5

    def test_to_dict_has_summary(self):
        db = self._make_dashboard([MetricStatus.GOOD])
        d = db.to_dict()
        assert "summary" in d
        assert "panels" in d


class TestInfraDashboard:
    @pytest.fixture(scope="class")
    def dashboard(self):
        ds = DashboardDataset.build()
        return ds.dashboard()

    def test_city_panel_returns_dashboard(self, dashboard):
        db = dashboard.city_panel(CityName.TOKYO, hour=8)
        assert db.city == "tokyo"
        assert len(db.panels) > 0

    def test_city_panel_has_3_metrics_per_station(self, dashboard):
        db = dashboard.city_panel(CityName.TOKYO, hour=8)
        for panel in db.panels:
            # 混雑・アクセシビリティ・浸水リスク = 3指標
            assert len(panel.metrics) == 3

    def test_alert_stations_returns_subset(self, dashboard):
        all_panels = dashboard.city_panel(CityName.TOKYO, hour=8).panels
        alert_panels = dashboard.alert_stations(CityName.TOKYO, hour=8)
        assert len(alert_panels) <= len(all_panels)

    def test_multi_city_summary_all_cities(self, dashboard):
        cities = [CityName.TOKYO, CityName.OSAKA, CityName.FUKUOKA]
        result = dashboard.multi_city_summary(cities, hour=8)
        assert len(result) == 3
        for row in result:
            assert "city" in row
            assert "total" in row

    def test_generated_at_is_iso(self, dashboard):
        db = dashboard.city_panel(CityName.OSAKA, hour=9)
        assert "T" in db.generated_at


# Sprint 75C API テスト (/v1/infra/ プレフィックス)

class TestDashboardAPI:
    def test_infra_city_ok(self, client):
        r = client.get("/v1/infra/tokyo", params={"hour": 8})
        assert r.status_code == 200
        d = r.json()
        assert d["city"] == "tokyo"
        assert "panels" in d
        assert "summary" in d

    def test_infra_city_not_found(self, client):
        r = client.get("/v1/infra/nowhere", params={"hour": 8})
        assert r.status_code == 404

    def test_infra_alerts_ok(self, client):
        r = client.get("/v1/infra/tokyo/alerts", params={"hour": 8})
        assert r.status_code == 200
        d = r.json()
        assert "alert_count" in d
        assert "panels" in d

    def test_infra_summary_ok(self, client):
        r = client.get("/v1/infra/summary", params={"cities": "tokyo,osaka", "hour": 8})
        assert r.status_code == 200
        d = r.json()
        assert len(d["cities"]) == 2

    def test_infra_summary_unknown_city_skipped(self, client):
        r = client.get("/v1/infra/summary", params={"cities": "tokyo,nowhere", "hour": 8})
        assert r.status_code == 200
        d = r.json()
        # "nowhere" はスキップされ tokyo のみ
        assert len(d["cities"]) == 1
