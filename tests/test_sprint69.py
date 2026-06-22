"""
Sprint 69 — 時系列予測統合 (TimesFM + マルチモデル) テスト

対象:
  open_mythos/skills/time_series.py
    — ForecastPoint / ForecastResult
    — LinearTrendForecaster / MockForecaster / TimesFMForecaster
    — TimesFMForecasterFactory
    — CampaignForecaster / ForecastStore / ForecastReportEngine
  serve/api.py
    — /v1/forecast/* エンドポイント
"""
from __future__ import annotations

import pytest

from open_mythos.skills.time_series import (
    ForecastPoint, ForecastResult,
    LinearTrendForecaster, MockForecaster, TimesFMForecaster,
    TimesFMForecasterFactory,
    CampaignForecaster, ForecastStore, ForecastReportEngine,
)
from open_mythos.skills.campaign_analytics import CampaignAnalyticsStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ヘルパ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _analytics_store_with_data(campaign_id: str = "c1", n: int = 10) -> CampaignAnalyticsStore:
    store = CampaignAnalyticsStore()
    m = store.get_or_create(campaign_id)
    for i in range(n):
        m.record(clicks=100 + i * 5, impressions=1000 + i * 20, spend=10.0, revenue=50.0)
    return store


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ForecastPoint / ForecastResult
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestForecastPoint:
    def test_to_dict_keys(self):
        p = ForecastPoint(step=1, value=100.0, lower=90.0, upper=110.0)
        d = p.to_dict()
        assert "step" in d and "value" in d and "lower" in d and "upper" in d

    def test_none_bounds_in_dict(self):
        p = ForecastPoint(step=1, value=100.0)
        d = p.to_dict()
        assert d["lower"] is None and d["upper"] is None


class TestForecastResult:
    def _result(self) -> ForecastResult:
        fc = LinearTrendForecaster()
        return fc.forecast([10, 20, 30, 40, 50], horizon=3)

    def test_horizon_correct(self):
        r = self._result()
        assert r.horizon == 3
        assert len(r.points) == 3

    def test_values_property(self):
        r = self._result()
        assert len(r.values) == 3

    def test_mean_value(self):
        r = self._result()
        assert r.mean_value == pytest.approx(sum(r.values) / 3, rel=1e-5)

    def test_to_dict_keys(self):
        d = self._result().to_dict()
        for k in ("forecast_id", "campaign_id", "metric", "model", "horizon", "points"):
            assert k in d

    def test_to_dict_points_list(self):
        d = self._result().to_dict()
        assert isinstance(d["points"], list)
        assert len(d["points"]) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LinearTrendForecaster
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLinearTrendForecaster:
    def setup_method(self):
        self.fc = LinearTrendForecaster()

    def test_model_name(self):
        assert self.fc.model_name == "linear_trend"

    def test_basic_forecast(self):
        result = self.fc.forecast([10, 20, 30, 40, 50], horizon=3)
        assert result.horizon == 3
        assert len(result.points) == 3

    def test_upward_trend(self):
        result = self.fc.forecast([10, 20, 30, 40, 50], horizon=3)
        # 上昇トレンドなので予測値は 50 以上
        assert result.points[0].value > 45

    def test_confidence_interval_present(self):
        result = self.fc.forecast([10, 20, 30, 40, 50], horizon=3)
        for p in result.points:
            assert p.lower is not None
            assert p.upper is not None

    def test_lower_lt_value_lt_upper(self):
        result = self.fc.forecast([10, 20, 30, 40, 50], horizon=3)
        for p in result.points:
            assert p.lower <= p.value <= p.upper

    def test_minimum_2_points_required(self):
        with pytest.raises(ValueError):
            self.fc.forecast([10], horizon=3)

    def test_flat_series(self):
        result = self.fc.forecast([100, 100, 100, 100], horizon=3)
        for p in result.points:
            assert abs(p.value - 100) < 1.0

    def test_horizon_parameter(self):
        result = self.fc.forecast([10, 20, 30], horizon=10)
        assert len(result.points) == 10

    def test_campaign_id_and_metric_set(self):
        result = self.fc.forecast([1, 2, 3], 2, campaign_id="c1", metric="clicks")
        assert result.campaign_id == "c1"
        assert result.metric == "clicks"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MockForecaster
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMockForecaster:
    def test_model_name(self):
        assert MockForecaster().model_name == "mock"

    def test_returns_mock_values(self):
        fc = MockForecaster(mock_values=[100.0, 200.0, 300.0])
        result = fc.forecast([1, 2, 3], horizon=3)
        assert result.values == [100.0, 200.0, 300.0]

    def test_repeats_when_short(self):
        fc = MockForecaster(mock_values=[50.0])
        result = fc.forecast([1, 2], horizon=4)
        assert len(result.points) == 4

    def test_none_falls_back_to_linear(self):
        fc = MockForecaster(mock_values=None)
        result = fc.forecast([10, 20, 30], horizon=3)
        assert result.model == "mock"
        assert len(result.points) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TimesFMForecaster (モデルロードなしでフォールバック動作を検証)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTimesFMForecaster:
    def setup_method(self):
        # 存在しない repo_id を使い、強制的にフォールバックへ
        self.fc = TimesFMForecaster(repo_id="nonexistent/repo-for-test")

    def test_model_name(self):
        assert "timesfm" in self.fc.model_name

    def test_not_loaded_initially(self):
        assert self.fc.is_loaded is False

    def test_fallback_forecast_works(self):
        """ロード失敗でもフォールバックで結果が返る"""
        result = self.fc.forecast([10, 20, 30, 40], horizon=3)
        assert len(result.points) == 3

    def test_fallback_model_name_contains_fallback(self):
        result = self.fc.forecast([10, 20, 30, 40], horizon=3)
        assert "fallback" in result.model

    def test_minimum_2_points(self):
        with pytest.raises(ValueError):
            self.fc.forecast([10], horizon=3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TimesFMForecasterFactory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTimesFMForecasterFactory:
    def test_rule_based_returns_linear(self):
        fc = TimesFMForecasterFactory.rule_based()
        assert fc.model_name == "linear_trend"

    def test_from_mock_returns_mock(self):
        fc = TimesFMForecasterFactory.from_mock([10.0, 20.0])
        assert fc.model_name == "mock"

    def test_from_pretrained_returns_timesfm(self):
        fc = TimesFMForecasterFactory.from_pretrained()
        assert "timesfm" in fc.model_name

    def test_available_models_list(self):
        models = TimesFMForecasterFactory.available_models()
        assert isinstance(models, list)
        assert len(models) >= 2
        names = [m["name"] for m in models]
        assert "timesfm-2.5-200m" in names
        assert "linear_trend" in names

    def test_available_models_has_required_keys(self):
        for m in TimesFMForecasterFactory.available_models():
            for k in ("name", "type", "description", "requires"):
                assert k in m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CampaignForecaster
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCampaignForecaster:
    def setup_method(self):
        self.store = _analytics_store_with_data("camp1", n=10)
        self.fc    = CampaignForecaster(LinearTrendForecaster(), self.store)

    def test_forecast_metric_clicks(self):
        result = self.fc.forecast_metric("camp1", metric="clicks", horizon=5)
        assert result.horizon == 5
        assert result.campaign_id == "camp1"
        assert result.metric == "clicks"

    def test_forecast_result_has_points(self):
        result = self.fc.forecast_metric("camp1", metric="clicks", horizon=7)
        assert len(result.points) == 7

    def test_forecast_campaign_not_found(self):
        with pytest.raises(ValueError, match="not found"):
            self.fc.forecast_metric("nonexistent", metric="clicks", horizon=3)

    def test_forecast_insufficient_data(self):
        store = CampaignAnalyticsStore()
        m = store.get_or_create("c_tiny")
        m.record(clicks=10)   # 1 点だけ
        fc = CampaignForecaster(LinearTrendForecaster(), store)
        with pytest.raises(ValueError, match="Insufficient"):
            fc.forecast_metric("c_tiny", metric="clicks", horizon=3)

    def test_forecast_all_metrics(self):
        results = self.fc.forecast_all_metrics("camp1", horizon=3)
        assert isinstance(results, dict)
        assert "clicks" in results
        assert "impressions" in results

    def test_forecast_batch(self):
        store = _analytics_store_with_data("b1", n=5)
        _analytics_store_with_data("b2", n=5)   # 別インスタンスなので b2 は入らない
        # 同一 store に b2 追加
        m2 = store.get_or_create("b2")
        for i in range(5):
            m2.record(clicks=50 + i)
        fc = CampaignForecaster(LinearTrendForecaster(), store)
        results = fc.forecast_batch(["b1", "b2"], metric="clicks", horizon=3)
        assert "b1" in results
        assert "b2" in results

    def test_forecast_batch_skips_missing(self):
        results = self.fc.forecast_batch(["camp1", "no-such"], metric="clicks", horizon=3)
        assert "camp1" in results
        assert "no-such" not in results

    def test_upward_trend_predicted(self):
        """上昇トレンドデータなら予測値が最後の実績より大きい"""
        store = CampaignAnalyticsStore()
        m = store.get_or_create("trend_up")
        for i in range(10):
            m.record(clicks=100 + i * 10)
        fc = CampaignForecaster(LinearTrendForecaster(), store)
        result = fc.forecast_metric("trend_up", metric="clicks", horizon=3)
        last_actual = 100 + 9 * 10   # = 190
        assert result.points[0].value > last_actual * 0.9


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ForecastStore / ForecastReportEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestForecastStore:
    def setup_method(self):
        self.store = ForecastStore()
        self.fc    = LinearTrendForecaster()

    def _result(self, campaign_id="c1", metric="clicks") -> ForecastResult:
        r = self.fc.forecast([10, 20, 30, 40], horizon=3, campaign_id=campaign_id, metric=metric)
        return r

    def test_save_and_get(self):
        r = self._result()
        self.store.save(r)
        assert self.store.get(r.forecast_id) is not None

    def test_get_missing_returns_none(self):
        assert self.store.get("nope") is None

    def test_list_by_campaign(self):
        self.store.save(self._result("c1"))
        self.store.save(self._result("c2"))
        assert len(self.store.list_by_campaign("c1")) == 1

    def test_list_by_metric(self):
        self.store.save(self._result(metric="clicks"))
        self.store.save(self._result(metric="spend"))
        assert len(self.store.list_by_metric("clicks")) == 1

    def test_latest_returns_newest(self):
        self.store.save(self._result())
        import time; time.sleep(0.01)
        r2 = self._result()
        self.store.save(r2)
        latest = self.store.latest("c1", "clicks")
        assert latest is not None
        assert latest.forecast_id == r2.forecast_id

    def test_count(self):
        self.store.save(self._result())
        self.store.save(self._result())
        assert self.store.count() == 2


class TestForecastReportEngine:
    def setup_method(self):
        self.store  = ForecastStore()
        self.engine = ForecastReportEngine(self.store)
        fc = LinearTrendForecaster()
        r = fc.forecast([10, 20, 30, 40, 50], horizon=5, campaign_id="c1", metric="clicks")
        self.store.save(r)
        self.result = r

    def test_summary_json_keys(self):
        d = self.engine.summary_json()
        assert "total_forecasts" in d
        assert "by_model" in d

    def test_markdown_contains_campaign_id(self):
        md = self.engine.markdown(self.result)
        assert "c1" in md

    def test_markdown_contains_metric(self):
        md = self.engine.markdown(self.result)
        assert "clicks" in md

    def test_markdown_contains_table(self):
        md = self.engine.markdown(self.result)
        assert "ステップ" in md


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API テスト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture(autouse=True)
def _reset_rl():
    """各テスト前後にレートリミッターをリセットしてテスト間干渉を防ぐ"""
    try:
        from serve.auth import _rate_limiter
        _rate_limiter.reset_all()
    except ImportError:
        pass
    yield
    try:
        from serve.auth import _rate_limiter
        _rate_limiter.reset_all()
    except ImportError:
        pass


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from serve.api import app
    return TestClient(app)


def _setup_campaign(client, campaign_id: str = "fc-camp", n: int = 10):
    """API 経由でキャンペーンデータを用意する"""
    for i in range(n):
        client.post(f"/v1/analytics/{campaign_id}/record", json={
            "clicks": 100 + i * 5,
            "impressions": 1000 + i * 20,
            "spend": 10.0,
            "revenue": 50.0,
        })


class TestForecastModelsApi:
    def test_models_list(self, client):
        resp = client.get("/v1/forecast/models")
        assert resp.status_code == 200
        d = resp.json()
        assert "models" in d
        assert len(d["models"]) >= 2

    def test_models_has_timesfm(self, client):
        models = client.get("/v1/forecast/models").json()["models"]
        names = [m["name"] for m in models]
        assert "timesfm-2.5-200m" in names


class TestForecastCampaignApi:
    def test_forecast_linear(self, client):
        _setup_campaign(client, "fc1")
        resp = client.post("/v1/forecast/fc1", json={
            "metric": "clicks", "horizon": 5, "model": "linear_trend"
        })
        assert resp.status_code == 200
        d = resp.json()
        assert d["horizon"] == 5
        assert len(d["points"]) == 5

    def test_forecast_mock_model(self, client):
        _setup_campaign(client, "fc2")
        resp = client.post("/v1/forecast/fc2", json={
            "metric": "impressions", "horizon": 3, "model": "mock"
        })
        assert resp.status_code == 200
        assert resp.json()["model"] == "mock"

    def test_forecast_not_found(self, client):
        resp = client.post("/v1/forecast/no-such-camp", json={
            "metric": "clicks", "horizon": 3
        })
        assert resp.status_code == 404

    def test_forecast_all_metrics(self, client):
        _setup_campaign(client, "fc3")
        resp = client.post("/v1/forecast/fc3/all", json={"horizon": 3})
        assert resp.status_code == 200
        d = resp.json()
        assert "forecasts" in d
        assert "clicks" in d["forecasts"]

    def test_forecast_result_keys(self, client):
        _setup_campaign(client, "fc4")
        resp = client.post("/v1/forecast/fc4", json={"metric": "clicks", "horizon": 5})
        d = resp.json()
        for k in ("forecast_id", "campaign_id", "metric", "model", "horizon", "points"):
            assert k in d

    def test_forecast_points_step_sequence(self, client):
        _setup_campaign(client, "fc5")
        resp = client.post("/v1/forecast/fc5", json={"metric": "clicks", "horizon": 4})
        points = resp.json()["points"]
        steps = [p["step"] for p in points]
        assert steps == [1, 2, 3, 4]


class TestForecastBatchApi:
    def test_batch_forecast(self, client):
        _setup_campaign(client, "b1")
        _setup_campaign(client, "b2")
        resp = client.post("/v1/forecast/batch", json={
            "campaign_ids": ["b1", "b2"],
            "metric": "clicks",
            "horizon": 3,
        })
        assert resp.status_code == 200
        d = resp.json()
        assert "b1" in d["forecasts"]
        assert "b2" in d["forecasts"]

    def test_batch_skips_missing(self, client):
        _setup_campaign(client, "bx")
        resp = client.post("/v1/forecast/batch", json={
            "campaign_ids": ["bx", "no-such-zzz"],
            "metric": "clicks",
            "horizon": 3,
        })
        assert resp.status_code == 200
        d = resp.json()
        assert "bx" in d["forecasts"]
        assert "no-such-zzz" not in d["forecasts"]


class TestForecastHistoryApi:
    def test_history_after_forecast(self, client):
        _setup_campaign(client, "hist1")
        client.post("/v1/forecast/hist1", json={"metric": "clicks", "horizon": 3})
        resp = client.get("/v1/forecast/hist1/history", params={"metric": "clicks"})
        assert resp.status_code == 200
        d = resp.json()
        assert d["campaign_id"] == "hist1"

    def test_history_not_found(self, client):
        resp = client.get("/v1/forecast/hist-nope/history", params={"metric": "clicks"})
        assert resp.status_code == 404

    def test_report_md(self, client):
        _setup_campaign(client, "rpt1")
        fid = client.post("/v1/forecast/rpt1", json={"metric": "clicks", "horizon": 3}).json()["forecast_id"]
        resp = client.get(f"/v1/forecast/report/md/{fid}")
        assert resp.status_code == 200
        assert "rpt1" in resp.text

    def test_report_md_not_found(self, client):
        resp = client.get("/v1/forecast/report/md/nonexistent-id")
        assert resp.status_code == 404
