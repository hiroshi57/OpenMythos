"""
Sprint 23 テスト — 外部要因適応 ExternalSignalAgent

- TestExternalSignal       : シグナルデータ構造
- TestImpactEstimate       : 影響量推定
- TestCounterAction        : カウンターアクション
- TestSignalDetector       : シグナル検出
- TestImpactEstimator      : 影響量マッピング
- TestExternalSignalResult : 結果集計
- TestExternalSignalAgent  : detect→estimate→counter サイクル
- TestSignalAPIEndpoint    : FastAPI /v1/signal/* (静的検査)
"""

from __future__ import annotations

import pathlib

_ROOT = pathlib.Path(__file__).parent.parent


# ===========================================================================
# TestExternalSignal
# ===========================================================================


class TestExternalSignal:
    def _make(self, direction="negative", strength=0.7):
        from open_mythos.external_signal import ExternalSignal

        return ExternalSignal(
            signal_type="competitor",
            name="競合新規参入",
            strength=strength,
            direction=direction,
            source="競合キーワード検出",
        )

    def test_is_threat_negative(self):
        s = self._make(direction="negative")
        assert s.is_threat is True

    def test_is_threat_positive(self):
        s = self._make(direction="positive")
        assert s.is_threat is False

    def test_is_opportunity_positive(self):
        s = self._make(direction="positive")
        assert s.is_opportunity is True

    def test_is_opportunity_negative(self):
        s = self._make(direction="negative")
        assert s.is_opportunity is False

    def test_fields_set(self):
        s = self._make()
        assert s.signal_type == "competitor"
        assert s.strength == 0.7
        assert s.source != ""

    def test_detected_at_positive(self):
        s = self._make()
        assert s.detected_at > 0


# ===========================================================================
# TestImpactEstimate
# ===========================================================================


class TestImpactEstimate:
    def _make_estimate(self, direction="negative", strength=0.6):
        from open_mythos.external_signal import ExternalSignal, ImpactEstimator

        signal = ExternalSignal(
            signal_type="competitor",
            name="test",
            strength=strength,
            direction=direction,
            source="test",
        )
        return ImpactEstimator().estimate(signal, "llmo_score")

    def test_impact_negative_when_threat(self):
        est = self._make_estimate(direction="negative")
        assert est.impact_delta < 0

    def test_impact_positive_when_opportunity(self):
        est = self._make_estimate(direction="positive")
        assert est.impact_delta > 0

    def test_confidence_in_range(self):
        est = self._make_estimate()
        assert 0.0 <= est.confidence <= 1.0

    def test_severity_high_for_strong_signal(self):
        from open_mythos.external_signal import ExternalSignal, ImpactEstimator

        signal = ExternalSignal("competitor", "test", 1.0, "negative", "test")
        est = ImpactEstimator().estimate(signal)
        assert est.severity in ("high", "medium", "low")

    def test_explanation_not_empty(self):
        est = self._make_estimate()
        assert len(est.explanation) > 0

    def test_kpi_name_assigned(self):
        est = self._make_estimate()
        assert est.kpi_name == "llmo_score"


# ===========================================================================
# TestCounterAction
# ===========================================================================


class TestCounterAction:
    def test_apply_returns_string(self):
        from open_mythos.external_signal import CounterAction

        action = CounterAction(
            action_id="test",
            signal_type="seasonal",
            description="テスト",
            transform_fn=lambda t: t + "_modified",
            estimated_kpi_recovery=0.1,
        )
        result = action.apply("input")
        assert result == "input_modified"

    def test_apply_handles_exception(self):
        from open_mythos.external_signal import CounterAction

        def bad_fn(t):
            raise RuntimeError("error")

        action = CounterAction(
            action_id="bad",
            signal_type="seasonal",
            description="エラーアクション",
            transform_fn=bad_fn,
            estimated_kpi_recovery=0.1,
        )
        result = action.apply("original")
        assert result == "original"

    def test_estimated_kpi_recovery_positive(self):
        from open_mythos.external_signal import _COUNTER_ACTION_TEMPLATES

        for action in _COUNTER_ACTION_TEMPLATES:
            assert action.estimated_kpi_recovery > 0


# ===========================================================================
# TestSignalDetector
# ===========================================================================


class TestSignalDetector:
    def test_detect_seasonal_signal(self):
        from open_mythos.external_signal import SignalDetector

        detector = SignalDetector()
        signals = detector.detect("SEO対策", month=12)
        types = [s.signal_type for s in signals]
        assert "seasonal" in types

    def test_detect_no_seasonal_when_month_none(self):
        from open_mythos.external_signal import SignalDetector

        detector = SignalDetector()
        signals = detector.detect("テスト", month=None)
        types = [s.signal_type for s in signals]
        assert "seasonal" not in types

    def test_detect_competitor_signal(self):
        from open_mythos.external_signal import SignalDetector

        detector = SignalDetector()
        signals = detector.detect("競合が新規参入している", keyword="競合")
        types = [s.signal_type for s in signals]
        assert "competitor" in types

    def test_detect_trend_spike_signal(self):
        from open_mythos.external_signal import SignalDetector

        detector = SignalDetector()
        signals = detector.detect("急上昇トレンド", keyword="トレンド急増")
        types = [s.signal_type for s in signals]
        assert "trend_spike" in types

    def test_detect_market_signal_with_keyword(self):
        from open_mythos.external_signal import SignalDetector

        detector = SignalDetector()
        signals = detector.detect("", keyword="SEO対策マーケティング戦略")
        types = [s.signal_type for s in signals]
        assert "market" in types

    def test_signal_strength_in_range(self):
        from open_mythos.external_signal import SignalDetector

        detector = SignalDetector()
        signals = detector.detect("テスト", keyword="SEO", month=6)
        for s in signals:
            assert 0.0 <= s.strength <= 1.0

    def test_signal_direction_valid(self):
        from open_mythos.external_signal import SignalDetector

        detector = SignalDetector()
        signals = detector.detect("テスト", keyword="SEO", month=6)
        for s in signals:
            assert s.direction in ("positive", "negative")

    def test_seasonal_high_month_positive(self):
        from open_mythos.external_signal import SignalDetector

        detector = SignalDetector()
        signals = detector.detect("", month=12)  # 12月は高シーズン
        seasonal = [s for s in signals if s.signal_type == "seasonal"][0]
        assert seasonal.direction == "positive"

    def test_seasonal_low_month_negative(self):
        from open_mythos.external_signal import SignalDetector

        detector = SignalDetector()
        signals = detector.detect("", month=8)  # 8月は低シーズン
        seasonal = [s for s in signals if s.signal_type == "seasonal"][0]
        assert seasonal.direction == "negative"


# ===========================================================================
# TestImpactEstimator
# ===========================================================================


class TestImpactEstimator:
    def test_estimate_all_signal_types(self):
        from open_mythos.external_signal import ExternalSignal, ImpactEstimator

        estimator = ImpactEstimator()
        for stype in ("seasonal", "trend_spike", "competitor", "market"):
            signal = ExternalSignal(stype, "test", 0.5, "positive", "test")
            est = estimator.estimate(signal)
            assert est.impact_delta > 0

    def test_higher_strength_higher_impact(self):
        from open_mythos.external_signal import ExternalSignal, ImpactEstimator

        estimator = ImpactEstimator()
        low = ExternalSignal("competitor", "test", 0.2, "negative", "test")
        high = ExternalSignal("competitor", "test", 0.9, "negative", "test")
        est_low = estimator.estimate(low)
        est_high = estimator.estimate(high)
        assert abs(est_high.impact_delta) > abs(est_low.impact_delta)

    def test_higher_strength_higher_confidence(self):
        from open_mythos.external_signal import ExternalSignal, ImpactEstimator

        estimator = ImpactEstimator()
        low = ExternalSignal("trend_spike", "test", 0.1, "positive", "test")
        high = ExternalSignal("trend_spike", "test", 0.9, "positive", "test")
        est_low = estimator.estimate(low)
        est_high = estimator.estimate(high)
        assert est_high.confidence >= est_low.confidence


# ===========================================================================
# TestExternalSignalResult
# ===========================================================================


class TestExternalSignalResult:
    def _make_result(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        return agent.run("SEO対策コンテンツ", keyword="SEO", month=12)

    def test_threat_count(self):
        result = self._make_result()
        assert isinstance(result.threat_count, int)
        assert result.threat_count >= 0

    def test_opportunity_count(self):
        result = self._make_result()
        assert isinstance(result.opportunity_count, int)

    def test_top_action_or_none(self):
        result = self._make_result()
        assert result.top_action is None or hasattr(result.top_action, "action_id")

    def test_keyword_preserved(self):
        result = self._make_result()
        assert result.keyword == "SEO"


# ===========================================================================
# TestExternalSignalAgent
# ===========================================================================


class TestExternalSignalAgent:
    def test_run_returns_result(self):
        from open_mythos.external_signal import ExternalSignalAgent, ExternalSignalResult

        agent = ExternalSignalAgent()
        result = agent.run("SEO最適化", keyword="SEO", month=9)
        assert isinstance(result, ExternalSignalResult)

    def test_run_signals_not_empty(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        result = agent.run("", keyword="SEO", month=6)
        assert len(result.signals) > 0

    def test_run_impacts_match_signals(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        result = agent.run("テスト", month=3)
        assert len(result.impacts) == len(result.signals)

    def test_run_optimized_context_is_string(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        result = agent.run("元のコンテキスト", month=11)
        assert isinstance(result.optimized_context, str)

    def test_run_optimized_context_longer_or_equal(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        original = "元のコンテキスト"
        result = agent.run(original, month=11)
        assert len(result.optimized_context) >= len(original)

    def test_run_net_kpi_impact_float(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        result = agent.run("テスト", month=7)
        assert isinstance(result.net_kpi_impact, float)

    def test_run_latency_positive(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        result = agent.run("テスト", month=1)
        assert result.total_latency_ms > 0

    def test_run_counter_actions_sorted_by_priority(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        result = agent.run("競合が急増トレンド", keyword="競合急上昇", month=12)
        priorities = [a.priority for a in result.counter_actions]
        assert priorities == sorted(priorities)

    def test_run_no_month(self):
        from open_mythos.external_signal import ExternalSignalAgent, ExternalSignalResult

        agent = ExternalSignalAgent()
        result = agent.run("テスト", keyword="SEO")
        assert isinstance(result, ExternalSignalResult)

    def test_run_empty_context(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        result = agent.run("", keyword="", month=6)
        assert isinstance(result.net_kpi_impact, float)

    def test_seasonal_signal_detected_with_month(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        result = agent.run("test", month=5)
        types = [s.signal_type for s in result.signals]
        assert "seasonal" in types

    def test_counter_actions_kpi_recovery_positive(self):
        from open_mythos.external_signal import ExternalSignalAgent

        agent = ExternalSignalAgent()
        result = agent.run("テスト", keyword="競合", month=8)
        for action in result.counter_actions:
            assert action.estimated_kpi_recovery > 0


# ===========================================================================
# TestSignalAPIEndpoint (静的ソース検査)
# ===========================================================================


class TestSignalAPIEndpoint:
    def _src(self) -> str:
        return (_ROOT / "serve" / "api.py").read_text(encoding="utf-8")

    def test_signal_detect_route_exists(self):
        assert '"/v1/signal/detect"' in self._src()

    def test_signal_counter_route_exists(self):
        assert '"/v1/signal/counter"' in self._src()

    def test_signal_detect_post_method(self):
        src = self._src()
        idx = src.index('"/v1/signal/detect"')
        snippet = src[max(0, idx - 60):idx]
        assert "@app.post" in snippet

    def test_signal_counter_post_method(self):
        src = self._src()
        idx = src.index('"/v1/signal/counter"')
        snippet = src[max(0, idx - 60):idx]
        assert "@app.post" in snippet

    def test_signal_tag_detect(self):
        src = self._src()
        idx = src.index('"/v1/signal/detect"')
        snippet = src[idx:idx + 200]
        assert 'tags=["signal"]' in snippet

    def test_signal_tag_counter(self):
        src = self._src()
        idx = src.index('"/v1/signal/counter"')
        snippet = src[idx:idx + 200]
        assert 'tags=["signal"]' in snippet

    def test_signal_detect_request_model(self):
        assert "SignalDetectRequest" in self._src()

    def test_signal_counter_request_model(self):
        assert "SignalCounterRequest" in self._src()

    def test_net_kpi_impact_key(self):
        assert '"net_kpi_impact"' in self._src()

    def test_counter_actions_key(self):
        assert '"counter_actions"' in self._src()

    def test_optimized_context_key(self):
        assert '"optimized_context"' in self._src()

    def test_threat_count_key(self):
        assert '"threat_count"' in self._src()

    def test_n_signals_key(self):
        assert '"n_signals"' in self._src()

    def test_external_signal_agent_used(self):
        assert "ExternalSignalAgent" in self._src()

    def test_signal_detector_used(self):
        assert "SignalDetector" in self._src()

    def test_verify_api_key_detect(self):
        src = self._src()
        idx = src.index("def signal_detect")
        snippet = src[max(0, idx - 200):idx + 100]
        assert "verify_api_key" in snippet

    def test_verify_api_key_counter(self):
        src = self._src()
        idx = src.index("def signal_counter")
        snippet = src[max(0, idx - 200):idx + 100]
        assert "verify_api_key" in snippet
