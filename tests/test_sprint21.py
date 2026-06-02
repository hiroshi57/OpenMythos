"""
Sprint 21 テスト — KPI駆動自己改善 KPIAgent

- TestKPIDefinition    : KPI定義の検証
- TestKPISnapshot      : スナップショットのギャップ計算
- TestGapReport        : Gap分析レポート
- TestActionPlan       : アクションプラン生成
- TestKPIAgentMeasure  : 計測機能
- TestKPIAgentAnalyze  : 分析機能
- TestKPIAgentPlan     : プラン生成
- TestKPIAgentExecute  : アクション実行
- TestKPIImproveLoop   : 自律改善サイクル
- TestKPIAPIEndpoint   : FastAPI /v1/kpi/* (静的検査)
"""

from __future__ import annotations

import pathlib

_ROOT = pathlib.Path(__file__).parent.parent


def _make_kpi(target=0.8, context="SEO対策は重要です。具体的には数値目標を設定することが有効です。"):
    from open_mythos.kpi_agent import KPIDefinition
    from open_mythos.llmo import LLMOScorer

    scorer = LLMOScorer()
    return KPIDefinition(
        name="llmo_score",
        target=target,
        measure_fn=lambda text: scorer.score(text).llmo_total,
        context=context,
    )


# ===========================================================================
# TestKPIDefinition
# ===========================================================================


class TestKPIDefinition:
    def test_defaults(self):
        from open_mythos.kpi_agent import KPIDefinition

        kpi = KPIDefinition(
            name="test_kpi",
            target=0.5,
            measure_fn=lambda x: 0.3,
        )
        assert kpi.name == "test_kpi"
        assert kpi.target == 0.5
        assert kpi.higher_is_better is True
        assert kpi.action_budget == 3

    def test_higher_is_better_false(self):
        from open_mythos.kpi_agent import KPIDefinition

        kpi = KPIDefinition(
            name="error_rate",
            target=0.05,
            measure_fn=lambda x: 0.1,
            higher_is_better=False,
        )
        assert kpi.higher_is_better is False

    def test_measure_fn_callable(self):
        from open_mythos.kpi_agent import KPIDefinition

        kpi = KPIDefinition(name="x", target=1.0, measure_fn=lambda t: len(t) / 100)
        assert callable(kpi.measure_fn)
        assert kpi.measure_fn("hello") == 0.05


# ===========================================================================
# TestKPISnapshot
# ===========================================================================


class TestKPISnapshot:
    def test_gap_to_higher_is_better(self):
        import pytest
        from open_mythos.kpi_agent import KPISnapshot

        s = KPISnapshot(kpi_name="llmo", value=0.5, context="")
        assert s.gap_to(0.8) == pytest.approx(0.3)

    def test_gap_to_achieved(self):
        from open_mythos.kpi_agent import KPISnapshot

        s = KPISnapshot(kpi_name="llmo", value=0.9, context="")
        assert s.gap_to(0.8) == 0.0

    def test_gap_to_lower_is_better(self):
        from open_mythos.kpi_agent import KPISnapshot

        s = KPISnapshot(kpi_name="error_rate", value=0.15, context="")
        gap = s.gap_to(0.05, higher_is_better=False)
        assert abs(gap - 0.10) < 1e-6

    def test_achieved_true(self):
        from open_mythos.kpi_agent import KPISnapshot

        s = KPISnapshot(kpi_name="x", value=0.8, context="")
        assert s.achieved(0.7) is True

    def test_achieved_false(self):
        from open_mythos.kpi_agent import KPISnapshot

        s = KPISnapshot(kpi_name="x", value=0.5, context="")
        assert s.achieved(0.8) is False

    def test_snapshot_id_unique(self):
        from open_mythos.kpi_agent import KPISnapshot

        s1 = KPISnapshot(kpi_name="x", value=0.5, context="")
        s2 = KPISnapshot(kpi_name="x", value=0.5, context="")
        assert s1.snapshot_id != s2.snapshot_id

    def test_cycle_default_zero(self):
        from open_mythos.kpi_agent import KPISnapshot

        s = KPISnapshot(kpi_name="x", value=0.5, context="")
        assert s.cycle == 0


# ===========================================================================
# TestGapReport
# ===========================================================================


class TestGapReport:
    def _make(self, gap=0.2, gap_pct=25.0):
        from open_mythos.kpi_agent import GapReport

        return GapReport(
            kpi_name="llmo",
            current_value=0.6,
            target_value=0.8,
            gap=gap,
            gap_pct=gap_pct,
            priority="medium",
            diagnosis="テスト",
        )

    def test_achieved_false_when_gap_positive(self):
        report = self._make(gap=0.2)
        assert report.achieved is False

    def test_achieved_true_when_gap_zero(self):
        report = self._make(gap=0.0)
        assert report.achieved is True

    def test_priority_field(self):
        report = self._make()
        assert report.priority in ("high", "medium", "low")


# ===========================================================================
# TestActionPlan
# ===========================================================================


class TestActionPlan:
    def _make_plan(self):
        from open_mythos.kpi_agent import ActionPlan, GapReport, Action

        gap_report = GapReport(
            kpi_name="llmo",
            current_value=0.5,
            target_value=0.8,
            gap=0.3,
            gap_pct=37.5,
            priority="high",
            diagnosis="大幅な改善が必要",
        )
        actions = [
            Action("a1", "アクション1", lambda x: x, estimated_impact=0.15, priority=0),
            Action("a2", "アクション2", lambda x: x, estimated_impact=0.10, priority=1),
            Action("a3", "アクション3", lambda x: x, estimated_impact=0.20, priority=2),
        ]
        return ActionPlan(kpi_name="llmo", gap_report=gap_report, actions=actions)

    def test_top_actions_sorted_by_impact(self):
        plan = self._make_plan()
        top = plan.top_actions(2)
        assert len(top) == 2
        assert top[0].estimated_impact >= top[1].estimated_impact

    def test_top_actions_limit(self):
        plan = self._make_plan()
        top = plan.top_actions(1)
        assert len(top) == 1

    def test_actions_not_empty(self):
        plan = self._make_plan()
        assert len(plan.actions) == 3


# ===========================================================================
# TestKPIAgentMeasure
# ===========================================================================


class TestKPIAgentMeasure:
    def test_measure_returns_snapshot(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = agent.measure()
        assert isinstance(snap, KPISnapshot)

    def test_measure_kpi_name(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = agent.measure()
        assert snap.kpi_name == "llmo_score"

    def test_measure_value_in_range(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = agent.measure()
        assert 0.0 <= snap.value <= 1.0

    def test_measure_custom_context(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = agent.measure("カスタムコンテキスト")
        assert snap.context == "カスタムコンテキスト"

    def test_measure_cycle_assigned(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = agent.measure(cycle=3)
        assert snap.cycle == 3


# ===========================================================================
# TestKPIAgentAnalyze
# ===========================================================================


class TestKPIAgentAnalyze:
    def test_analyze_returns_gap_report(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot, GapReport

        kpi = _make_kpi(target=0.9)
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo_score", value=0.5, context="")
        report = agent.analyze(snap)
        assert isinstance(report, GapReport)

    def test_analyze_gap_positive_when_below_target(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot

        kpi = _make_kpi(target=0.9)
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo_score", value=0.5, context="")
        report = agent.analyze(snap)
        assert report.gap > 0

    def test_analyze_achieved_when_above_target(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot

        kpi = _make_kpi(target=0.3)
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo_score", value=0.8, context="")
        report = agent.analyze(snap)
        assert report.achieved is True

    def test_analyze_priority_high_for_large_gap(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot

        kpi = _make_kpi(target=1.0)
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo_score", value=0.1, context="")
        report = agent.analyze(snap)
        assert report.priority == "high"

    def test_analyze_diagnosis_is_string(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo_score", value=0.5, context="")
        report = agent.analyze(snap)
        assert isinstance(report.diagnosis, str)
        assert len(report.diagnosis) > 0


# ===========================================================================
# TestKPIAgentPlan
# ===========================================================================


class TestKPIAgentPlan:
    def test_plan_returns_action_plan(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot, ActionPlan

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo_score", value=0.4, context="")
        gap = agent.analyze(snap)
        plan = agent.plan(gap)
        assert isinstance(plan, ActionPlan)

    def test_plan_actions_not_empty(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo_score", value=0.4, context="")
        gap = agent.analyze(snap)
        plan = agent.plan(gap)
        assert len(plan.actions) > 0

    def test_plan_respects_action_budget(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot, KPIDefinition
        from open_mythos.llmo import LLMOScorer

        scorer = LLMOScorer()
        kpi = KPIDefinition(
            name="llmo", target=0.8,
            measure_fn=lambda t: scorer.score(t).llmo_total,
            action_budget=2,
        )
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo", value=0.4, context="")
        gap = agent.analyze(snap)
        plan = agent.plan(gap)
        assert len(plan.top_actions(2)) <= 2

    def test_plan_cycle_assigned(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo_score", value=0.4, context="")
        gap = agent.analyze(snap)
        plan = agent.plan(gap, cycle=2)
        assert plan.cycle == 2


# ===========================================================================
# TestKPIAgentExecute
# ===========================================================================


class TestKPIAgentExecute:
    def test_execute_returns_string(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo_score", value=0.4, context="test")
        gap = agent.analyze(snap)
        plan = agent.plan(gap)
        result = agent.execute(plan, "test context")
        assert isinstance(result, str)

    def test_execute_modifies_context(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        original = "元のコンテキスト"
        snap = KPISnapshot(kpi_name="llmo_score", value=0.3, context=original)
        gap = agent.analyze(snap)
        plan = agent.plan(gap)
        result = agent.execute(plan, original)
        assert len(result) >= len(original)

    def test_execute_not_empty(self):
        from open_mythos.kpi_agent import KPIAgent, KPISnapshot

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        snap = KPISnapshot(kpi_name="llmo_score", value=0.4, context="")
        gap = agent.analyze(snap)
        plan = agent.plan(gap)
        result = agent.execute(plan, "入力テキスト")
        assert result != ""


# ===========================================================================
# TestKPIImproveLoop
# ===========================================================================


class TestKPIImproveLoop:
    def test_improve_loop_returns_result(self):
        from open_mythos.kpi_agent import KPIAgent, KPIImproveResult

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=2)
        assert isinstance(result, KPIImproveResult)

    def test_improve_loop_snapshots_length(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=2)
        assert len(result.snapshots) >= 1

    def test_improve_loop_plans_generated(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=2)
        assert len(result.plans) >= 1

    def test_improve_loop_kpi_name(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=1)
        assert result.kpi_name == "llmo_score"

    def test_improve_loop_initial_snapshot(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=1)
        assert result.initial_snapshot.cycle == 0

    def test_improve_loop_final_value_in_range(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=2)
        assert 0.0 <= result.final_snapshot.value <= 2.0

    def test_improve_loop_latency_positive(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=1)
        assert result.total_latency_ms > 0

    def test_improve_loop_n_cycles_used(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=3)
        assert 1 <= result.n_cycles_used <= 3

    def test_improve_loop_achieved_field_bool(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=1)
        assert isinstance(result.achieved_target, bool)

    def test_improve_loop_improvement_pct_field(self):
        from open_mythos.kpi_agent import KPIAgent

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=2)
        assert isinstance(result.improvement_pct, float)

    def test_improve_loop_achieved_when_low_target(self):
        from open_mythos.kpi_agent import KPIAgent, KPIDefinition
        from open_mythos.llmo import LLMOScorer

        scorer = LLMOScorer()
        kpi = KPIDefinition(
            name="llmo", target=0.0,
            measure_fn=lambda t: scorer.score(t).llmo_total,
            context="テスト",
        )
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=1)
        assert result.achieved_target is True

    def test_improve_loop_early_stop_on_target(self):
        from open_mythos.kpi_agent import KPIAgent, KPIDefinition

        kpi = KPIDefinition(
            name="always_achieved", target=0.0,
            measure_fn=lambda t: 1.0,
            context="test",
        )
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=5, early_stop=True)
        assert result.n_cycles_used <= 5

    def test_improve_loop_no_early_stop_runs_all_cycles(self):
        from open_mythos.kpi_agent import KPIAgent, KPIDefinition

        kpi = KPIDefinition(
            name="always_achieved", target=0.0,
            measure_fn=lambda t: 1.0,
            context="test",
        )
        agent = KPIAgent(kpi)
        result = agent.improve_loop(n_cycles=3, early_stop=False)
        assert result.n_cycles_used == 3

    def test_builtin_actions_available(self):
        from open_mythos.kpi_agent import KPIAgent, _BUILTIN_ACTIONS

        kpi = _make_kpi()
        agent = KPIAgent(kpi)
        assert len(agent._actions) >= len(_BUILTIN_ACTIONS)

    def test_extra_actions_added(self):
        from open_mythos.kpi_agent import KPIAgent, Action, _BUILTIN_ACTIONS

        extra = [Action("custom", "カスタム", lambda x: x + "追加", estimated_impact=0.5)]
        kpi = _make_kpi()
        agent = KPIAgent(kpi, extra_actions=extra)
        assert len(agent._actions) == len(_BUILTIN_ACTIONS) + 1


# ===========================================================================
# TestKPIAPIEndpoint (静的ソース検査)
# ===========================================================================


class TestKPIAPIEndpoint:
    def _src(self) -> str:
        return (_ROOT / "serve" / "api.py").read_text(encoding="utf-8")

    def test_kpi_measure_route_exists(self):
        assert '"/v1/kpi/measure"' in self._src()

    def test_kpi_improve_route_exists(self):
        assert '"/v1/kpi/improve"' in self._src()

    def test_kpi_measure_post_method(self):
        src = self._src()
        idx = src.index('"/v1/kpi/measure"')
        snippet = src[max(0, idx - 60):idx]
        assert "@app.post" in snippet

    def test_kpi_improve_post_method(self):
        src = self._src()
        idx = src.index('"/v1/kpi/improve"')
        snippet = src[max(0, idx - 60):idx]
        assert "@app.post" in snippet

    def test_kpi_tag_in_measure(self):
        src = self._src()
        idx = src.index('"/v1/kpi/measure"')
        snippet = src[idx:idx + 200]
        assert 'tags=["kpi"]' in snippet

    def test_kpi_tag_in_improve(self):
        src = self._src()
        idx = src.index('"/v1/kpi/improve"')
        snippet = src[idx:idx + 200]
        assert 'tags=["kpi"]' in snippet

    def test_kpi_define_request_model(self):
        assert "KPIDefineRequest" in self._src()

    def test_kpi_improve_request_model(self):
        assert "KPIImproveRequest" in self._src()

    def test_kpi_measure_request_model(self):
        assert "KPIMeasureRequest" in self._src()

    def test_kpi_improve_response_initial_value(self):
        assert '"initial_value"' in self._src()

    def test_kpi_improve_response_final_value(self):
        assert '"final_value"' in self._src()

    def test_kpi_improve_response_achieved_target(self):
        assert '"achieved_target"' in self._src()

    def test_kpi_improve_response_improvement_pct(self):
        assert '"improvement_pct"' in self._src()

    def test_kpi_measure_response_gap(self):
        assert '"gap"' in self._src()

    def test_kpi_measure_response_priority(self):
        assert '"priority"' in self._src()

    def test_kpi_improve_uses_kpi_agent(self):
        assert "KPIAgent" in self._src()

    def test_verify_api_key_in_measure(self):
        src = self._src()
        idx = src.index("def kpi_measure")
        snippet = src[max(0, idx - 200):idx + 100]
        assert "verify_api_key" in snippet

    def test_verify_api_key_in_improve(self):
        src = self._src()
        idx = src.index("def kpi_improve")
        snippet = src[max(0, idx - 200):idx + 100]
        assert "verify_api_key" in snippet
