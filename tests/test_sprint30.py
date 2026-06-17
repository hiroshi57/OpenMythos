"""
Sprint 30 — GrowingAIOrchestrator テストスイート (40 tests)
"""

from open_mythos.growing_ai_orchestrator import (
    PatternType,
    GrowthContext,
    PatternResult,
    OrchestratorResult,
    PatternSelector,
    GrowingAIOrchestrator,
)


# =========================================================================
# PatternType enum (10 tests)
# =========================================================================

def test_pattern_type_debate():
    assert PatternType.DEBATE.value == "debate"

def test_pattern_type_kpi():
    assert PatternType.KPI.value == "kpi"

def test_pattern_type_profiler():
    assert PatternType.PROFILER.value == "profiler"

def test_pattern_type_signal():
    assert PatternType.SIGNAL.value == "signal"

def test_pattern_type_mistake():
    assert PatternType.MISTAKE.value == "mistake"

def test_pattern_type_distill():
    assert PatternType.DISTILL.value == "distill"

def test_pattern_type_memory():
    assert PatternType.MEMORY.value == "memory"

def test_pattern_type_ensemble():
    assert PatternType.ENSEMBLE.value == "ensemble"

def test_pattern_type_evolve():
    assert PatternType.EVOLVE.value == "evolve"

def test_pattern_type_plan():
    assert PatternType.PLAN.value == "plan"


# =========================================================================
# GrowthContext (3 tests)
# =========================================================================

def test_growth_context_defaults():
    ctx = GrowthContext(goal="improve quality")
    assert ctx.goal == "improve quality"
    assert ctx.hints == []
    assert ctx.metadata == {}
    assert ctx.history == []

def test_growth_context_with_hints():
    ctx = GrowthContext(goal="test", hints=["kpi", "improve"])
    assert len(ctx.hints) == 2

def test_growth_context_with_metadata():
    ctx = GrowthContext(goal="test", metadata={"priority": "high"})
    assert ctx.metadata["priority"] == "high"


# =========================================================================
# PatternSelector (12 tests)
# =========================================================================

def test_selector_debate_keyword():
    sel = PatternSelector()
    ctx = GrowthContext(goal="このトピックについて議論してください")
    selected = sel.select(ctx, max_patterns=3)
    assert PatternType.DEBATE in selected

def test_selector_kpi_keyword():
    sel = PatternSelector()
    ctx = GrowthContext(goal="KPIを改善して目標を達成する")
    selected = sel.select(ctx, max_patterns=3)
    assert PatternType.KPI in selected

def test_selector_profiler_keyword():
    sel = PatternSelector()
    ctx = GrowthContext(goal="処理が遅いボトルネックを解消したい")
    selected = sel.select(ctx, max_patterns=3)
    assert PatternType.PROFILER in selected

def test_selector_signal_keyword():
    sel = PatternSelector()
    ctx = GrowthContext(goal="市場のtrendを分析する")
    selected = sel.select(ctx, max_patterns=3)
    assert PatternType.SIGNAL in selected

def test_selector_mistake_keyword():
    sel = PatternSelector()
    ctx = GrowthContext(goal="エラーを防止するguardを実装")
    selected = sel.select(ctx, max_patterns=3)
    assert PatternType.MISTAKE in selected

def test_selector_memory_keyword():
    sel = PatternSelector()
    ctx = GrowthContext(goal="過去の履歴を参照して回答する")
    selected = sel.select(ctx, max_patterns=3)
    assert PatternType.MEMORY in selected

def test_selector_evolve_keyword():
    sel = PatternSelector()
    ctx = GrowthContext(goal="プロンプトを進化させて最適化する")
    selected = sel.select(ctx, max_patterns=3)
    assert PatternType.EVOLVE in selected

def test_selector_plan_keyword():
    sel = PatternSelector()
    ctx = GrowthContext(goal="タスクを計画してステップに分解する")
    selected = sel.select(ctx, max_patterns=3)
    assert PatternType.PLAN in selected

def test_selector_default_no_match():
    sel = PatternSelector()
    ctx = GrowthContext(goal="xyzzy foobar 12345")
    selected = sel.select(ctx, max_patterns=3)
    assert len(selected) >= 1  # デフォルトは必ず 1 件以上

def test_selector_max_patterns_respected():
    sel = PatternSelector()
    ctx = GrowthContext(goal="plan task decompose evaluate score kpi improve profile bottleneck")
    selected = sel.select(ctx, max_patterns=2)
    assert len(selected) <= 2

def test_selector_hints_boost():
    sel = PatternSelector()
    # goal には kpi キーワードなし、hint に kpi あり
    ctx = GrowthContext(goal="make it better", hints=["kpi", "改善"])
    scores = sel.score_all(ctx)
    assert PatternType.KPI in scores
    assert scores[PatternType.KPI] > 0

def test_selector_score_all_returns_dict():
    sel = PatternSelector()
    ctx = GrowthContext(goal="debate and compare options")
    scores = sel.score_all(ctx)
    assert isinstance(scores, dict)
    assert PatternType.DEBATE in scores


# =========================================================================
# PatternResult (3 tests)
# =========================================================================

def test_pattern_result_success_flag():
    r = PatternResult(pattern=PatternType.ENSEMBLE, output="ok", score=0.8, latency_ms=5.0)
    assert r.success is True

def test_pattern_result_error_flag():
    r = PatternResult(pattern=PatternType.PLAN, output="", score=0.0, latency_ms=1.0, error="timeout")
    assert r.success is False

def test_pattern_result_score_clamp():
    orch = GrowingAIOrchestrator()
    # 直接 _execute_patterns でスコアが 0〜1 に収まることを確認
    ctx = GrowthContext(goal="test")
    results = orch._execute_patterns("test", [PatternType.ENSEMBLE], ctx)
    assert all(0.0 <= r.score <= 1.0 for r in results)


# =========================================================================
# GrowingAIOrchestrator — 基本動作 (12 tests)
# =========================================================================

def test_orchestrator_instantiation():
    orch = GrowingAIOrchestrator()
    assert orch.max_patterns == 3
    assert orch.timeout_s == 30.0

def test_orchestrator_custom_params():
    orch = GrowingAIOrchestrator(max_patterns=1, timeout_s=5.0)
    assert orch.max_patterns == 1

def test_orchestrator_run_returns_result():
    orch = GrowingAIOrchestrator(max_patterns=1)
    result = orch.run("evaluate the quality of this text")
    assert isinstance(result, OrchestratorResult)

def test_orchestrator_run_patterns_nonempty():
    orch = GrowingAIOrchestrator(max_patterns=1)
    result = orch.run("evaluate the quality")
    assert len(result.patterns_used) >= 1

def test_orchestrator_run_final_output_nonempty():
    orch = GrowingAIOrchestrator(max_patterns=1)
    result = orch.run("plan the workflow steps")
    assert len(result.final_output) > 0

def test_orchestrator_run_overall_score_range():
    orch = GrowingAIOrchestrator(max_patterns=1)
    result = orch.run("evaluate quality score")
    assert 0.0 <= result.overall_score <= 1.0

def test_orchestrator_run_latency_positive():
    orch = GrowingAIOrchestrator(max_patterns=1)
    result = orch.run("decompose this task into steps")
    assert result.total_latency_ms > 0

def test_orchestrator_history_empty_initially():
    orch = GrowingAIOrchestrator()
    assert orch.history() == []

def test_orchestrator_history_accumulates():
    orch = GrowingAIOrchestrator(max_patterns=1)
    orch.run("plan task")
    orch.run("evaluate quality")
    assert len(orch.history()) == 2

def test_orchestrator_clear_history():
    orch = GrowingAIOrchestrator(max_patterns=1)
    orch.run("test goal")
    orch.clear_history()
    assert orch.history() == []

def test_orchestrator_error_resilience():
    """パターン実行中に例外が起きても OrchestratorResult が返る"""
    orch = GrowingAIOrchestrator(max_patterns=1)
    # 存在しないゴールでも結果が返るはず
    result = orch.run("xyzzy foobar incomprehensible goal 999")
    assert isinstance(result, OrchestratorResult)

def test_orchestrator_metadata_passthrough():
    orch = GrowingAIOrchestrator(max_patterns=1)
    result = orch.run("evaluate quality", metadata={"user": "test"})
    # metadata は result には含まれないが、実行がエラーにならないことを確認
    assert isinstance(result, OrchestratorResult)


# =========================================================================
# OrchestratorResult 構造 (3 tests)
# =========================================================================

def test_orchestrator_result_structure():
    orch = GrowingAIOrchestrator(max_patterns=1)
    result = orch.run("plan workflow steps")
    assert hasattr(result, "goal")
    assert hasattr(result, "patterns_used")
    assert hasattr(result, "results")
    assert hasattr(result, "final_output")
    assert hasattr(result, "overall_score")
    assert hasattr(result, "total_latency_ms")

def test_orchestrator_result_goal_preserved():
    orch = GrowingAIOrchestrator(max_patterns=1)
    goal = "unique goal string for test"
    result = orch.run(goal)
    assert result.goal == goal

def test_orchestrator_result_patterns_list():
    orch = GrowingAIOrchestrator(max_patterns=2)
    result = orch.run("evaluate and plan the task steps")
    assert isinstance(result.patterns_used, list)
    assert all(isinstance(p, PatternType) for p in result.patterns_used)


# =========================================================================
# _synthesize / _aggregate_score (4 tests)
# =========================================================================

def test_synthesize_picks_best_score():
    results = [
        PatternResult(PatternType.ENSEMBLE, "low",  score=0.3, latency_ms=1.0),
        PatternResult(PatternType.PLAN,     "high", score=0.9, latency_ms=1.0),
    ]
    output = GrowingAIOrchestrator._synthesize("goal", results)
    assert output == "high"

def test_synthesize_fallback_on_all_errors():
    results = [
        PatternResult(PatternType.ENSEMBLE, "", score=0.0, latency_ms=1.0, error="fail"),
    ]
    output = GrowingAIOrchestrator._synthesize("fallback goal", results)
    assert output == "fallback goal"

def test_aggregate_score_average():
    results = [
        PatternResult(PatternType.ENSEMBLE, "a", score=0.6, latency_ms=1.0),
        PatternResult(PatternType.PLAN,     "b", score=0.8, latency_ms=1.0),
    ]
    score = GrowingAIOrchestrator._aggregate_score(results)
    assert abs(score - 0.7) < 1e-4

def test_aggregate_score_zero_on_all_errors():
    results = [
        PatternResult(PatternType.PLAN, "", score=0.0, latency_ms=1.0, error="err"),
    ]
    score = GrowingAIOrchestrator._aggregate_score(results)
    assert score == 0.0
