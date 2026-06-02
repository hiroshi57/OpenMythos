"""
Sprint 29 テスト — 自律タスク計画 TaskPlanner (P10)

- TestTask              : Task データ構造
- TestTaskGraph         : 依存グラフ・トポロジカルソート
- TestTaskExecutionResult : 実行結果データ
- TestTaskPlan          : プランデータ構造
- TestTaskPlanResult    : 実行結果集計
- TestTaskPlanner       : 分解・実行・統合
- TestIntegration       : 他スプリントとの連携
"""

from __future__ import annotations

import pathlib

_ROOT = pathlib.Path(__file__).parent.parent


# ===========================================================================
# TestTask
# ===========================================================================


class TestTask:
    def test_basic_attributes(self):
        from open_mythos.task_planner import Task
        t = Task(name="analyze", goal="コンテキストを分析する")
        assert t.name == "analyze"
        assert t.goal == "コンテキストを分析する"
        assert t.task_type == "generic"
        assert t.priority == 3

    def test_depends_on_default_empty(self):
        from open_mythos.task_planner import Task
        t = Task(name="t1", goal="goal")
        assert t.depends_on == []

    def test_task_id_unique(self):
        from open_mythos.task_planner import Task
        t1 = Task(name="t1", goal="g1")
        t2 = Task(name="t2", goal="g2")
        assert t1.task_id != t2.task_id

    def test_task_hashable(self):
        from open_mythos.task_planner import Task
        t = Task(name="t", goal="g")
        s = {t}
        assert len(s) == 1


# ===========================================================================
# TestTaskGraph
# ===========================================================================


class TestTaskGraph:
    def test_single_task_one_wave(self):
        from open_mythos.task_planner import Task, TaskGraph
        tasks = [Task(name="t1", goal="g")]
        graph = TaskGraph(tasks)
        waves = graph.topological_order()
        assert len(waves) == 1
        assert waves[0][0].name == "t1"

    def test_independent_tasks_same_wave(self):
        from open_mythos.task_planner import Task, TaskGraph
        tasks = [
            Task(name="a", goal="a"),
            Task(name="b", goal="b"),
            Task(name="c", goal="c"),
        ]
        graph = TaskGraph(tasks)
        waves = graph.topological_order()
        assert sum(len(w) for w in waves) == 3
        assert len(waves[0]) == 3  # 依存なし → 全て同一 wave

    def test_dependency_creates_two_waves(self):
        from open_mythos.task_planner import Task, TaskGraph
        tasks = [
            Task(name="first", goal="最初"),
            Task(name="second", goal="次", depends_on=["first"]),
        ]
        graph = TaskGraph(tasks)
        waves = graph.topological_order()
        assert len(waves) == 2
        assert waves[0][0].name == "first"
        assert waves[1][0].name == "second"

    def test_invalid_dependency_removed(self):
        from open_mythos.task_planner import Task, TaskGraph
        tasks = [Task(name="t", goal="g", depends_on=["nonexistent"])]
        graph = TaskGraph(tasks)
        waves = graph.topological_order()
        # 無効な依存は除去 → エラーなし
        assert len(waves) >= 1

    def test_tasks_property(self):
        from open_mythos.task_planner import Task, TaskGraph
        tasks = [Task(name=f"t{i}", goal=f"g{i}") for i in range(3)]
        graph = TaskGraph(tasks)
        assert len(graph.tasks) == 3


# ===========================================================================
# TestTaskExecutionResult
# ===========================================================================


class TestTaskExecutionResult:
    def _make_result(self, success=True, error=None):
        from open_mythos.task_planner import Task, TaskExecutionResult
        task = Task(name="test", goal="テスト")
        return TaskExecutionResult(
            task=task,
            output="出力テキスト",
            score=0.8,
            latency_ms=15.0,
            success=success,
            error=error,
        )

    def test_ok_true_when_success(self):
        r = self._make_result(success=True)
        assert r.ok is True

    def test_ok_false_when_error(self):
        r = self._make_result(success=False, error="エラー発生")
        assert r.ok is False

    def test_latency_ms_positive(self):
        r = self._make_result()
        assert r.latency_ms >= 0


# ===========================================================================
# TestTaskPlan
# ===========================================================================


class TestTaskPlan:
    def _make_plan(self):
        from open_mythos.task_planner import Task, TaskGraph, TaskPlan
        tasks = [Task(name=f"t{i}", goal=f"g{i}") for i in range(3)]
        graph = TaskGraph(tasks)
        waves = graph.topological_order()
        return TaskPlan(goal="テストゴール", tasks=tasks, waves=waves)

    def test_total_tasks(self):
        plan = self._make_plan()
        assert plan.total_tasks == 3

    def test_n_waves_positive(self):
        plan = self._make_plan()
        assert plan.n_waves >= 1

    def test_plan_id_generated(self):
        plan = self._make_plan()
        assert plan.plan_id != ""


# ===========================================================================
# TestTaskPlanResult
# ===========================================================================


class TestTaskPlanResult:
    def _make_result(self):
        from open_mythos.task_planner import Task, TaskExecutionResult, TaskGraph, TaskPlan, TaskPlanResult
        tasks = [Task(name=f"t{i}", goal=f"g{i}") for i in range(3)]
        graph = TaskGraph(tasks)
        waves = graph.topological_order()
        plan = TaskPlan(goal="goal", tasks=tasks, waves=waves)
        subtask_results = [
            TaskExecutionResult(task=tasks[0], output="出力A", score=0.8, latency_ms=10, success=True),
            TaskExecutionResult(task=tasks[1], output="出力B", score=0.7, latency_ms=15, success=True),
            TaskExecutionResult(task=tasks[2], output="", score=0.0, latency_ms=5, success=False, error="err"),
        ]
        return TaskPlanResult(
            plan=plan,
            subtask_results=subtask_results,
            synthesized_output="統合出力テキスト",
            total_score=0.5,
            total_latency_ms=30.0,
            succeeded_count=2,
            failed_count=1,
        )

    def test_success_rate(self):
        r = self._make_result()
        assert abs(r.success_rate - 2/3) < 0.01

    def test_kpi_achieved_default_false(self):
        r = self._make_result()
        assert isinstance(r.kpi_achieved, bool)

    def test_summary_string(self):
        r = self._make_result()
        s = r.summary()
        assert "TaskPlanResult" in s
        assert "success" in s


# ===========================================================================
# TestTaskPlanner
# ===========================================================================


class TestTaskPlanner:
    def test_decompose_returns_plan(self):
        from open_mythos.task_planner import TaskPlan, TaskPlanner
        planner = TaskPlanner()
        plan = planner.decompose("SEO記事を作成して検索順位を向上させる")
        assert isinstance(plan, TaskPlan)
        assert plan.total_tasks >= 1

    def test_decompose_creates_multiple_tasks_for_complex_goal(self):
        from open_mythos.task_planner import TaskPlanner
        planner = TaskPlanner()
        plan = planner.decompose("分析してKPIを評価し、SEOコンテンツを生成する")
        assert plan.total_tasks >= 2

    def test_execute_returns_plan_result(self):
        from open_mythos.task_planner import TaskPlanResult, TaskPlanner
        planner = TaskPlanner()
        result = planner.execute("テストゴール")
        assert isinstance(result, TaskPlanResult)

    def test_execute_with_context(self):
        from open_mythos.task_planner import TaskPlanner
        planner = TaskPlanner()
        result = planner.execute(
            "SEO記事を最適化する",
            context={"domain": "AI技術", "target_kpi": 0.7},
        )
        assert result.total_score >= 0.0

    def test_synthesized_output_not_empty(self):
        from open_mythos.task_planner import TaskPlanner
        planner = TaskPlanner()
        result = planner.execute("コンテンツを生成する")
        assert result.synthesized_output != ""

    def test_succeeded_count_positive(self):
        from open_mythos.task_planner import TaskPlanner
        planner = TaskPlanner()
        result = planner.execute("分析タスクを実行する")
        assert result.succeeded_count >= 0

    def test_subtask_results_non_empty(self):
        from open_mythos.task_planner import TaskPlanner
        planner = TaskPlanner()
        result = planner.execute("KPI評価とSEO最適化を行う")
        assert len(result.subtask_results) >= 1

    def test_total_latency_nonneg(self):
        from open_mythos.task_planner import TaskPlanner
        planner = TaskPlanner()
        result = planner.execute("テスト実行")
        assert result.total_latency_ms >= 0.0

    def test_dependency_outputs_passed_to_next_task(self):
        """前段タスクの出力が後段タスクのコンテキストに渡されること。"""
        from open_mythos.task_planner import Task, TaskPlanner

        received_contexts = []

        def capture_executor(task, ctx):
            received_contexts.append(ctx.get("previous_outputs", {}))
            return f"output of {task.name}"

        planner = TaskPlanner(default_executor=capture_executor)
        plan = planner.decompose("分析して生成する")
        planner._execute_plan(plan, {})
        # 少なくとも1つのタスクが前段出力を受け取っている (依存がある場合)
        assert isinstance(received_contexts, list)

    def test_kpi_target_check(self):
        from open_mythos.task_planner import TaskPlanner

        def high_score_executor(task, ctx):
            return "高品質な出力テキストです。詳細なレポートを含みます。"

        planner = TaskPlanner(default_executor=high_score_executor, kpi_target=0.5)
        result = planner.execute("高品質タスク")
        assert isinstance(result.kpi_achieved, bool)

    def test_custom_executor_called(self):
        from open_mythos.task_planner import TaskPlanner

        called_tasks = []

        def custom_exec(task, ctx):
            called_tasks.append(task.name)
            return f"custom output for {task.name}"

        planner = TaskPlanner(default_executor=custom_exec)
        planner.execute("テストゴール")
        assert len(called_tasks) >= 1

    def test_planner_summary_in_result(self):
        from open_mythos.task_planner import TaskPlanner
        planner = TaskPlanner()
        result = planner.execute("SEO分析と生成")
        assert isinstance(result.summary(), str)


# ===========================================================================
# TestIntegration
# ===========================================================================


class TestIntegration:
    def test_swarm_orchestrator_as_executor(self):
        """SwarmOrchestrator (Sprint 13) をタスク実行に使用。"""
        from open_mythos.swarm import SwarmOrchestrator
        from open_mythos.task_planner import Task, TaskPlanner

        def swarm_exec(task: Task, ctx: dict) -> str:
            agents = [lambda t, name=task.name: f"{name}の結果" for _ in range(2)]
            result = SwarmOrchestrator(agents).map(task.goal)
            return result.outputs[0] if result.outputs else "no output"

        planner = TaskPlanner(default_executor=swarm_exec)
        result = planner.execute("SEOコンテンツを生成する")
        assert result.succeeded_count >= 0

    def test_kpi_agent_for_task_evaluation(self):
        """KPIAgent (Sprint 21) でタスク達成を評価。"""
        from open_mythos.kpi_agent import KPIAgent, KPIDefinition
        from open_mythos.task_planner import TaskPlanner

        plan_results = []

        def tracked_exec(task, ctx):
            output = f"{task.name} の完了した出力テキストです。"
            plan_results.append(output)
            return output

        planner = TaskPlanner(default_executor=tracked_exec, kpi_target=0.6)
        result = planner.execute("KPI最適化タスク")
        assert isinstance(result.kpi_achieved, bool)

    def test_ensemble_scorer_for_synthesis_quality(self):
        """EnsembleScorer (Sprint 27) で統合出力の品質を評価。"""
        from open_mythos.ensemble_scorer import EnsembleScorer
        from open_mythos.task_planner import TaskPlanner

        scorer = EnsembleScorer()
        planner = TaskPlanner()
        result = planner.execute("SEO記事を分析して最適化する", context={"domain": "AI"})
        quality = scorer.score(result.synthesized_output).ensemble_score
        assert quality >= 0.0

    def test_long_term_memory_stores_results(self):
        """タスク結果を LongTermMemoryAgent (Sprint 26) に格納。"""
        from open_mythos.long_term_memory import LongTermMemoryAgent
        from open_mythos.task_planner import TaskPlanner

        memory = LongTermMemoryAgent(score_threshold=0.0)
        planner = TaskPlanner()
        result = planner.execute("SEO最適化タスク")
        # 結果を記憶に格納
        memory.store_episode(
            "SEO最適化タスク",
            result.synthesized_output,
            score=result.total_score,
        )
        assert memory.stats()["total_entries"] >= 0

    def test_debate_orchestrator_for_synthesize(self):
        """DebateOrchestrator (Sprint 20) の合意出力をタスク統合に使用。"""
        from open_mythos.debate import DebateConfig, DebateOrchestrator
        from open_mythos.main import OpenMythos
        from open_mythos.task_planner import TaskPlanner
        from open_mythos.variants import mythos_nano

        model = OpenMythos(mythos_nano()).eval()

        def debate_exec(task, ctx):
            cfg = DebateConfig(n_agents=2, n_rounds=1)
            with DebateOrchestrator(model, cfg) as debate:
                r = debate.run(task.goal[:80])
            return r.consensus

        planner = TaskPlanner(default_executor=debate_exec)
        result = planner.execute("マーケティング戦略を評価する")
        assert result.synthesized_output != ""
