"""
Sprint 43 — HermesOrchestrator: Layer 2 Ultracode Mode テスト

対象:
  - open_mythos/hermes_orchestrator.py
      SubTask, SubAgentSpec, HermesAgentResult, VerificationResult, HermesReport
      TaskDecomposer, AgentSpawner, ParallelExecutor, ResultVerifier, ReportBuilder
      HermesOrchestrator (plan / spawn / verify / report / run_async / run)
  - serve/api.py
      POST /v1/hermes/run
      POST /v1/hermes/plan

テスト構成:
  Section A: データクラス (SubTask, SubAgentSpec, HermesAgentResult, Verification, Report)
  Section B: TaskDecomposer (Phase 1: Plan)
  Section C: AgentSpawner (Phase 2: Spawn)
  Section D: ParallelExecutor (Phase 3: asyncio 並列実行)
  Section E: ResultVerifier (Phase 4: 品質チェック)
  Section F: ReportBuilder (Phase 5: レポート生成)
  Section G: HermesOrchestrator (フルパイプライン)
  Section H: API エンドポイント (/v1/hermes/*)
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# transformers モック (autouse)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def mock_transformers():
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock
    yield mock
    sys.modules.pop("transformers", None)


# ---------------------------------------------------------------------------
# 共通インポート
# ---------------------------------------------------------------------------

from open_mythos.hermes_orchestrator import (
    SubTask,
    SubAgentSpec,
    HermesAgentResult,
    VerificationResult,
    HermesReport,
    TaskDecomposer,
    AgentSpawner,
    ParallelExecutor,
    ResultVerifier,
    ReportBuilder,
    HermesOrchestrator,
    _local_call_fn,
    _execute_single,
)


# ---------------------------------------------------------------------------
# Section A: データクラス
# ---------------------------------------------------------------------------

class TestSubTask:
    def test_subtask_creation(self):
        st = SubTask(task_id="t1", name="テスト", description="説明")
        assert st.task_id == "t1"
        assert st.name == "テスト"
        assert st.description == "説明"
        assert st.priority == 1
        assert st.depends_on == []
        assert st.metadata == {}

    def test_subtask_with_dependencies(self):
        st = SubTask(
            task_id="t2",
            name="依存タスク",
            description="先行タスクに依存",
            priority=2,
            depends_on=["t1"],
            metadata={"domain": "seo"},
        )
        assert st.depends_on == ["t1"]
        assert st.metadata["domain"] == "seo"

    def test_subtask_repr(self):
        st = SubTask(task_id="abc", name="テスト名", description="desc")
        r = repr(st)
        assert "abc" in r
        assert "テスト名" in r


class TestSubAgentSpec:
    def test_spec_creation(self):
        st = SubTask(task_id="t1", name="test", description="desc")
        spec = SubAgentSpec(
            agent_id="agent-1",
            task=st,
            endpoint="/generate",
            payload={"prompt": "hello"},
        )
        assert spec.agent_id == "agent-1"
        assert spec.endpoint == "/generate"
        assert spec.timeout_s == 30.0

    def test_spec_repr(self):
        st = SubTask(task_id="t1", name="myTask", description="d")
        spec = SubAgentSpec(
            agent_id="ag-1",
            task=st,
            endpoint="/api",
            payload={},
        )
        r = repr(spec)
        assert "ag-1" in r
        assert "myTask" in r


class TestHermesAgentResult:
    def test_success_result(self):
        r = HermesAgentResult(
            agent_id="a1",
            task_id="t1",
            task_name="タスク1",
            output="完了しました",
            latency_ms=120.5,
            success=True,
        )
        assert r.is_valid is True
        assert r.error is None

    def test_failure_result(self):
        r = HermesAgentResult(
            agent_id="a1",
            task_id="t1",
            task_name="タスク1",
            output="",
            latency_ms=10.0,
            success=False,
            error="Connection refused",
        )
        assert r.is_valid is False
        assert r.error == "Connection refused"

    def test_empty_output_is_invalid(self):
        r = HermesAgentResult(
            agent_id="a1", task_id="t1", task_name="T",
            output="   ", latency_ms=10.0, success=True,
        )
        assert r.is_valid is False

    def test_result_repr(self):
        r = HermesAgentResult(
            agent_id="ag99", task_id="t1", task_name="myT",
            output="ok", latency_ms=5.0, success=True,
        )
        rep = repr(r)
        assert "ag99" in rep
        assert "myT" in rep


class TestVerificationResult:
    def test_passed_result(self):
        vr = VerificationResult(
            agent_id="a1", task_id="t1", task_name="T",
            passed=True, score=0.9, issues=[], verified_output="出力"
        )
        assert vr.passed is True
        assert vr.score == 0.9

    def test_failed_result_repr(self):
        vr = VerificationResult(
            agent_id="a1", task_id="t1", task_name="myTask",
            passed=False, score=0.1, issues=["短すぎる"],
            verified_output=""
        )
        r = repr(vr)
        assert "FAIL" in r
        assert "myTask" in r


class TestHermesReport:
    def _make_report(self) -> HermesReport:
        st = SubTask(task_id="t1", name="T", description="D")
        ar = HermesAgentResult(
            agent_id="a1", task_id="t1", task_name="T",
            output="出力テキスト", latency_ms=100.0, success=True,
        )
        vr = VerificationResult(
            agent_id="a1", task_id="t1", task_name="T",
            passed=True, score=0.85, issues=[], verified_output="出力テキスト",
        )
        return HermesReport(
            run_id="abc123",
            goal="テストゴール",
            subtasks=[st],
            agent_results=[ar],
            verification_results=[vr],
            final_output="最終出力",
            overall_score=0.85,
            total_latency_ms=150.0,
            success_rate=1.0,
            phase_timings={"plan_ms": 5.0, "spawn_ms": 2.0, "execute_ms": 100.0},
        )

    def test_report_fields(self):
        rpt = self._make_report()
        assert rpt.run_id == "abc123"
        assert rpt.overall_score == pytest.approx(0.85)
        assert rpt.success_rate == pytest.approx(1.0)

    def test_report_summary(self):
        rpt = self._make_report()
        s = rpt.summary()
        assert "abc123" in s
        assert "テストゴール" in s

    def test_report_repr(self):
        rpt = self._make_report()
        assert "HermesReport" in repr(rpt)


# ---------------------------------------------------------------------------
# Section B: TaskDecomposer (Phase 1: Plan)
# ---------------------------------------------------------------------------

class TestTaskDecomposer:
    def setup_method(self):
        self.decomposer = TaskDecomposer()

    def test_seo_domain_detected(self):
        domain = self.decomposer.detect_domain("SEO最適化でキーワードを改善", {})
        assert domain == "seo"

    def test_analysis_domain_detected(self):
        domain = self.decomposer.detect_domain("データ分析レポートを作成", {})
        assert domain == "analysis"

    def test_creative_domain_detected(self):
        domain = self.decomposer.detect_domain("広告コピーを作成してCTRを上げる", {})
        assert domain == "creative"

    def test_plan_domain_detected(self):
        domain = self.decomposer.detect_domain("実装計画を立てて開発ワークフローを設計", {})
        assert domain == "plan"

    def test_review_domain_detected(self):
        domain = self.decomposer.detect_domain("コードレビューを実施して品質確認", {})
        assert domain == "review"

    def test_general_fallback_domain(self):
        domain = self.decomposer.detect_domain("ランダムなテキスト xyz abc 123", {})
        assert domain == "general"

    def test_decompose_returns_subtasks(self):
        subtasks = self.decomposer.decompose("SEO記事を書いて検索順位を上げる", max_subtasks=4)
        assert len(subtasks) >= 1
        assert all(isinstance(st, SubTask) for st in subtasks)

    def test_decompose_max_subtasks_respected(self):
        subtasks = self.decomposer.decompose("SEO記事を書く", max_subtasks=2)
        assert len(subtasks) <= 2

    def test_decompose_max_subtasks_ceiling(self):
        subtasks = self.decomposer.decompose("分析レポート作成", max_subtasks=100)
        assert len(subtasks) <= 8

    def test_subtask_has_unique_ids(self):
        subtasks = self.decomposer.decompose("計画を立てて実装する", max_subtasks=4)
        ids = [st.task_id for st in subtasks]
        assert len(ids) == len(set(ids)), "task_id が重複している"

    def test_subtask_dependencies_chain(self):
        subtasks = self.decomposer.decompose("計画を立てて実装する", max_subtasks=3)
        if len(subtasks) >= 2:
            # 2番目以降は先行タスクに依存するはず
            assert len(subtasks[1].depends_on) == 1
            assert subtasks[1].depends_on[0] == subtasks[0].task_id

    def test_subtask_metadata_has_domain(self):
        subtasks = self.decomposer.decompose("SEO記事", max_subtasks=2)
        for st in subtasks:
            assert "domain" in st.metadata

    def test_decompose_with_context(self):
        ctx = {"target": "AI記事", "market": "日本"}
        subtasks = self.decomposer.decompose("コンテンツ作成", context=ctx, max_subtasks=3)
        assert len(subtasks) >= 1

    def test_general_decompose(self):
        subtasks = self.decomposer.decompose("何か処理する", max_subtasks=3)
        assert len(subtasks) >= 1

    def test_context_influences_detection(self):
        # context に seo キーワードが入っている場合
        domain = self.decomposer.detect_domain("作業する", {"task": "seo最適化"})
        assert domain == "seo"


# ---------------------------------------------------------------------------
# Section C: AgentSpawner (Phase 2: Spawn)
# ---------------------------------------------------------------------------

class TestAgentSpawner:
    def setup_method(self):
        self.spawner = AgentSpawner(base_url="http://test:8000")

    def _make_subtask(self, name="テスト", n=1):
        tasks = []
        for i in range(n):
            tasks.append(SubTask(
                task_id=f"t{i+1}-{uuid.uuid4().hex[:4]}",
                name=f"{name}{i+1}",
                description=f"説明{i+1}",
                metadata={"goal_fragment": "テストゴール"},
            ))
        return tasks

    def test_spawn_returns_specs(self):
        subtasks = self._make_subtask(n=3)
        specs = self.spawner.spawn(subtasks)
        assert len(specs) == 3
        assert all(isinstance(s, SubAgentSpec) for s in specs)

    def test_spawn_1to1_correspondence(self):
        subtasks = self._make_subtask(n=2)
        specs = self.spawner.spawn(subtasks)
        assert len(specs) == len(subtasks)

    def test_spec_agent_id_format(self):
        subtasks = self._make_subtask(n=1)
        specs = self.spawner.spawn(subtasks)
        assert specs[0].agent_id.startswith("agent-")

    def test_spec_endpoint_contains_base_url(self):
        subtasks = self._make_subtask(n=1)
        specs = self.spawner.spawn(subtasks)
        assert "http://test:8000" in specs[0].endpoint

    def test_spec_payload_has_prompt(self):
        subtasks = self._make_subtask(n=1)
        specs = self.spawner.spawn(subtasks)
        assert "prompt" in specs[0].payload

    def test_spec_payload_contains_task_name(self):
        subtasks = [SubTask(
            task_id="t1", name="キーワードリサーチ", description="KW 調査",
            metadata={"goal_fragment": "SEO強化"},
        )]
        specs = self.spawner.spawn(subtasks)
        assert "キーワードリサーチ" in specs[0].payload["prompt"]

    def test_spawn_empty_list(self):
        specs = self.spawner.spawn([])
        assert specs == []

    def test_spawn_max_new_tokens(self):
        subtasks = self._make_subtask(n=1)
        specs = self.spawner.spawn(subtasks, max_new_tokens=512)
        assert specs[0].payload["max_new_tokens"] == 512


# ---------------------------------------------------------------------------
# Section D: ParallelExecutor (Phase 3: asyncio 並列実行)
# ---------------------------------------------------------------------------

class TestParallelExecutor:
    def setup_method(self):
        self.executor = ParallelExecutor(max_concurrent=3)

    def _make_spec(self, n=3, task_name="タスク") -> List[SubAgentSpec]:
        specs = []
        for i in range(n):
            st = SubTask(
                task_id=f"t{i+1}", name=f"{task_name}{i+1}", description="d"
            )
            specs.append(SubAgentSpec(
                agent_id=f"ag{i+1}",
                task=st,
                endpoint="/generate",
                payload={"prompt": f"プロンプト{i+1}"},
            ))
        return specs

    def test_parallel_execute_all_succeed(self):
        specs = self._make_spec(n=3)
        call_fn = lambda ep, pl: f"出力: {pl['prompt'][:10]}"
        results = asyncio.run(self.executor.execute(specs, call_fn))
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_parallel_execute_order_preserved(self):
        specs = self._make_spec(n=4)
        call_fn = lambda ep, pl: pl["prompt"]
        results = asyncio.run(self.executor.execute(specs, call_fn))
        for i, r in enumerate(results):
            assert r.agent_id == f"ag{i+1}"

    def test_parallel_execute_error_handling(self):
        specs = self._make_spec(n=2)
        def failing_fn(ep, pl):
            raise RuntimeError("API 呼び出し失敗")
        results = asyncio.run(self.executor.execute(specs, failing_fn))
        assert all(not r.success for r in results)
        assert all(r.error is not None for r in results)

    def test_parallel_execute_partial_failure(self):
        specs = self._make_spec(n=3)
        def partial_fn(ep, pl):
            prompt = pl.get("prompt", "")
            if "プロンプト2" in prompt:
                raise ValueError("エラー2")
            return "正常出力"
        results = asyncio.run(self.executor.execute(specs, partial_fn))
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 2
        assert len(failures) == 1

    def test_parallel_execute_latency_recorded(self):
        specs = self._make_spec(n=2)
        call_fn = lambda ep, pl: "ok"
        results = asyncio.run(self.executor.execute(specs, call_fn))
        for r in results:
            assert r.latency_ms >= 0

    def test_parallel_execute_empty_specs(self):
        results = asyncio.run(self.executor.execute([], lambda ep, pl: ""))
        assert results == []

    def test_execute_single_helper_success(self):
        st = SubTask(task_id="t1", name="T", description="D")
        spec = SubAgentSpec(agent_id="a1", task=st, endpoint="/gen", payload={"prompt": "test"})
        result = asyncio.run(_execute_single(spec, lambda ep, pl: "成功出力"))
        assert result.success is True
        assert result.output == "成功出力"

    def test_execute_single_helper_failure(self):
        st = SubTask(task_id="t1", name="T", description="D")
        spec = SubAgentSpec(agent_id="a1", task=st, endpoint="/gen", payload={})
        def boom(ep, pl): raise Exception("テストエラー")
        result = asyncio.run(_execute_single(spec, boom))
        assert result.success is False
        assert "テストエラー" in result.error

    def test_max_concurrent_semaphore(self):
        """max_concurrent=1 でも全タスクが完了すること"""
        executor = ParallelExecutor(max_concurrent=1)
        specs = self._make_spec(n=4)
        call_fn = lambda ep, pl: "出力"
        results = asyncio.run(executor.execute(specs, call_fn))
        assert len(results) == 4


# ---------------------------------------------------------------------------
# Section E: ResultVerifier (Phase 4: 品質チェック)
# ---------------------------------------------------------------------------

class TestResultVerifier:
    def setup_method(self):
        self.verifier = ResultVerifier(min_len=20, good_len=100)

    def _make_result(
        self, output="", success=True, error=None, agent_id="a1"
    ) -> HermesAgentResult:
        return HermesAgentResult(
            agent_id=agent_id,
            task_id="t1",
            task_name="テスト",
            output=output,
            latency_ms=100.0,
            success=success,
            error=error,
        )

    def test_good_output_passes(self):
        r = self._make_result(output="これは十分な長さの品質の高い出力テキストです。詳細な情報を含んでいます。")
        vr = self.verifier.verify_one(r)
        assert vr.passed is True
        assert vr.score > 0.3

    def test_empty_output_fails(self):
        r = self._make_result(output="")
        vr = self.verifier.verify_one(r)
        assert vr.passed is False
        assert vr.score == 0.0

    def test_failed_execution_fails(self):
        r = self._make_result(output="", success=False, error="エラー")
        vr = self.verifier.verify_one(r)
        assert vr.passed is False
        assert len(vr.issues) > 0

    def test_short_output_low_score(self):
        r = self._make_result(output="短い")
        vr = self.verifier.verify_one(r)
        assert vr.score <= 0.4

    def test_error_marker_in_output_penalizes_score(self):
        long_output = "Error: 処理に失敗しました。" + "詳細情報。" * 20
        r = self._make_result(output=long_output)
        vr = self.verifier.verify_one(r)
        assert len(vr.issues) > 0  # マーカーが検出される

    def test_verified_output_stripped(self):
        r = self._make_result(output="  スペースある出力  これは十分な長さです  " * 5)
        vr = self.verifier.verify_one(r)
        if vr.verified_output:
            assert not vr.verified_output.startswith(" ")

    def test_verify_multiple(self):
        results = [
            self._make_result("良い出力その1。詳細なテキストが入っています。品質高め。" * 2, agent_id="a1"),
            self._make_result("", success=False, error="err", agent_id="a2"),
            self._make_result("短い", agent_id="a3"),
        ]
        vrs = self.verifier.verify(results)
        assert len(vrs) == 3
        assert vrs[0].passed is True
        assert vrs[1].passed is False

    def test_score_range_0_to_1(self):
        for output in ["", "短", "a" * 50, "b" * 200]:
            r = self._make_result(output=output)
            vr = self.verifier.verify_one(r)
            assert 0.0 <= vr.score <= 1.0


# ---------------------------------------------------------------------------
# Section F: ReportBuilder (Phase 5: レポート生成)
# ---------------------------------------------------------------------------

class TestReportBuilder:
    def setup_method(self):
        self.builder = ReportBuilder()

    def _make_subtask(self, tid="t1") -> SubTask:
        return SubTask(task_id=tid, name=f"タスク{tid}", description="説明")

    def _make_agent_result(self, tid="t1", success=True, output="テスト出力です。品質OK。") -> HermesAgentResult:
        return HermesAgentResult(
            agent_id=f"a-{tid}", task_id=tid, task_name=f"タスク{tid}",
            output=output, latency_ms=100.0, success=success,
        )

    def _make_vr(self, tid="t1", passed=True, score=0.8, output="テスト出力です。") -> VerificationResult:
        return VerificationResult(
            agent_id=f"a-{tid}", task_id=tid, task_name=f"タスク{tid}",
            passed=passed, score=score, issues=[],
            verified_output=output if passed else "",
        )

    def test_build_basic(self):
        subtasks = [self._make_subtask("t1")]
        ars = [self._make_agent_result("t1")]
        vrs = [self._make_vr("t1")]
        timings = {"plan_ms": 5.0, "spawn_ms": 2.0, "execute_ms": 100.0, "verify_ms": 3.0, "report_ms": 1.0}
        rpt = self.builder.build("run1", "テストゴール", subtasks, ars, vrs, timings)
        assert isinstance(rpt, HermesReport)
        assert rpt.run_id == "run1"

    def test_success_rate_all_success(self):
        subtasks = [self._make_subtask(f"t{i}") for i in range(3)]
        ars = [self._make_agent_result(f"t{i}") for i in range(3)]
        vrs = [self._make_vr(f"t{i}") for i in range(3)]
        rpt = self.builder.build("r", "goal", subtasks, ars, vrs, {"execute_ms": 300.0})
        assert rpt.success_rate == pytest.approx(1.0)

    def test_success_rate_partial(self):
        subtasks = [self._make_subtask(f"t{i}") for i in range(2)]
        ars = [
            self._make_agent_result("t0", success=True),
            self._make_agent_result("t1", success=False, output=""),
        ]
        vrs = [
            self._make_vr("t0", passed=True),
            self._make_vr("t1", passed=False, score=0.0, output=""),
        ]
        rpt = self.builder.build("r", "goal", subtasks, ars, vrs, {"execute_ms": 100.0})
        assert rpt.success_rate == pytest.approx(0.5)

    def test_overall_score_average(self):
        subtasks = [self._make_subtask(f"t{i}") for i in range(2)]
        ars = [self._make_agent_result(f"t{i}") for i in range(2)]
        vrs = [
            self._make_vr("t0", score=0.8),
            self._make_vr("t1", score=0.4),
        ]
        rpt = self.builder.build("r", "goal", subtasks, ars, vrs, {"execute_ms": 100.0})
        assert rpt.overall_score == pytest.approx(0.6)

    def test_final_output_contains_passed_sections(self):
        subtasks = [self._make_subtask("t1")]
        ars = [self._make_agent_result("t1", output="パス済み出力テキスト")]
        vrs = [self._make_vr("t1", passed=True, output="パス済み出力テキスト")]
        rpt = self.builder.build("r", "goal", subtasks, ars, vrs, {"execute_ms": 50.0})
        assert "パス済み出力テキスト" in rpt.final_output

    def test_final_output_fallback_all_fail(self):
        subtasks = [self._make_subtask("t1")]
        ars = [self._make_agent_result("t1", success=False, output="")]
        vrs = [self._make_vr("t1", passed=False, score=0.1, output="")]
        rpt = self.builder.build("r", "goal", subtasks, ars, vrs, {"execute_ms": 10.0})
        assert rpt.final_output  # 空でないこと

    def test_phase_timings_recorded(self):
        subtasks = [self._make_subtask()]
        ars = [self._make_agent_result()]
        vrs = [self._make_vr()]
        timings = {"plan_ms": 10.0, "spawn_ms": 5.0}
        rpt = self.builder.build("r", "goal", subtasks, ars, vrs, timings)
        assert rpt.phase_timings["plan_ms"] == pytest.approx(10.0)

    def test_total_latency_is_sum_of_timings(self):
        subtasks = [self._make_subtask()]
        ars = [self._make_agent_result()]
        vrs = [self._make_vr()]
        timings = {"plan_ms": 10.0, "execute_ms": 90.0}
        rpt = self.builder.build("r", "goal", subtasks, ars, vrs, timings)
        assert rpt.total_latency_ms == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Section G: HermesOrchestrator (フルパイプライン)
# ---------------------------------------------------------------------------

class TestHermesOrchestrator:
    def _make_orch(self, n=3) -> HermesOrchestrator:
        """モック call_fn を使った HermesOrchestrator"""
        def mock_call(ep, pl):
            prompt = pl.get("prompt", "")
            return (
                f"エージェント出力: {prompt[:30]}\n"
                "詳細なタスク実行結果を返します。品質は高く保たれています。"
            )
        return HermesOrchestrator(
            base_url="http://mock:8000",
            max_subtasks=n,
            max_concurrent=n,
            call_fn=mock_call,
        )

    def test_plan_returns_subtasks(self):
        orch = self._make_orch()
        subtasks = orch.plan("SEO記事を書いて検索順位を改善する")
        assert len(subtasks) >= 1
        assert all(isinstance(st, SubTask) for st in subtasks)

    def test_spawn_returns_specs(self):
        orch = self._make_orch()
        subtasks = orch.plan("データ分析レポートを作成する")
        specs = orch.spawn(subtasks)
        assert len(specs) == len(subtasks)

    def test_verify_returns_verifications(self):
        orch = self._make_orch()
        ar = HermesAgentResult(
            agent_id="a1", task_id="t1", task_name="T",
            output="検証用の十分な長さの出力テキストです。品質も良好です。" * 2,
            latency_ms=100.0, success=True,
        )
        vrs = orch.verify([ar])
        assert len(vrs) == 1
        assert isinstance(vrs[0], VerificationResult)

    def test_run_returns_hermes_report(self):
        orch = self._make_orch(n=3)
        rpt = orch.run("SEO記事を書いてトラフィックを増やす")
        assert isinstance(rpt, HermesReport)

    def test_run_report_has_subtasks(self):
        orch = self._make_orch(n=2)
        rpt = orch.run("広告コピーを作成する")
        assert len(rpt.subtasks) >= 1

    def test_run_report_has_agent_results(self):
        orch = self._make_orch(n=2)
        rpt = orch.run("データ分析")
        assert len(rpt.agent_results) >= 1

    def test_run_report_has_verification(self):
        orch = self._make_orch(n=2)
        rpt = orch.run("計画を立てる")
        assert len(rpt.verification_results) >= 1

    def test_run_report_score_range(self):
        orch = self._make_orch(n=2)
        rpt = orch.run("タスクを実行する")
        assert 0.0 <= rpt.overall_score <= 1.0

    def test_run_report_success_rate_range(self):
        orch = self._make_orch(n=3)
        rpt = orch.run("SEO最適化を実施する")
        assert 0.0 <= rpt.success_rate <= 1.0

    def test_run_report_phase_timings(self):
        orch = self._make_orch(n=2)
        rpt = orch.run("レビューを実施する")
        assert "plan_ms" in rpt.phase_timings
        assert "execute_ms" in rpt.phase_timings

    def test_run_async(self):
        orch = self._make_orch(n=2)
        rpt = asyncio.run(orch.run_async("非同期テスト"))
        assert isinstance(rpt, HermesReport)

    def test_run_report_final_output_nonempty(self):
        orch = self._make_orch(n=2)
        rpt = orch.run("何かを達成する")
        assert rpt.final_output  # 空でない

    def test_run_with_context(self):
        orch = self._make_orch(n=2)
        ctx = {"domain": "AI技術", "target_audience": "エンジニア"}
        rpt = orch.run("技術記事を書く", context=ctx)
        assert isinstance(rpt, HermesReport)

    def test_local_call_fn_returns_string(self):
        output = _local_call_fn("/generate", {"prompt": "テスト"})
        assert isinstance(output, str)
        assert len(output) > 10

    def test_report_summary_contains_goal(self):
        orch = self._make_orch(n=2)
        rpt = orch.run("マイゴール")
        assert "マイゴール" in rpt.summary()


# ---------------------------------------------------------------------------
# Section H: API エンドポイント (/v1/hermes/*)
# ---------------------------------------------------------------------------

import serve.api as api_module


@pytest.fixture(scope="module")
def hermes_client():
    """HermesOrchestrator テスト用 TestClient"""
    from fastapi.testclient import TestClient
    from open_mythos.main import MythosConfig, OpenMythos
    from open_mythos.agents import OpenMythosLLM

    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.eos_token_id = 50256
    tok.decode.side_effect = lambda ids, **kwargs: "response " * max(len(ids), 1)
    tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}

    cfg = MythosConfig(
        vocab_size=50257, dim=64, n_heads=4, n_kv_heads=2, max_seq_len=128,
        max_loop_iters=2, prelude_layers=1, coda_layers=1, attn_type="gqa",
        n_experts=4, n_shared_experts=1, n_experts_per_tok=1, expert_dim=32,
        act_threshold=0.99, lora_rank=4, kv_lora_rank=32, q_lora_rank=64,
        qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=8,
    )
    model = OpenMythos(cfg)
    model.eval()

    api_module.state.model = model
    api_module.state.tokenizer = tok
    api_module.state.device = torch.device("cpu")
    api_module.state.n_params = sum(p.numel() for p in model.parameters())
    api_module.state.llm = OpenMythosLLM(model=model, device="cpu")
    api_module.state.agents = {}

    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        yield c


class TestHermesPlanEndpoint:
    def test_plan_200(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/plan",
            json={"goal": "SEO記事を書いて検索順位を向上させる", "max_subtasks": 3},
        )
        assert r.status_code == 200

    def test_plan_response_has_subtasks(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/plan",
            json={"goal": "データ分析レポートを作成する", "max_subtasks": 3},
        )
        data = r.json()
        assert "subtasks" in data
        assert isinstance(data["subtasks"], list)
        assert len(data["subtasks"]) >= 1

    def test_plan_response_has_goal(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/plan",
            json={"goal": "目標テキスト", "max_subtasks": 2},
        )
        data = r.json()
        assert data["goal"] == "目標テキスト"

    def test_plan_response_subtask_count(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/plan",
            json={"goal": "テスト", "max_subtasks": 2},
        )
        data = r.json()
        assert data["subtask_count"] == len(data["subtasks"])

    def test_plan_subtask_has_required_fields(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/plan",
            json={"goal": "計画タスク", "max_subtasks": 2},
        )
        data = r.json()
        for st in data["subtasks"]:
            assert "task_id" in st
            assert "name" in st
            assert "description" in st

    def test_plan_with_context(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/plan",
            json={
                "goal": "広告コピー作成",
                "context": {"brand": "テストブランド"},
                "max_subtasks": 2,
            },
        )
        assert r.status_code == 200

    def test_plan_missing_goal_422(self, hermes_client):
        r = hermes_client.post("/v1/hermes/plan", json={"max_subtasks": 2})
        assert r.status_code == 422


class TestHermesRunEndpoint:
    def test_run_200(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/run",
            json={
                "goal": "SEO記事を書く",
                "max_subtasks": 2,
                "max_concurrent": 2,
            },
        )
        assert r.status_code == 200

    def test_run_response_has_run_id(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/run",
            json={"goal": "テストゴール", "max_subtasks": 2},
        )
        data = r.json()
        assert "run_id" in data
        assert data["run_id"]  # 非空

    def test_run_response_has_final_output(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/run",
            json={"goal": "広告コピー作成", "max_subtasks": 2},
        )
        data = r.json()
        assert "final_output" in data
        assert data["final_output"]

    def test_run_response_has_subtasks(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/run",
            json={"goal": "分析レポート作成", "max_subtasks": 2},
        )
        data = r.json()
        assert "subtasks" in data
        assert len(data["subtasks"]) >= 1

    def test_run_response_has_scores(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/run",
            json={"goal": "テスト", "max_subtasks": 2},
        )
        data = r.json()
        assert "overall_score" in data
        assert "success_rate" in data
        assert 0.0 <= data["overall_score"] <= 1.0
        assert 0.0 <= data["success_rate"] <= 1.0

    def test_run_response_has_phase_timings(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/run",
            json={"goal": "計画実行テスト", "max_subtasks": 2},
        )
        data = r.json()
        assert "phase_timings" in data
        assert "plan_ms" in data["phase_timings"]

    def test_run_response_has_verification(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/run",
            json={"goal": "品質チェックテスト", "max_subtasks": 2},
        )
        data = r.json()
        assert "verification_results" in data
        assert isinstance(data["verification_results"], list)

    def test_run_missing_goal_422(self, hermes_client):
        r = hermes_client.post("/v1/hermes/run", json={"max_subtasks": 2})
        assert r.status_code == 422

    def test_run_with_context(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/run",
            json={
                "goal": "コンテキスト付きテスト",
                "context": {"target": "エンジニア"},
                "max_subtasks": 2,
            },
        )
        assert r.status_code == 200

    def test_run_subtask_count_matches(self, hermes_client):
        r = hermes_client.post(
            "/v1/hermes/run",
            json={"goal": "テスト", "max_subtasks": 2},
        )
        data = r.json()
        assert data["subtask_count"] == len(data["subtasks"])
