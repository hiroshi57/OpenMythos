"""
Sprint 80F — TraceCompiler テスト
arXiv:2608.02680 の OpenMythos 移植検証
累計: 4541 + ? → 目標 +60 PASS
"""
from __future__ import annotations

import time
from typing import Dict, Any

import pytest

from open_mythos.skills.trace_compiler import (
    # Enums
    StepType, WorkflowStepKind, CompilationStatus, ExecutionStatus,
    # Data models
    TraceStep, AgentTrace, SkillPattern, SkillStore,
    WorkflowStep, CompiledWorkflow, CompilationResult,
    StepExecutionResult, ExecutionResult,
    # Core classes
    TraceMiner, TraceStore, TraceCompiler, WorkflowExecutor,
    WorkflowStore, TraceCompilerPipeline,
    # Helpers
    _lcs_length, _lcs_sequence,
    # Constants
    MIN_SKILL_FREQUENCY, MIN_SKILL_LENGTH, MAX_WORKFLOW_STEPS,
)


# ─── ヘルパー ─────────────────────────────────────────────────────

def make_step(action: str, step_type: StepType = StepType.TOOL_CALL,
              inputs=None, outputs=None, success=True) -> TraceStep:
    return TraceStep(
        step_id=f"s-{action}",
        step_type=step_type,
        action=action,
        inputs=inputs or {},
        outputs=outputs or {},
        success=success,
    )


def make_trace(actions: list[str], task: str = "test-task",
               success: bool = True) -> AgentTrace:
    steps = [make_step(a) for a in actions]
    return AgentTrace(task=task, steps=steps, success=success)


# ─── 定数 ────────────────────────────────────────────────────────

class TestConstants:
    def test_min_frequency(self):
        assert MIN_SKILL_FREQUENCY >= 2

    def test_min_length(self):
        assert MIN_SKILL_LENGTH >= 2

    def test_max_steps(self):
        assert MAX_WORKFLOW_STEPS >= 10


# ─── Enums ───────────────────────────────────────────────────────

class TestEnums:
    def test_step_types(self):
        assert len(StepType) == 6
        assert StepType.LLM_CALL  == "llm_call"
        assert StepType.TOOL_CALL == "tool_call"
        assert StepType.OUTPUT    == "output"

    def test_workflow_step_kinds(self):
        assert WorkflowStepKind.DETERMINISTIC == "deterministic"
        assert WorkflowStepKind.LLM_ASSISTED  == "llm_assisted"

    def test_compilation_status(self):
        assert CompilationStatus.SUCCESS == "success"
        assert CompilationStatus.PARTIAL == "partial"
        assert CompilationStatus.FAILED  == "failed"

    def test_execution_status(self):
        assert ExecutionStatus.SUCCESS == "success"
        assert ExecutionStatus.FAILED  == "failed"
        assert ExecutionStatus.PARTIAL == "partial"


# ─── TraceStep ───────────────────────────────────────────────────

class TestTraceStep:
    def test_to_dict_keys(self):
        s = make_step("search", StepType.RETRIEVAL, {"query": "q"}, {"result": "r"})
        d = s.to_dict()
        for k in ("step_id", "step_type", "action", "inputs", "outputs",
                   "duration_ms", "success", "timestamp"):
            assert k in d

    def test_signature_format(self):
        s = make_step("search", StepType.RETRIEVAL)
        assert s.signature() == "retrieval:search"

    def test_signature_llm(self):
        s = make_step("generate", StepType.LLM_CALL)
        assert s.signature() == "llm_call:generate"


# ─── AgentTrace ──────────────────────────────────────────────────

class TestAgentTrace:
    def test_signatures(self):
        t = make_trace(["search", "parse", "output"])
        sigs = t.signatures()
        assert sigs == ["tool_call:search", "tool_call:parse", "tool_call:output"]

    def test_to_dict_keys(self):
        t = make_trace(["a", "b"])
        d = t.to_dict()
        for k in ("trace_id", "task", "step_count", "success", "total_ms", "steps"):
            assert k in d

    def test_step_count(self):
        t = make_trace(["a", "b", "c"])
        assert t.to_dict()["step_count"] == 3

    def test_failed_trace(self):
        t = make_trace(["a"], success=False)
        assert t.success is False


# ─── SkillPattern ─────────────────────────────────────────────────

class TestSkillPattern:
    def _make(self):
        return SkillPattern(
            name="search→parse",
            signatures=["tool_call:search", "tool_call:parse"],
            frequency=3,
            avg_success_rate=0.9,
            source_traces=["t1", "t2", "t3"],
        )

    def test_length(self):
        s = self._make()
        assert s.length == 2

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        for k in ("skill_id", "name", "signatures", "length", "frequency",
                   "avg_success_rate", "source_traces"):
            assert k in d

    def test_frequency_in_dict(self):
        assert self._make().to_dict()["frequency"] == 3


# ─── SkillStore ──────────────────────────────────────────────────

class TestSkillStore:
    def test_add_get(self):
        store = SkillStore()
        skill = SkillPattern(name="s1", signatures=["a"], frequency=2)
        store.add(skill)
        assert store.get(skill.skill_id) is skill

    def test_count(self):
        store = SkillStore()
        for _ in range(3):
            store.add(SkillPattern(name="x", signatures=["a"], frequency=1))
        assert store.count() == 3

    def test_list_sorted_by_freq(self):
        store = SkillStore()
        store.add(SkillPattern(name="low",  signatures=["a"], frequency=1))
        store.add(SkillPattern(name="high", signatures=["b"], frequency=5))
        skills = store.list_all()
        assert skills[0].frequency >= skills[-1].frequency

    def test_top_k(self):
        store = SkillStore()
        for i in range(5):
            store.add(SkillPattern(name=f"s{i}", signatures=["x"], frequency=i))
        assert len(store.top_k(3)) == 3

    def test_remove(self):
        store = SkillStore()
        skill = SkillPattern(name="s", signatures=["a"], frequency=2)
        store.add(skill)
        assert store.remove(skill.skill_id) is True
        assert store.get(skill.skill_id) is None

    def test_remove_missing(self):
        store = SkillStore()
        assert store.remove("nonexistent") is False


# ─── LCS ヘルパー ─────────────────────────────────────────────────

class TestLCS:
    def test_lcs_length_same(self):
        assert _lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_lcs_length_no_common(self):
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_lcs_length_partial(self):
        assert _lcs_length(["a", "b", "c"], ["a", "c"]) == 2

    def test_lcs_sequence_result(self):
        common = _lcs_sequence(["a", "b", "c", "d"], ["b", "c"])
        assert common == ["b", "c"]

    def test_lcs_sequence_empty(self):
        assert _lcs_sequence([], ["a"]) == []

    def test_lcs_sequence_full(self):
        a = ["x", "y", "z"]
        assert _lcs_sequence(a, a) == a


# ─── TraceMiner ──────────────────────────────────────────────────

class TestTraceMiner:
    def _make_traces(self):
        """共通パターン search→parse を持つ 3 本のトレース。"""
        return [
            make_trace(["search", "parse", "output"]),
            make_trace(["fetch", "search", "parse", "store"]),
            make_trace(["search", "parse", "validate", "output"]),
        ]

    def test_mine_finds_skills(self):
        miner = TraceMiner(min_frequency=2, min_length=2)
        skills = miner.mine_skills(self._make_traces())
        assert len(skills) > 0

    def test_mine_common_pattern(self):
        miner = TraceMiner(min_frequency=2, min_length=2)
        skills = miner.mine_skills(self._make_traces())
        sigs_set = [frozenset(s.signatures) for s in skills]
        # search→parse が共通パターンとして出現するはず
        expected = frozenset(["tool_call:search", "tool_call:parse"])
        assert any(expected.issubset(s) for s in sigs_set)

    def test_mine_frequency_filter(self):
        miner = TraceMiner(min_frequency=3, min_length=2)
        traces = self._make_traces()
        skills = miner.mine_skills(traces)
        for s in skills:
            assert s.frequency >= 3

    def test_mine_empty_traces(self):
        miner = TraceMiner()
        assert miner.mine_skills([]) == []

    def test_mine_no_successful(self):
        miner = TraceMiner()
        traces = [make_trace(["a", "b"], success=False) for _ in range(3)]
        assert miner.mine_skills(traces) == []

    def test_mine_skill_length(self):
        miner = TraceMiner(min_frequency=2, min_length=2)
        skills = miner.mine_skills(self._make_traces())
        for s in skills:
            assert s.length >= 2

    def test_is_subsequence_true(self):
        assert TraceMiner._is_subsequence(["a", "b"], ["x", "a", "y", "b"]) is True

    def test_is_subsequence_false(self):
        assert TraceMiner._is_subsequence(["b", "a"], ["a", "b"]) is False

    def test_auto_name(self):
        name = TraceMiner._auto_name(["tool_call:search", "tool_call:parse"])
        assert "search" in name or "parse" in name


# ─── TraceStore ──────────────────────────────────────────────────

class TestTraceStore:
    def test_add_get(self):
        store = TraceStore()
        t = make_trace(["a"])
        store.add(t)
        assert store.get(t.trace_id) is t

    def test_count(self):
        store = TraceStore()
        for _ in range(4):
            store.add(make_trace(["a"]))
        assert store.count() == 4

    def test_list_successful(self):
        store = TraceStore()
        store.add(make_trace(["a"], success=True))
        store.add(make_trace(["b"], success=False))
        assert len(store.list_successful()) == 1

    def test_remove(self):
        store = TraceStore()
        t = make_trace(["a"])
        store.add(t)
        assert store.remove(t.trace_id) is True
        assert store.count() == 0

    def test_get_missing(self):
        store = TraceStore()
        assert store.get("not-exist") is None


# ─── TraceCompiler ────────────────────────────────────────────────

class TestTraceCompiler:
    def _setup(self):
        traces = [
            make_trace(["search", "parse", "output"]),
            make_trace(["fetch", "search", "parse", "store"]),
            make_trace(["search", "parse", "validate", "output"]),
        ]
        miner = TraceMiner(min_frequency=2, min_length=2)
        skills = miner.mine_skills(traces)
        return traces, skills

    def test_compile_success(self):
        traces, skills = self._setup()
        compiler = TraceCompiler()
        result = compiler.compile(traces, skills)
        assert result.status in (CompilationStatus.SUCCESS, CompilationStatus.PARTIAL)

    def test_compile_returns_workflow(self):
        traces, skills = self._setup()
        result = TraceCompiler().compile(traces, skills, name="test-wf")
        assert isinstance(result.workflow, CompiledWorkflow)
        assert result.workflow.name == "test-wf"

    def test_compile_has_steps(self):
        traces, skills = self._setup()
        result = TraceCompiler().compile(traces, skills)
        assert result.workflow.step_count > 0

    def test_compile_determinism_score(self):
        traces, skills = self._setup()
        result = TraceCompiler().compile(traces, skills)
        assert 0.0 <= result.workflow.determinism_score <= 1.0

    def test_compile_no_traces(self):
        result = TraceCompiler().compile([], [])
        assert result.status == CompilationStatus.FAILED

    def test_compile_no_skills(self):
        traces = [make_trace(["a", "b"]), make_trace(["a", "b"])]
        result = TraceCompiler().compile(traces, [])
        # スキルなし → 全ステップ LLM_ASSISTED → PARTIAL or SUCCESS
        assert result.workflow.step_count > 0
        assert result.workflow.determinism_score == 0.0

    def test_compile_coverage(self):
        traces, skills = self._setup()
        result = TraceCompiler().compile(traces, skills)
        assert 0.0 <= result.coverage <= 1.0

    def test_compile_source_traces(self):
        traces, skills = self._setup()
        result = TraceCompiler().compile(traces, skills)
        assert len(result.workflow.source_traces) > 0


# ─── CompiledWorkflow ────────────────────────────────────────────

class TestCompiledWorkflow:
    def _make_wf(self):
        steps = [
            WorkflowStep(kind=WorkflowStepKind.DETERMINISTIC, action="a"),
            WorkflowStep(kind=WorkflowStepKind.DETERMINISTIC, action="b"),
            WorkflowStep(kind=WorkflowStepKind.LLM_ASSISTED,  action="c"),
        ]
        return CompiledWorkflow(name="test", steps=steps)

    def test_determinism_score(self):
        wf = self._make_wf()
        assert abs(wf.determinism_score - 2/3) < 1e-6

    def test_det_steps(self):
        wf = self._make_wf()
        assert len(wf.deterministic_steps) == 2

    def test_llm_steps(self):
        wf = self._make_wf()
        assert len(wf.llm_steps) == 1

    def test_to_dict_keys(self):
        d = self._make_wf().to_dict()
        for k in ("workflow_id", "name", "step_count", "determinism_score",
                   "status", "steps"):
            assert k in d

    def test_empty_workflow_score(self):
        wf = CompiledWorkflow()
        assert wf.determinism_score == 0.0


# ─── WorkflowExecutor ────────────────────────────────────────────

class TestWorkflowExecutor:
    def _make_wf(self):
        steps = [
            WorkflowStep(kind=WorkflowStepKind.DETERMINISTIC, action="add"),
            WorkflowStep(kind=WorkflowStepKind.LLM_ASSISTED,  action="generate"),
        ]
        return CompiledWorkflow(name="exec-test", steps=steps)

    def test_execute_with_handler(self):
        def add_handler(step, ctx):
            return {"result": ctx.get("x", 0) + 1}

        executor = WorkflowExecutor(step_handlers={"add": add_handler})
        wf = self._make_wf()
        result = executor.execute(wf, {"x": 5})
        assert result.status in (ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL)
        assert result.success_count > 0

    def test_execute_with_llm_handler(self):
        def llm_h(step, ctx):
            return {"llm_output": "done"}

        executor = WorkflowExecutor(llm_handler=llm_h)
        wf = self._make_wf()
        result = executor.execute(wf)
        assert result.llm_calls > 0

    def test_execute_no_handler_still_ok(self):
        executor = WorkflowExecutor()
        wf = self._make_wf()
        result = executor.execute(wf)
        # ハンドラなしはスキップ扱い（エラーではない）
        assert result.step_count == 2

    def test_execution_result_structure(self):
        executor = WorkflowExecutor()
        result = executor.execute(self._make_wf())
        d = result.to_dict()
        for k in ("execution_id", "workflow_id", "status", "total_ms",
                   "success_count", "step_count", "llm_calls"):
            assert k in d

    def test_handler_exception_marks_failed(self):
        def bad_handler(step, ctx):
            raise ValueError("simulated error")

        executor = WorkflowExecutor(step_handlers={"add": bad_handler})
        result = executor.execute(self._make_wf())
        failed = [r for r in result.step_results if not r.success]
        assert len(failed) > 0

    def test_register_handler(self):
        executor = WorkflowExecutor()
        executor.register_handler("new_action", lambda s, c: {"x": 1})
        assert "new_action" in executor._handlers


# ─── WorkflowStore ────────────────────────────────────────────────

class TestWorkflowStore:
    def test_add_get(self):
        store = WorkflowStore()
        wf = CompiledWorkflow(name="wf1")
        store.add(wf)
        assert store.get(wf.workflow_id) is wf

    def test_list_sorted_by_det_score(self):
        store = WorkflowStore()
        wf_low = CompiledWorkflow(steps=[WorkflowStep(kind=WorkflowStepKind.LLM_ASSISTED)])
        wf_high = CompiledWorkflow(steps=[WorkflowStep(kind=WorkflowStepKind.DETERMINISTIC)])
        store.add(wf_low)
        store.add(wf_high)
        lst = store.list_all()
        assert lst[0].determinism_score >= lst[-1].determinism_score

    def test_count(self):
        store = WorkflowStore()
        for _ in range(3):
            store.add(CompiledWorkflow())
        assert store.count() == 3


# ─── TraceCompilerPipeline ────────────────────────────────────────

class TestTraceCompilerPipeline:
    def _setup_pipeline(self):
        pipeline = TraceCompilerPipeline(min_frequency=2, min_skill_length=2)
        for actions in [
            ["search", "parse", "output"],
            ["fetch", "search", "parse", "store"],
            ["search", "parse", "validate"],
        ]:
            pipeline.ingest(make_trace(actions))
        return pipeline

    def test_ingest_increases_count(self):
        p = TraceCompilerPipeline()
        p.ingest(make_trace(["a"]))
        assert p.trace_store.count() == 1

    def test_mine_populates_skills(self):
        p = self._setup_pipeline()
        skills = p.mine()
        assert len(skills) > 0
        assert p.skill_store.count() > 0

    def test_compile_after_mine(self):
        p = self._setup_pipeline()
        p.mine()
        result = p.compile(name="end-to-end")
        assert result.status != CompilationStatus.FAILED

    def test_run_workflow(self):
        p = self._setup_pipeline()
        p.mine()
        result = p.compile()
        wf_id = result.workflow.workflow_id
        exec_result = p.run(wf_id)
        assert exec_result is not None

    def test_run_missing_workflow(self):
        p = TraceCompilerPipeline()
        assert p.run("nonexistent") is None

    def test_summary_keys(self):
        p = self._setup_pipeline()
        s = p.summary()
        for k in ("traces", "skills", "workflows", "top_skills"):
            assert k in s

    def test_summary_trace_count(self):
        p = self._setup_pipeline()
        assert p.summary()["traces"] == 3
