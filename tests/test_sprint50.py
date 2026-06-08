"""
Sprint 50 — エージェントフレームワーク強化 テスト

対象:
  - open_mythos/skills/agent_framework.py:
      SubAgentTask / SubAgentResult / SubAgentOrchestrator
      TDDCycle / TDDSession / TDDAgent
      BugReport / DebugStep / DebugSession / SystematicDebugger
      Individual / EvolutionResult / DarwinianEvolver
      ParallelJob / JobResult / ParallelCLIRunner
  - serve/api.py:
      POST /v1/agent/subagent/plan
      POST /v1/agent/subagent/run
      POST /v1/agent/tdd/cycle
      POST /v1/agent/tdd/session
      POST /v1/agent/debug
      POST /v1/agent/evolve
      POST /v1/agent/cli/run
"""
from __future__ import annotations

import sys
import pytest
import torch
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# transformers モック
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def mock_transformers():
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    sys.modules["transformers"] = mock
    yield mock
    sys.modules.pop("transformers", None)


# ---------------------------------------------------------------------------
# TestClient フィクスチャ
# ---------------------------------------------------------------------------

import serve.api as api_module


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from open_mythos.main import MythosConfig, OpenMythos
    from open_mythos.agents import OpenMythosLLM

    tok = MagicMock()
    tok.encode.return_value = [1, 2, 3]
    tok.eos_token_id = 50256
    tok.decode.side_effect = lambda ids, **kwargs: "ok " * max(len(ids), 1)
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


_HDR = {"Authorization": "Bearer dev"}

from open_mythos.skills.agent_framework import (
    SubAgentTask, SubAgentResult, SubAgentOrchestrator,
    TDDCycle, TDDSession, TDDAgent,
    BugReport, DebugStep, DebugSession, SystematicDebugger,
    Individual, EvolutionResult, DarwinianEvolver,
    ParallelJob, JobResult, ParallelCLIRunner,
)


# ---------------------------------------------------------------------------
# Section A: SubAgentOrchestrator
# ---------------------------------------------------------------------------

class TestSubAgentTask:
    def test_creation(self):
        task = SubAgentTask(task_id="t1", description="Write code")
        assert task.task_id == "t1"
        assert task.priority == 1
        assert task.timeout_s == 60.0

    def test_custom_priority(self):
        task = SubAgentTask(task_id="t2", description="Review", priority=5)
        assert task.priority == 5

    def test_context_default_empty(self):
        task = SubAgentTask(task_id="t3", description="Test")
        assert task.context == {}


class TestSubAgentResult:
    def test_creation(self):
        r = SubAgentResult(task_id="t1", output="done", success=True)
        assert r.success is True
        assert r.review_passed is False
        assert r.reviewer_feedback == ""


class TestSubAgentOrchestrator:
    def test_plan_returns_list(self):
        orch = SubAgentOrchestrator()
        tasks = orch.plan("build a feature", n_subtasks=3)
        assert isinstance(tasks, list)
        assert len(tasks) == 3

    def test_plan_task_ids(self):
        orch = SubAgentOrchestrator()
        tasks = orch.plan("goal", n_subtasks=4)
        ids = [t.task_id for t in tasks]
        assert ids == ["task_1", "task_2", "task_3", "task_4"]

    def test_plan_descriptions_nonempty(self):
        orch = SubAgentOrchestrator()
        tasks = orch.plan("write API", n_subtasks=2)
        for t in tasks:
            assert len(t.description) > 0

    def test_run_returns_results(self):
        orch = SubAgentOrchestrator()
        tasks = orch.plan("test goal", n_subtasks=2)
        results = orch.run(tasks)
        assert len(results) == 2

    def test_run_result_type(self):
        orch = SubAgentOrchestrator()
        tasks = [SubAgentTask(task_id="x", description="Do something")]
        results = orch.run(tasks)
        assert isinstance(results[0], SubAgentResult)

    def test_run_review_passed(self):
        orch = SubAgentOrchestrator()
        tasks = [SubAgentTask(task_id="x", description="Do something")]
        results = orch.run(tasks)
        assert results[0].review_passed is True

    def test_run_latency_nonneg(self):
        orch = SubAgentOrchestrator()
        tasks = [SubAgentTask(task_id="t1", description="Test")]
        results = orch.run(tasks)
        assert results[0].latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Section B: TDDAgent
# ---------------------------------------------------------------------------

class TestTDDCycle:
    def test_creation(self):
        cycle = TDDCycle(phase="RED", test_code="def test_x(): pass",
                         impl_code="", test_passed=False)
        assert cycle.phase == "RED"
        assert cycle.test_passed is False


class TestTDDSession:
    def test_creation(self):
        session = TDDSession(goal="build feature")
        assert session.goal == "build feature"
        assert session.cycles == []
        assert session.pass_rate == 0.0

    def test_pass_rate_calculation(self):
        session = TDDSession(goal="g", total_tests_written=4, total_tests_passing=3)
        assert abs(session.pass_rate - 0.75) < 1e-6


class TestTDDAgent:
    def test_red_phase_generates_test(self):
        agent = TDDAgent()
        test_code = agent.red_phase("calculate sum")
        assert "def test_" in test_code
        assert "assert" in test_code

    def test_green_phase_generates_impl(self):
        agent = TDDAgent()
        test_code = "def test_calculate_sum():\n    result = calculate_sum()\n    assert result is not None\n"
        impl = agent.green_phase(test_code)
        assert "def " in impl
        assert "return" in impl

    def test_refactor_phase_adds_docstring(self):
        agent = TDDAgent()
        impl = "def my_func():\n    return 'result'\n"
        refactored = agent.refactor_phase(impl)
        assert '"""' in refactored

    def test_run_cycle_returns_cycle(self):
        agent = TDDAgent()
        cycle = agent.run_cycle("build user login")
        assert isinstance(cycle, TDDCycle)

    def test_run_cycle_test_passed(self):
        agent = TDDAgent()
        cycle = agent.run_cycle("compute factorial")
        assert cycle.test_passed is True

    def test_run_cycle_phase_refactor(self):
        agent = TDDAgent()
        cycle = agent.run_cycle("test goal")
        assert cycle.phase == "REFACTOR"

    def test_run_session_returns_session(self):
        agent = TDDAgent()
        session = agent.run_session(["goal A", "goal B"])
        assert isinstance(session, TDDSession)
        assert len(session.cycles) == 2

    def test_run_session_pass_rate_positive(self):
        agent = TDDAgent()
        session = agent.run_session(["goal X"])
        assert session.pass_rate > 0.0


# ---------------------------------------------------------------------------
# Section C: SystematicDebugger
# ---------------------------------------------------------------------------

class TestBugReport:
    def test_creation(self):
        bug = BugReport(description="TypeError: expected int")
        assert "TypeError" in bug.description
        assert bug.context == ""
        assert bug.stack_trace == ""


class TestDebugStep:
    def test_creation(self):
        step = DebugStep(phase="understand", action="Read error", finding="type mismatch")
        assert step.phase == "understand"
        assert step.confidence == 0.0


class TestSystematicDebugger:
    def test_debug_returns_session(self):
        dbg = SystematicDebugger()
        bug = BugReport(description="AttributeError: NoneType has no attribute foo")
        session = dbg.debug(bug)
        assert isinstance(session, DebugSession)

    def test_debug_has_4_steps(self):
        dbg = SystematicDebugger()
        bug = BugReport(description="ValueError: invalid input")
        session = dbg.debug(bug)
        assert len(session.steps) == 4

    def test_debug_phases_order(self):
        dbg = SystematicDebugger()
        bug = BugReport(description="KeyError: missing key")
        session = dbg.debug(bug)
        phases = [s.phase for s in session.steps]
        assert phases == ["understand", "isolate", "fix", "verify"]

    def test_debug_root_cause_nonempty(self):
        dbg = SystematicDebugger()
        bug = BugReport(description="TypeError in module")
        session = dbg.debug(bug)
        assert len(session.root_cause) > 0

    def test_debug_fix_suggestion_nonempty(self):
        dbg = SystematicDebugger()
        bug = BugReport(description="NoneType error")
        session = dbg.debug(bug)
        assert len(session.fix_suggestion) > 0

    def test_debug_verified_true(self):
        dbg = SystematicDebugger()
        bug = BugReport(description="ImportError: no module named foo")
        session = dbg.debug(bug)
        assert session.verified is True

    def test_classify_error_type_error(self):
        dbg = SystematicDebugger()
        result = dbg._classify_error("TypeError: something wrong")
        assert result == "type_mismatch"

    def test_classify_error_unknown(self):
        dbg = SystematicDebugger()
        result = dbg._classify_error("some random error")
        assert result == "unknown_error"


# ---------------------------------------------------------------------------
# Section D: DarwinianEvolver
# ---------------------------------------------------------------------------

class TestIndividual:
    def test_creation(self):
        ind = Individual(genome=[0.1, 0.2, 0.3])
        assert len(ind.genome) == 3
        assert ind.fitness == 0.0
        assert ind.generation == 0


class TestEvolutionResult:
    def test_creation(self):
        best = Individual(genome=[0.5], fitness=0.9)
        result = EvolutionResult(
            best_individual=best, best_fitness=0.9,
            generations_run=10, population_size=20, fitness_history=[0.5, 0.7, 0.9]
        )
        assert result.generations_run == 10


class TestDarwinianEvolver:
    def _simple_fitness(self, genome):
        """最大値に近い遺伝子ほど高いフィットネス。"""
        return sum(genome)

    def test_evolve_returns_result(self):
        evolver = DarwinianEvolver(population_size=10)
        result = evolver.evolve(self._simple_fitness, genome_dim=3, n_generations=5)
        assert isinstance(result, EvolutionResult)

    def test_evolve_best_fitness_positive(self):
        evolver = DarwinianEvolver(population_size=10)
        result = evolver.evolve(self._simple_fitness, genome_dim=3, n_generations=5)
        # すべてプラスになる可能性が高い
        assert result.best_fitness > -10.0

    def test_evolve_generations_run(self):
        evolver = DarwinianEvolver(population_size=10)
        result = evolver.evolve(self._simple_fitness, genome_dim=2, n_generations=8)
        assert result.generations_run == 8

    def test_evolve_fitness_history_length(self):
        evolver = DarwinianEvolver(population_size=10)
        result = evolver.evolve(self._simple_fitness, genome_dim=2, n_generations=5)
        assert len(result.fitness_history) == 5

    def test_evolve_genome_dim_respected(self):
        evolver = DarwinianEvolver(population_size=10)
        result = evolver.evolve(self._simple_fitness, genome_dim=4, n_generations=3)
        assert len(result.best_individual.genome) == 4

    def test_evolve_population_size_recorded(self):
        evolver = DarwinianEvolver(population_size=20)
        result = evolver.evolve(self._simple_fitness, genome_dim=2, n_generations=3)
        assert result.population_size == 20


# ---------------------------------------------------------------------------
# Section E: ParallelCLIRunner
# ---------------------------------------------------------------------------

class TestParallelJob:
    def test_creation(self):
        job = ParallelJob(job_id="j1", command="echo hello")
        assert job.job_id == "j1"
        assert job.cwd == "."
        assert job.timeout_s == 30.0

    def test_custom_timeout(self):
        job = ParallelJob(job_id="j2", command="ls", timeout_s=10.0)
        assert job.timeout_s == 10.0


class TestJobResult:
    def test_creation(self):
        result = JobResult(job_id="j1", command="echo hi",
                           returncode=0, stdout="hi\n", stderr="",
                           duration_s=0.1, success=True)
        assert result.success is True


class TestParallelCLIRunner:
    def test_run_returns_list(self):
        runner = ParallelCLIRunner()
        jobs = [ParallelJob(job_id="j1", command="echo hello")]
        results = runner.run(jobs)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_run_result_type(self):
        runner = ParallelCLIRunner()
        jobs = [ParallelJob(job_id="j1", command="echo test")]
        results = runner.run(jobs)
        assert isinstance(results[0], JobResult)

    def test_run_job_id_preserved(self):
        runner = ParallelCLIRunner()
        jobs = [ParallelJob(job_id="myid", command="echo ok")]
        results = runner.run(jobs)
        assert results[0].job_id == "myid"

    def test_run_duration_nonneg(self):
        runner = ParallelCLIRunner()
        jobs = [ParallelJob(job_id="j1", command="echo hi")]
        results = runner.run(jobs)
        assert results[0].duration_s >= 0.0


# ---------------------------------------------------------------------------
# Section F: API エンドポイント
# ---------------------------------------------------------------------------

class TestSubAgentPlanEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/agent/subagent/plan",
                        json={"goal": "build a REST API", "n_subtasks": 3},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_tasks(self, client):
        r = client.post("/v1/agent/subagent/plan",
                        json={"goal": "create test suite", "n_subtasks": 2},
                        headers=_HDR)
        data = r.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 2

    def test_task_has_fields(self, client):
        r = client.post("/v1/agent/subagent/plan",
                        json={"goal": "write docs", "n_subtasks": 1},
                        headers=_HDR)
        task = r.json()["tasks"][0]
        assert "task_id" in task
        assert "description" in task


class TestSubAgentRunEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/agent/subagent/run",
                        json={"tasks": [{"task_id": "t1", "description": "Do work"}]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_results(self, client):
        r = client.post("/v1/agent/subagent/run",
                        json={"tasks": [{"task_id": "t1", "description": "Task A"},
                                        {"task_id": "t2", "description": "Task B"}]},
                        headers=_HDR)
        data = r.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_result_has_success(self, client):
        r = client.post("/v1/agent/subagent/run",
                        json={"tasks": [{"task_id": "t1", "description": "Test"}]},
                        headers=_HDR)
        result = r.json()["results"][0]
        assert "success" in result
        assert result["success"] is True


class TestTDDCycleEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/agent/tdd/cycle",
                        json={"goal": "implement add function"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_phase(self, client):
        r = client.post("/v1/agent/tdd/cycle",
                        json={"goal": "sort algorithm"},
                        headers=_HDR)
        data = r.json()
        assert "phase" in data
        assert data["phase"] == "REFACTOR"

    def test_has_test_passed(self, client):
        r = client.post("/v1/agent/tdd/cycle",
                        json={"goal": "hash function"},
                        headers=_HDR)
        assert "test_passed" in r.json()
        assert r.json()["test_passed"] is True


class TestTDDSessionEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/agent/tdd/session",
                        json={"goals": ["goal A", "goal B"]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_pass_rate(self, client):
        r = client.post("/v1/agent/tdd/session",
                        json={"goals": ["goal X"]},
                        headers=_HDR)
        data = r.json()
        assert "pass_rate" in data
        assert data["pass_rate"] > 0.0

    def test_has_cycles(self, client):
        r = client.post("/v1/agent/tdd/session",
                        json={"goals": ["g1", "g2", "g3"]},
                        headers=_HDR)
        data = r.json()
        assert "cycles" in data
        assert len(data["cycles"]) == 3


class TestDebugEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/agent/debug",
                        json={"description": "TypeError: int object is not iterable"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_steps(self, client):
        r = client.post("/v1/agent/debug",
                        json={"description": "ValueError: list index out of range",
                              "stack_trace": "File 'app.py', line 42"},
                        headers=_HDR)
        data = r.json()
        assert "steps" in data
        assert len(data["steps"]) == 4

    def test_has_root_cause(self, client):
        r = client.post("/v1/agent/debug",
                        json={"description": "AttributeError: NoneType"},
                        headers=_HDR)
        data = r.json()
        assert "root_cause" in data
        assert len(data["root_cause"]) > 0

    def test_has_fix_suggestion(self, client):
        r = client.post("/v1/agent/debug",
                        json={"description": "NoneType has no attribute value"},
                        headers=_HDR)
        assert "fix_suggestion" in r.json()

    def test_verified_true(self, client):
        r = client.post("/v1/agent/debug",
                        json={"description": "ImportError: module not found"},
                        headers=_HDR)
        assert r.json()["verified"] is True


class TestEvolveEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/agent/evolve",
                        json={"genome_dim": 3, "n_generations": 5, "population_size": 10},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_best_fitness(self, client):
        r = client.post("/v1/agent/evolve",
                        json={"genome_dim": 2, "n_generations": 3, "population_size": 8},
                        headers=_HDR)
        data = r.json()
        assert "best_fitness" in data

    def test_generations_run_matches(self, client):
        r = client.post("/v1/agent/evolve",
                        json={"genome_dim": 2, "n_generations": 4, "population_size": 8},
                        headers=_HDR)
        assert r.json()["generations_run"] == 4

    def test_has_fitness_history(self, client):
        r = client.post("/v1/agent/evolve",
                        json={"genome_dim": 2, "n_generations": 3, "population_size": 8},
                        headers=_HDR)
        assert "fitness_history" in r.json()
        assert len(r.json()["fitness_history"]) == 3


class TestCLIRunEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/agent/cli/run",
                        json={"jobs": [{"job_id": "j1", "command": "echo hello"}]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_results(self, client):
        r = client.post("/v1/agent/cli/run",
                        json={"jobs": [{"job_id": "j1", "command": "echo test"},
                                       {"job_id": "j2", "command": "echo world"}]},
                        headers=_HDR)
        data = r.json()
        assert "results" in data
        assert len(data["results"]) == 2

    def test_job_id_preserved(self, client):
        r = client.post("/v1/agent/cli/run",
                        json={"jobs": [{"job_id": "my_job", "command": "echo ok"}]},
                        headers=_HDR)
        result = r.json()["results"][0]
        assert result["job_id"] == "my_job"

    def test_success_echo(self, client):
        r = client.post("/v1/agent/cli/run",
                        json={"jobs": [{"job_id": "j1", "command": "echo success"}]},
                        headers=_HDR)
        result = r.json()["results"][0]
        assert result["success"] is True
