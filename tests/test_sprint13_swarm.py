"""
Sprint 13.2.1 テスト — SwarmOrchestrator

並列マルチエージェント実行の全戦略を検証する。
モデルは module-scoped fixture で 1 回だけ構築してテスト間で共有する。
"""

from __future__ import annotations

import pytest
import torch

from open_mythos.main import OpenMythos
from open_mythos.swarm import (
    SwarmAgentResult,
    SwarmConfig,
    SwarmOrchestrator,
    SwarmResult,
)
from open_mythos.variants import mythos_nano
from open_mythos import (
    SwarmConfig as SwarmConfigExported,
    SwarmOrchestrator as SwarmOrchestratorExported,
)


# ---------------------------------------------------------------------------
# RNG isolation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _rng_isolation():
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


# ---------------------------------------------------------------------------
# Shared model fixture (built once per module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    cfg = mythos_nano()
    return OpenMythos(cfg).eval()


@pytest.fixture(scope="module")
def swarm_cfg():
    return SwarmConfig(n_agents=3, max_workers=3, strategy="broadcast")


@pytest.fixture(scope="module")
def swarm(model, swarm_cfg):
    """Re-usable orchestrator (max_new_tokens=2 for speed)."""
    orch = SwarmOrchestrator(model, swarm_cfg, max_new_tokens=2, top_k=5)
    yield orch
    orch.shutdown()


# ---------------------------------------------------------------------------
# 1. SwarmConfig
# ---------------------------------------------------------------------------


class TestSwarmConfig:
    def test_defaults(self):
        cfg = SwarmConfig()
        assert cfg.n_agents == 4
        assert cfg.max_workers == 4
        assert cfg.strategy == "broadcast"
        assert cfg.agent_name_prefix == "agent"

    def test_custom(self):
        cfg = SwarmConfig(n_agents=8, strategy="vote", agent_name_prefix="worker")
        assert cfg.n_agents == 8
        assert cfg.strategy == "vote"
        assert cfg.agent_name_prefix == "worker"

    def test_exported_from_package(self):
        assert SwarmConfigExported is SwarmConfig


# ---------------------------------------------------------------------------
# 2. SwarmAgentResult
# ---------------------------------------------------------------------------


class TestSwarmAgentResult:
    def test_ok_when_no_error(self):
        r = SwarmAgentResult(
            task_id="t0", agent_id=0, agent_name="a0",
            output="hello", latency_ms=10.0
        )
        assert r.ok is True

    def test_not_ok_when_error(self):
        r = SwarmAgentResult(
            task_id="t0", agent_id=0, agent_name="a0",
            output="", latency_ms=5.0, error="timeout"
        )
        assert r.ok is False

    def test_fields(self):
        r = SwarmAgentResult(
            task_id="t1", agent_id=2, agent_name="agent_2",
            output="result", latency_ms=20.5
        )
        assert r.task_id == "t1"
        assert r.agent_id == 2
        assert r.agent_name == "agent_2"
        assert r.output == "result"
        assert r.latency_ms == pytest.approx(20.5)


# ---------------------------------------------------------------------------
# 3. SwarmResult
# ---------------------------------------------------------------------------


class TestSwarmResult:
    def _make_results(self, n: int, n_err: int = 0) -> list:
        results = []
        for i in range(n):
            error = "err" if i < n_err else None
            results.append(SwarmAgentResult(
                task_id=f"t{i}", agent_id=i, agent_name=f"a{i}",
                output="" if error else f"out_{i}", latency_ms=1.0,
                error=error,
            ))
        return results

    def test_n_agents(self):
        sr = SwarmResult("broadcast", self._make_results(4), "out", 100.0)
        assert sr.n_agents == 4

    def test_n_successful(self):
        sr = SwarmResult("broadcast", self._make_results(4, n_err=1), "out", 100.0)
        assert sr.n_successful == 3

    def test_success_rate_all_ok(self):
        sr = SwarmResult("broadcast", self._make_results(4), "out", 100.0)
        assert sr.success_rate == pytest.approx(1.0)

    def test_success_rate_partial(self):
        sr = SwarmResult("broadcast", self._make_results(4, n_err=2), "out", 100.0)
        assert sr.success_rate == pytest.approx(0.5)

    def test_success_rate_empty(self):
        sr = SwarmResult("broadcast", [], "out", 0.0)
        assert sr.success_rate == pytest.approx(0.0)

    def test_agent_outputs_excludes_errors(self):
        sr = SwarmResult("broadcast", self._make_results(3, n_err=1), "out", 50.0)
        outputs = sr.agent_outputs()
        assert len(outputs) == 2
        assert all(o != "" for o in outputs)


# ---------------------------------------------------------------------------
# 4. SwarmOrchestrator — construction
# ---------------------------------------------------------------------------


class TestSwarmOrchestratorConstruction:
    def test_agents_created(self, model):
        cfg = SwarmConfig(n_agents=3)
        orch = SwarmOrchestrator(model, cfg, max_new_tokens=2)
        assert len(orch._agents) == 3
        orch.shutdown()

    def test_agent_names(self, model):
        cfg = SwarmConfig(n_agents=2, agent_name_prefix="worker")
        orch = SwarmOrchestrator(model, cfg, max_new_tokens=2)
        names = [a.agent_name for a in orch._agents]
        assert names == ["worker_0", "worker_1"]
        orch.shutdown()

    def test_default_cfg(self, model):
        orch = SwarmOrchestrator(model, max_new_tokens=2)
        assert orch.cfg.n_agents == 4
        orch.shutdown()

    def test_context_manager(self, model):
        cfg = SwarmConfig(n_agents=2)
        with SwarmOrchestrator(model, cfg, max_new_tokens=2) as orch:
            assert len(orch._agents) == 2
        # pool should be shut down after exit (no exception)

    def test_exported_from_package(self):
        assert SwarmOrchestratorExported is SwarmOrchestrator


# ---------------------------------------------------------------------------
# 5. broadcast strategy
# ---------------------------------------------------------------------------


class TestBroadcast:
    def test_returns_swarm_result(self, swarm):
        result = swarm.broadcast("hello")
        assert isinstance(result, SwarmResult)

    def test_strategy_label(self, swarm):
        result = swarm.broadcast("hello")
        assert result.strategy == "broadcast"

    def test_n_results_equals_n_agents(self, swarm):
        result = swarm.broadcast("hello")
        assert result.n_agents == swarm.cfg.n_agents

    def test_all_successful(self, swarm):
        result = swarm.broadcast("hello")
        assert result.n_successful == swarm.cfg.n_agents

    def test_latency_positive(self, swarm):
        result = swarm.broadcast("hello")
        assert result.total_latency_ms > 0

    def test_final_output_is_string(self, swarm):
        result = swarm.broadcast("hello")
        assert isinstance(result.final_output, str)

    def test_agent_histories_reset_after(self, swarm):
        """Agents should have empty history after broadcast."""
        swarm.broadcast("test message")
        for agent in swarm._agents:
            assert agent._history == []


# ---------------------------------------------------------------------------
# 6. map strategy
# ---------------------------------------------------------------------------


class TestMap:
    def test_n_results_equals_n_tasks(self, swarm):
        tasks = ["task A", "task B", "task C"]
        result = swarm.map(tasks)
        assert result.n_agents == len(tasks)

    def test_strategy_label(self, swarm):
        result = swarm.map(["x", "y"])
        assert result.strategy == "map"

    def test_single_task(self, swarm):
        result = swarm.map(["single task"])
        assert result.n_agents == 1

    def test_more_tasks_than_agents(self, swarm):
        """Round-robin: 5 tasks, 3 agents."""
        tasks = [f"task_{i}" for i in range(5)]
        result = swarm.map(tasks)
        assert result.n_agents == 5

    def test_all_successful(self, swarm):
        result = swarm.map(["a", "b", "c"])
        assert result.n_successful == 3

    def test_empty_task_list_no_crash(self, swarm):
        """空タスクリストで map() がクラッシュしないことを確認。"""
        result = swarm.map([])
        assert result.n_agents == 0
        assert result.success_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 7. pipeline strategy
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_strategy_label(self, swarm):
        result = swarm.pipeline("input text")
        assert result.strategy == "pipeline"

    def test_n_results_equals_n_agents(self, swarm):
        result = swarm.pipeline("input text")
        assert result.n_agents == swarm.cfg.n_agents

    def test_final_output_is_string(self, swarm):
        result = swarm.pipeline("start")
        assert isinstance(result.final_output, str)

    def test_stages_applied(self, swarm):
        """stages プレフィックス付きで正常完了すること。"""
        stages = ["Step 1:", "Step 2:", "Step 3:"]
        result = swarm.pipeline("original task", stages=stages)
        assert result.n_agents == swarm.cfg.n_agents

    def test_histories_reset_after_pipeline(self, swarm):
        """Pipeline 完了後、全エージェント履歴がリセットされること。"""
        swarm.pipeline("test pipeline")
        for agent in swarm._agents:
            assert agent._history == []


# ---------------------------------------------------------------------------
# 8. vote strategy
# ---------------------------------------------------------------------------


class TestVote:
    def test_strategy_label(self, swarm):
        result = swarm.vote("hello")
        assert result.strategy == "vote"

    def test_returns_one_string(self, swarm):
        result = swarm.vote("hello")
        assert isinstance(result.final_output, str)

    def test_winner_in_agent_outputs(self, swarm):
        """多数決勝者はいずれかのエージェント出力と一致すること。"""
        result = swarm.vote("hello")
        if result.final_output:  # 空文字でなければ
            assert result.final_output in result.agent_outputs()

    def test_n_results_equals_n_agents(self, swarm):
        result = swarm.vote("hello")
        assert result.n_agents == swarm.cfg.n_agents


# ---------------------------------------------------------------------------
# 9. run() dispatch
# ---------------------------------------------------------------------------


class TestRunDispatch:
    def test_list_always_maps(self, swarm):
        result = swarm.run(["t1", "t2"], strategy="vote")  # list → map
        assert result.strategy == "map"

    def test_broadcast_by_default(self, swarm):
        result = swarm.run("hello")  # cfg.strategy == "broadcast"
        assert result.strategy == "broadcast"

    def test_override_to_pipeline(self, swarm):
        result = swarm.run("hello", strategy="pipeline")
        assert result.strategy == "pipeline"

    def test_override_to_vote(self, swarm):
        result = swarm.run("hello", strategy="vote")
        assert result.strategy == "vote"

    def test_stages_passed_to_pipeline(self, swarm):
        stages = ["A:", "B:", "C:"]
        result = swarm.run("hello", strategy="pipeline", stages=stages)
        assert result.strategy == "pipeline"
        assert result.n_agents == swarm.cfg.n_agents
