"""
Sprint 52 — DevOps・クラウド統合 テスト

対象:
  - open_mythos/skills/devops_cloud.py:
      ModalFunctionConfig / ModalRunResult / ModalRunner
      ContainerInfo / BuildResult / DockerManager
      WatchRule / FileEvent / FileWatcher
      SLiMeConfig / SLiMeResult / SLiMeModel
  - serve/api.py:
      POST /v1/modal/run
      POST /v1/modal/stub
      POST /v1/docker/containers
      POST /v1/docker/build
      POST /v1/watch/config
      POST /v1/slime/fit
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

from open_mythos.skills.devops_cloud import (
    ModalFunctionConfig, ModalRunResult, ModalRunner,
    ContainerInfo, BuildResult, DockerManager,
    WatchRule, FileEvent, FileWatcher,
    SLiMeConfig, SLiMeResult, SLiMeModel,
)


# ---------------------------------------------------------------------------
# Section A: ModalRunner
# ---------------------------------------------------------------------------

class TestModalFunctionConfig:
    def test_defaults(self):
        cfg = ModalFunctionConfig(name="my_func")
        assert cfg.gpu == ""
        assert cfg.memory == 512
        assert cfg.timeout == 300

    def test_custom(self):
        cfg = ModalFunctionConfig(name="train", gpu="A100", memory=2048)
        assert cfg.gpu == "A100"
        assert cfg.memory == 2048


class TestModalRunResult:
    def test_creation(self):
        r = ModalRunResult(output=42, run_id="abc", duration_s=1.5, gpu_used="none", success=True)
        assert r.success is True
        assert r.error == ""


class TestModalRunner:
    def test_is_native_bool(self):
        runner = ModalRunner()
        assert isinstance(runner.is_native, bool)

    def test_run_function_returns_result(self):
        runner = ModalRunner()
        cfg = ModalFunctionConfig(name="add")
        result = runner.run_function(lambda x, y: x + y, cfg, 2, 3)
        assert isinstance(result, ModalRunResult)

    def test_run_function_output(self):
        runner = ModalRunner()
        cfg = ModalFunctionConfig(name="square")
        result = runner.run_function(lambda x: x * x, cfg, 5)
        assert result.output == 25

    def test_run_function_success(self):
        runner = ModalRunner()
        cfg = ModalFunctionConfig(name="echo")
        result = runner.run_function(lambda s: s, cfg, "hello")
        assert result.success is True

    def test_run_function_duration_nonneg(self):
        runner = ModalRunner()
        cfg = ModalFunctionConfig(name="noop")
        result = runner.run_function(lambda: None, cfg)
        assert result.duration_s >= 0.0

    def test_build_stub_contains_name(self):
        runner = ModalRunner(app_name="myapp")
        cfg = ModalFunctionConfig(name="train_model")
        stub = runner.build_stub(cfg)
        assert "train_model" in stub
        assert "myapp" in stub

    def test_build_stub_has_import(self):
        runner = ModalRunner()
        cfg = ModalFunctionConfig(name="fn")
        stub = runner.build_stub(cfg)
        assert "import modal" in stub


# ---------------------------------------------------------------------------
# Section B: DockerManager
# ---------------------------------------------------------------------------

class TestContainerInfo:
    def test_creation(self):
        c = ContainerInfo(
            container_id="abc123", name="my-app", image="nginx:latest",
            status="running", ports={"80/tcp": "8080"},
        )
        assert c.status == "running"
        assert c.created == ""


class TestBuildResult:
    def test_creation(self):
        r = BuildResult(image_id="sha256:abc", tag="my-image:latest",
                        build_time_s=5.0, size_mb=120.0, success=True)
        assert r.success is True
        assert r.error == ""


class TestDockerManager:
    def test_is_native_bool(self):
        dm = DockerManager()
        assert isinstance(dm.is_native, bool)

    def test_list_containers_returns_list(self):
        dm = DockerManager()
        containers = dm.list_containers()
        assert isinstance(containers, list)

    def test_build_returns_result(self):
        dm = DockerManager()
        result = dm.build("/nonexistent/Dockerfile", "test:latest", "/nonexistent")
        assert isinstance(result, BuildResult)

    def test_build_returns_build_result_type(self):
        dm = DockerManager()
        result = dm.build("./Dockerfile", "tag:latest")
        assert hasattr(result, "success")
        assert hasattr(result, "tag")


# ---------------------------------------------------------------------------
# Section C: FileWatcher
# ---------------------------------------------------------------------------

class TestWatchRule:
    def test_creation(self):
        rule = WatchRule(path="/tmp/watch")
        assert rule.pattern == "**/*"
        assert rule.recursive is True
        assert "created" in rule.event_types

    def test_custom(self):
        rule = WatchRule(path="/app", pattern="*.py", recursive=False)
        assert rule.pattern == "*.py"
        assert rule.recursive is False


class TestFileEvent:
    def test_creation(self):
        ev = FileEvent(event_type="created", src_path="/tmp/test.txt")
        assert ev.event_type == "created"
        assert ev.dest_path == ""
        assert ev.timestamp > 0


class TestFileWatcher:
    def test_is_native_bool(self):
        fw = FileWatcher()
        assert isinstance(fw.is_native, bool)

    def test_get_recent_events_empty(self):
        fw = FileWatcher()
        events = fw.get_recent_events()
        assert isinstance(events, list)

    def test_build_config_has_rules(self):
        rules = [WatchRule(path="/app", pattern="*.py")]
        fw = FileWatcher(rules=rules)
        config = fw.build_config()
        assert "rules" in config
        assert len(config["rules"]) == 1

    def test_build_config_rule_fields(self):
        rules = [WatchRule(path="/tmp", pattern="*.log", recursive=False)]
        fw = FileWatcher(rules=rules)
        config = fw.build_config()
        rule = config["rules"][0]
        assert rule["path"] == "/tmp"
        assert rule["pattern"] == "*.log"
        assert rule["recursive"] is False

    def test_poll_returns_list(self):
        fw = FileWatcher()
        events = fw.poll(paths=[], interval_s=0.0, n_checks=0)
        assert isinstance(events, list)


# ---------------------------------------------------------------------------
# Section D: SLiMeModel
# ---------------------------------------------------------------------------

class TestSLiMeConfig:
    def test_defaults(self):
        cfg = SLiMeConfig()
        assert cfg.n_components == 64
        assert cfg.alpha == 0.01
        assert cfg.normalize is True

    def test_custom(self):
        cfg = SLiMeConfig(n_components=32, alpha=0.1)
        assert cfg.n_components == 32


class TestSLiMeModel:
    def _sample_data(self, n=20, d=8):
        import random
        rng = random.Random(42)
        return [[rng.gauss(0, 1) for _ in range(d)] for _ in range(n)]

    def test_is_native_bool(self):
        model = SLiMeModel(SLiMeConfig())
        assert isinstance(model.is_native, bool)

    def test_fit_returns_result(self):
        model = SLiMeModel(SLiMeConfig(n_components=4))
        X = self._sample_data(20, 8)
        result = model.fit(X)
        assert isinstance(result, SLiMeResult)

    def test_fit_components_list(self):
        model = SLiMeModel(SLiMeConfig(n_components=4))
        result = model.fit(self._sample_data())
        assert isinstance(result.components, list)
        assert len(result.components) > 0

    def test_fit_sparsity_in_range(self):
        model = SLiMeModel(SLiMeConfig(n_components=4))
        result = model.fit(self._sample_data())
        assert 0.0 <= result.sparsity <= 1.0

    def test_fit_reconstruction_error_nonneg(self):
        model = SLiMeModel(SLiMeConfig(n_components=4))
        result = model.fit(self._sample_data())
        assert result.reconstruction_error >= 0.0

    def test_fit_n_iter_positive(self):
        model = SLiMeModel(SLiMeConfig(n_components=4))
        result = model.fit(self._sample_data())
        assert result.n_iter > 0


# ---------------------------------------------------------------------------
# Section E: API エンドポイント
# ---------------------------------------------------------------------------

class TestModalRunEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/modal/run",
                        json={"name": "add", "operation": "sum", "args": [2, 3]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_success(self, client):
        r = client.post("/v1/modal/run",
                        json={"name": "echo_fn", "operation": "echo", "args": [5]},
                        headers=_HDR)
        assert "success" in r.json()
        assert r.json()["success"] is True

    def test_has_run_id(self, client):
        r = client.post("/v1/modal/run",
                        json={"name": "noop", "operation": "noop", "args": []},
                        headers=_HDR)
        assert "run_id" in r.json()


class TestModalStubEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/modal/stub",
                        json={"name": "train_model", "gpu": "A100", "memory": 4096},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_stub_code(self, client):
        r = client.post("/v1/modal/stub",
                        json={"name": "inference", "gpu": "", "memory": 512},
                        headers=_HDR)
        data = r.json()
        assert "stub_code" in data
        assert "import modal" in data["stub_code"]

    def test_stub_contains_name(self, client):
        r = client.post("/v1/modal/stub",
                        json={"name": "my_function"},
                        headers=_HDR)
        assert "my_function" in r.json()["stub_code"]


class TestDockerContainersEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/docker/containers",
                        json={"all": False},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_containers(self, client):
        r = client.post("/v1/docker/containers",
                        json={"all": True},
                        headers=_HDR)
        assert "containers" in r.json()
        assert isinstance(r.json()["containers"], list)


class TestDockerBuildEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/docker/build",
                        json={"dockerfile_path": "./Dockerfile",
                              "tag": "test:latest", "context": "."},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_result(self, client):
        r = client.post("/v1/docker/build",
                        json={"dockerfile_path": "./Dockerfile", "tag": "img:v1"},
                        headers=_HDR)
        data = r.json()
        assert "tag" in data
        assert "success" in data


class TestWatchConfigEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/watch/config",
                        json={"rules": [{"path": "/app", "pattern": "*.py"}]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_rules(self, client):
        r = client.post("/v1/watch/config",
                        json={"rules": [{"path": "/src", "pattern": "*.ts",
                                         "recursive": True}]},
                        headers=_HDR)
        data = r.json()
        assert "rules" in data
        assert len(data["rules"]) == 1

    def test_rule_has_path(self, client):
        r = client.post("/v1/watch/config",
                        json={"rules": [{"path": "/myapp"}]},
                        headers=_HDR)
        assert r.json()["rules"][0]["path"] == "/myapp"


class TestSLiMeFitEndpoint:
    def test_returns_200(self, client):
        X = [[float(i + j) for j in range(8)] for i in range(20)]
        r = client.post("/v1/slime/fit",
                        json={"data": X, "n_components": 4},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_sparsity(self, client):
        X = [[float(i) for i in range(8)] for _ in range(20)]
        r = client.post("/v1/slime/fit",
                        json={"data": X, "n_components": 4},
                        headers=_HDR)
        data = r.json()
        assert "sparsity" in data
        assert 0.0 <= data["sparsity"] <= 1.0

    def test_has_components(self, client):
        X = [[float(i + j * 0.1) for j in range(8)] for i in range(20)]
        r = client.post("/v1/slime/fit",
                        json={"data": X, "n_components": 4},
                        headers=_HDR)
        assert "components" in r.json()
        assert isinstance(r.json()["components"], list)
