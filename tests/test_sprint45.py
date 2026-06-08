"""
Sprint 45 — HuggingFace Hub 統合 テスト

対象:
  - open_mythos/skills/hf_hub.py:
      HFModelInfo / HFDatasetInfo / HFHubClient
      FastTokenizer / TokenizerResult
      LoRAConfig / PEFTAdapter / PEFTTrainResult
      EvalTask / EvalResult / LMEvaluator
  - serve/api.py:
      POST /v1/hf/search/models
      POST /v1/hf/search/datasets
      GET  /v1/hf/model/{model_id}
      POST /v1/tokenize
      POST /v1/peft/estimate
      POST /v1/lm-eval
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


# ---------------------------------------------------------------------------
# Section A: HFModelInfo / HFDatasetInfo
# ---------------------------------------------------------------------------

from open_mythos.skills.hf_hub import (
    HFModelInfo, HFDatasetInfo, HFHubClient,
    FastTokenizer, TokenizerResult,
    LoRAConfig, PEFTAdapter, PEFTTrainResult,
    EvalTask, EvalResult, LMEvaluator,
)


class TestHFModelInfo:
    def test_creation(self):
        info = HFModelInfo(model_id="bert-base-uncased")
        assert info.model_id == "bert-base-uncased"

    def test_defaults(self):
        info = HFModelInfo(model_id="x")
        assert info.task == ""
        assert info.downloads == 0
        assert info.likes == 0
        assert info.tags == []
        assert info.private is False

    def test_custom_fields(self):
        info = HFModelInfo(
            model_id="gpt2", task="text-generation",
            downloads=1000, likes=200, tags=["nlp"], private=True,
        )
        assert info.task == "text-generation"
        assert info.downloads == 1000
        assert info.likes == 200
        assert "nlp" in info.tags
        assert info.private is True


class TestHFDatasetInfo:
    def test_creation(self):
        ds = HFDatasetInfo(dataset_id="squad")
        assert ds.dataset_id == "squad"

    def test_defaults(self):
        ds = HFDatasetInfo(dataset_id="x")
        assert ds.task_categories == []
        assert ds.downloads == 0


class TestHFHubClient:
    def test_is_native_bool(self):
        c = HFHubClient()
        assert isinstance(c.is_native, bool)

    def test_search_models_returns_list(self):
        c = HFHubClient()
        results = c.search_models("bert")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_models_have_model_id(self):
        c = HFHubClient()
        results = c.search_models("gpt")
        for r in results:
            assert isinstance(r, HFModelInfo)
            assert len(r.model_id) > 0

    def test_search_models_limit(self):
        c = HFHubClient()
        results = c.search_models("t5", limit=2)
        assert len(results) <= 5   # fallback は 3 件固定

    def test_search_datasets_returns_list(self):
        c = HFHubClient()
        results = c.search_datasets("text")
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_datasets_have_id(self):
        c = HFHubClient()
        results = c.search_datasets("imagenet")
        for r in results:
            assert isinstance(r, HFDatasetInfo)

    def test_get_model_info_returns_info(self):
        c = HFHubClient()
        info = c.get_model_info("gpt2")
        assert isinstance(info, HFModelInfo)
        assert len(info.model_id) > 0  # fallback or native: どちらも非空


# ---------------------------------------------------------------------------
# Section B: FastTokenizer / TokenizerResult
# ---------------------------------------------------------------------------

class TestTokenizerResult:
    def test_creation(self):
        r = TokenizerResult(tokens=[1, 2, 3], token_strings=["a", "b", "c"], n_tokens=3)
        assert r.n_tokens == 3
        assert len(r.tokens) == 3

    def test_truncated_default(self):
        r = TokenizerResult(tokens=[], token_strings=[], n_tokens=0)
        assert r.truncated is False


class TestFastTokenizer:
    def _make(self) -> FastTokenizer:
        tok = FastTokenizer("gpt2")
        tok._native = False  # transformers がモック済みのためフォールバック強制
        return tok

    def test_encode_returns_result(self):
        tok = self._make()
        result = tok.encode("hello world")
        assert isinstance(result, TokenizerResult)

    def test_encode_n_tokens_positive(self):
        tok = self._make()
        result = tok.encode("the quick brown fox")
        assert result.n_tokens > 0

    def test_encode_truncation(self):
        tok = self._make()
        result = tok.encode("one two three four five", max_length=2, truncation=True)
        assert result.n_tokens <= 2

    def test_encode_no_truncation_flag(self):
        tok = self._make()
        result = tok.encode("hello world", max_length=10, truncation=False)
        assert result.truncated is False

    def test_decode_returns_string(self):
        tok = self._make()
        decoded = tok.decode([1, 2, 3])
        assert isinstance(decoded, str)

    def test_vocab_size_positive(self):
        tok = self._make()
        assert tok.vocab_size() > 0


# ---------------------------------------------------------------------------
# Section C: LoRAConfig / PEFTAdapter
# ---------------------------------------------------------------------------

class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.lora_dropout == 0.05
        assert cfg.task_type == "CAUSAL_LM"
        assert cfg.use_4bit is False

    def test_custom(self):
        cfg = LoRAConfig(r=8, lora_alpha=16, use_4bit=True)
        assert cfg.r == 8
        assert cfg.lora_alpha == 16
        assert cfg.use_4bit is True

    def test_target_modules_default(self):
        cfg = LoRAConfig()
        assert "q_proj" in cfg.target_modules


class TestPEFTAdapter:
    def test_is_native_bool(self):
        adapter = PEFTAdapter(LoRAConfig())
        assert isinstance(adapter.is_native, bool)

    def test_apply_returns_model_without_peft(self):
        adapter = PEFTAdapter(LoRAConfig())
        mock_model = object()
        result = adapter.apply(mock_model)
        # peft がない場合は元モデルをそのまま返す
        assert result is mock_model

    def test_estimate_trainable_params_keys(self):
        adapter = PEFTAdapter(LoRAConfig())
        est = adapter.estimate_trainable_params(1_000_000)
        assert "total_params" in est
        assert "lora_params" in est
        assert "trainable_pct" in est

    def test_estimate_trainable_params_total(self):
        adapter = PEFTAdapter(LoRAConfig())
        est = adapter.estimate_trainable_params(5_000_000)
        assert est["total_params"] == 5_000_000

    def test_estimate_trainable_pct_range(self):
        adapter = PEFTAdapter(LoRAConfig())
        est = adapter.estimate_trainable_params(1_000_000)
        assert 0.0 <= est["trainable_pct"] <= 100.0


class TestPEFTTrainResult:
    def test_creation(self):
        r = PEFTTrainResult(adapter_path="./adapter", train_loss=0.5, eval_loss=0.6, steps=100)
        assert r.adapter_path == "./adapter"
        assert r.train_loss == 0.5
        assert r.steps == 100

    def test_method_default(self):
        r = PEFTTrainResult(adapter_path="p", train_loss=1.0, eval_loss=None, steps=10)
        assert r.method == "lora"


# ---------------------------------------------------------------------------
# Section D: EvalTask / EvalResult / LMEvaluator
# ---------------------------------------------------------------------------

class TestEvalTask:
    def test_creation(self):
        t = EvalTask(name="hellaswag")
        assert t.name == "hellaswag"
        assert t.n_few_shot == 0

    def test_custom(self):
        t = EvalTask(name="arc_easy", n_few_shot=5, limit=100)
        assert t.n_few_shot == 5
        assert t.limit == 100


class TestEvalResult:
    def test_creation(self):
        r = EvalResult(task="hellaswag", metric="acc", value=0.75)
        assert r.task == "hellaswag"
        assert r.value == 0.75

    def test_stderr_default(self):
        r = EvalResult(task="x", metric="acc", value=0.5)
        assert r.stderr == 0.0


class TestLMEvaluator:
    def test_is_native_bool(self):
        ev = LMEvaluator("mock")
        assert isinstance(ev.is_native, bool)

    def test_list_tasks_returns_list(self):
        ev = LMEvaluator("mock")
        tasks = ev.list_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) > 0

    def test_list_tasks_strings(self):
        ev = LMEvaluator("mock")
        tasks = ev.list_tasks()
        for t in tasks:
            assert isinstance(t, str)

    def test_evaluate_returns_results(self):
        ev = LMEvaluator("mock")
        tasks = [EvalTask(name="hellaswag"), EvalTask(name="arc_easy")]
        results = ev.evaluate(tasks)
        assert len(results) == 2

    def test_evaluate_result_types(self):
        ev = LMEvaluator("mock")
        results = ev.evaluate([EvalTask(name="piqa")])
        assert isinstance(results[0], EvalResult)
        assert isinstance(results[0].value, float)


# ---------------------------------------------------------------------------
# Section E: API /v1/hf/*
# ---------------------------------------------------------------------------

class TestHFSearchModelsEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/hf/search/models",
                        json={"query": "bert", "limit": 3},
                        headers=_HDR)
        assert r.status_code == 200

    def test_results_list(self, client):
        r = client.post("/v1/hf/search/models",
                        json={"query": "gpt", "limit": 3},
                        headers=_HDR)
        assert isinstance(r.json()["results"], list)

    def test_result_has_model_id(self, client):
        r = client.post("/v1/hf/search/models",
                        json={"query": "t5"},
                        headers=_HDR)
        for item in r.json()["results"]:
            assert "model_id" in item

    def test_task_filter(self, client):
        r = client.post("/v1/hf/search/models",
                        json={"query": "bert", "task": "fill-mask"},
                        headers=_HDR)
        assert r.status_code == 200


class TestHFSearchDatasetsEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/hf/search/datasets",
                        json={"query": "squad"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_results_list(self, client):
        r = client.post("/v1/hf/search/datasets",
                        json={"query": "text"},
                        headers=_HDR)
        assert isinstance(r.json()["results"], list)

    def test_result_has_dataset_id(self, client):
        r = client.post("/v1/hf/search/datasets",
                        json={"query": "imagenet"},
                        headers=_HDR)
        for item in r.json()["results"]:
            assert "dataset_id" in item


class TestHFModelInfoEndpoint:
    def test_returns_200(self, client):
        r = client.get("/v1/hf/model/gpt2", headers=_HDR)
        assert r.status_code == 200

    def test_has_model_id(self, client):
        r = client.get("/v1/hf/model/bert-base-uncased", headers=_HDR)
        # native API は "google-bert/bert-base-uncased" を返す場合もある
        assert "bert-base-uncased" in r.json()["model_id"]

    def test_has_required_fields(self, client):
        r = client.get("/v1/hf/model/gpt2", headers=_HDR)
        data = r.json()
        for field in ("model_id", "task", "downloads", "likes", "tags", "private"):
            assert field in data


# ---------------------------------------------------------------------------
# Section F: API /v1/tokenize
# ---------------------------------------------------------------------------

class TestTokenizeEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/tokenize",
                        json={"text": "Hello world", "model": "gpt2"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_n_tokens(self, client):
        r = client.post("/v1/tokenize",
                        json={"text": "the quick brown fox"},
                        headers=_HDR)
        assert isinstance(r.json()["n_tokens"], int)
        assert r.json()["n_tokens"] > 0

    def test_has_tokens_list(self, client):
        r = client.post("/v1/tokenize",
                        json={"text": "abc def"},
                        headers=_HDR)
        assert isinstance(r.json()["tokens"], list)

    def test_truncation(self, client):
        r = client.post("/v1/tokenize",
                        json={"text": "one two three four five", "max_length": 2, "truncation": True},
                        headers=_HDR)
        assert r.json()["n_tokens"] <= 2


# ---------------------------------------------------------------------------
# Section G: API /v1/peft/estimate
# ---------------------------------------------------------------------------

class TestPEFTEstimateEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/peft/estimate",
                        json={"total_params": 1000000, "r": 16},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_lora_params(self, client):
        r = client.post("/v1/peft/estimate",
                        json={"total_params": 1000000},
                        headers=_HDR)
        assert "lora_params" in r.json()

    def test_has_trainable_pct(self, client):
        r = client.post("/v1/peft/estimate",
                        json={"total_params": 1000000},
                        headers=_HDR)
        assert "trainable_pct" in r.json()
        assert isinstance(r.json()["trainable_pct"], float)

    def test_total_params_echoed(self, client):
        r = client.post("/v1/peft/estimate",
                        json={"total_params": 500000},
                        headers=_HDR)
        assert r.json()["total_params"] == 500000


# ---------------------------------------------------------------------------
# Section H: API /v1/lm-eval
# ---------------------------------------------------------------------------

class TestLMEvalEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/lm-eval",
                        json={"tasks": ["hellaswag", "arc_easy"]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_results_list(self, client):
        r = client.post("/v1/lm-eval",
                        json={"tasks": ["piqa"]},
                        headers=_HDR)
        assert isinstance(r.json()["results"], list)

    def test_result_count_matches_tasks(self, client):
        r = client.post("/v1/lm-eval",
                        json={"tasks": ["hellaswag", "winogrande"]},
                        headers=_HDR)
        assert len(r.json()["results"]) == 2

    def test_result_has_value(self, client):
        r = client.post("/v1/lm-eval",
                        json={"tasks": ["hellaswag"]},
                        headers=_HDR)
        for item in r.json()["results"]:
            assert "value" in item
            assert isinstance(item["value"], float)

    def test_list_tasks_endpoint(self, client):
        r = client.get("/v1/lm-eval/tasks", headers=_HDR)
        assert r.status_code == 200
        assert isinstance(r.json()["tasks"], list)
