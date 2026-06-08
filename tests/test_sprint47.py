"""
Sprint 47 — 研究ツール統合 テスト

対象:
  - open_mythos/skills/research_tools.py:
      ArxivPaper / ArxivSearcher
      DSPySignature / DSPyPrediction / DSPyOptimizer
      SearchResult / WebSearcher
      KernelExecutionResult / JupyterKernelClient
  - serve/api.py:
      POST /v1/arxiv/search
      GET  /v1/arxiv/paper/{arxiv_id}
      POST /v1/dspy/predict
      POST /v1/dspy/chain-of-thought
      POST /v1/search/web
      POST /v1/search/news
      POST /v1/jupyter/execute
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

from open_mythos.skills.research_tools import (
    ArxivPaper, ArxivSearcher,
    DSPySignature, DSPyPrediction, DSPyOptimizer,
    SearchResult, WebSearcher,
    KernelExecutionResult, JupyterKernelClient,
)


# ---------------------------------------------------------------------------
# Section A: ArxivPaper / ArxivSearcher
# ---------------------------------------------------------------------------

class TestArxivPaper:
    def test_creation(self):
        p = ArxivPaper(arxiv_id="2401.00001", title="Test", authors=["A"], abstract="abc")
        assert p.arxiv_id == "2401.00001"
        assert p.title == "Test"

    def test_url_property(self):
        p = ArxivPaper(arxiv_id="2401.00001", title="T", authors=[], abstract="")
        assert "arxiv.org" in p.url
        assert "2401.00001" in p.url

    def test_categories_default(self):
        p = ArxivPaper(arxiv_id="x", title="T", authors=[], abstract="")
        assert p.categories == []


class TestArxivSearcher:
    def test_is_native_bool(self):
        s = ArxivSearcher()
        assert isinstance(s.is_native, bool)

    def test_search_returns_list(self):
        s = ArxivSearcher()
        results = s.search("transformer", max_results=3)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_result_type(self):
        s = ArxivSearcher()
        results = s.search("attention", max_results=2)
        for r in results:
            assert isinstance(r, ArxivPaper)

    def test_search_has_title(self):
        s = ArxivSearcher()
        results = s.search("llm", max_results=2)
        for r in results:
            assert len(r.title) > 0

    def test_get_by_id_returns_paper(self):
        s = ArxivSearcher()
        paper = s.get_by_id("2401.00001")
        # fallback か native かどちらでも Paper か None
        assert paper is None or isinstance(paper, ArxivPaper)


# ---------------------------------------------------------------------------
# Section B: DSPyOptimizer
# ---------------------------------------------------------------------------

class TestDSPySignature:
    def test_creation(self):
        sig = DSPySignature(
            name="QA",
            inputs={"question": "The question to answer"},
            outputs={"answer": "The answer"},
        )
        assert sig.name == "QA"
        assert "question" in sig.inputs

    def test_instructions_default(self):
        sig = DSPySignature(name="X", inputs={}, outputs={})
        assert sig.instructions == ""


class TestDSPyOptimizer:
    def test_is_native_bool(self):
        opt = DSPyOptimizer()
        assert isinstance(opt.is_native, bool)

    def test_predict_returns_prediction(self):
        opt = DSPyOptimizer()
        sig = DSPySignature(
            name="Classify",
            inputs={"text": "Input text"},
            outputs={"label": "Classification label"},
        )
        pred = opt.predict(sig, {"text": "hello world"})
        assert isinstance(pred, DSPyPrediction)

    def test_predict_outputs_have_keys(self):
        opt = DSPyOptimizer()
        sig = DSPySignature(
            name="T", inputs={"x": "input"}, outputs={"y": "output"}
        )
        pred = opt.predict(sig, {"x": "test"})
        assert "y" in pred.outputs

    def test_predict_success_flag(self):
        opt = DSPyOptimizer()
        sig = DSPySignature(name="T", inputs={"q": "question"}, outputs={"a": "answer"})
        pred = opt.predict(sig, {"q": "what?"})
        assert pred.success is True

    def test_build_chain_of_thought(self):
        opt = DSPyOptimizer()
        sig = DSPySignature(
            name="CoT",
            inputs={"premise": "A logical statement"},
            outputs={"conclusion": "The conclusion"},
        )
        cot = opt.build_chain_of_thought(sig)
        assert "premise" in cot
        assert "conclusion" in cot
        assert "step" in cot.lower()


class TestDSPyPrediction:
    def test_creation(self):
        p = DSPyPrediction(inputs={"x": 1}, outputs={"y": 2})
        assert p.inputs["x"] == 1
        assert p.success is True


# ---------------------------------------------------------------------------
# Section C: WebSearcher
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_creation(self):
        r = SearchResult(title="T", url="https://example.com", snippet="abc")
        assert r.title == "T"
        assert r.source == "web"

    def test_custom_source(self):
        r = SearchResult(title="N", url="https://news.com", snippet="x", source="news")
        assert r.source == "news"


class TestWebSearcher:
    def test_is_native_bool(self):
        ws = WebSearcher()
        assert isinstance(ws.is_native, bool)

    def test_search_returns_list(self):
        ws = WebSearcher()
        results = ws.search("OpenMythos AI", max_results=3)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_result_type(self):
        ws = WebSearcher()
        results = ws.search("python machine learning", max_results=2)
        for r in results:
            assert isinstance(r, SearchResult)

    def test_search_has_title_and_url(self):
        ws = WebSearcher()
        results = ws.search("deep learning", max_results=2)
        for r in results:
            assert len(r.title) > 0
            assert len(r.url) > 0

    def test_news_returns_list(self):
        ws = WebSearcher()
        results = ws.news("AI research", max_results=3)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Section D: JupyterKernelClient
# ---------------------------------------------------------------------------

class TestKernelExecutionResult:
    def test_creation(self):
        r = KernelExecutionResult(stdout="hello\n", stderr="", outputs=[], execution_count=1, success=True)
        assert r.stdout == "hello\n"
        assert r.success is True

    def test_defaults(self):
        r = KernelExecutionResult(stdout="", stderr="", outputs=[], execution_count=0, success=True)
        assert r.error_name == ""
        assert r.error_traceback == []


class TestJupyterKernelClient:
    def test_is_native_bool(self):
        jk = JupyterKernelClient()
        assert isinstance(jk.is_native, bool)

    def test_execute_simple_code(self):
        jk = JupyterKernelClient()
        jk._native = False  # fallback 強制
        result = jk.execute("x = 1 + 2")
        assert isinstance(result, KernelExecutionResult)
        assert result.success is True

    def test_execute_print_captured(self):
        jk = JupyterKernelClient()
        jk._native = False
        result = jk.execute("print('hello')")
        assert "hello" in result.stdout

    def test_execute_error_caught(self):
        jk = JupyterKernelClient()
        jk._native = False
        result = jk.execute("raise ValueError('test error')")
        assert result.success is False
        assert result.error_name == "ValueError"

    def test_execute_execution_count(self):
        jk = JupyterKernelClient()
        jk._native = False
        result = jk.execute("y = 42")
        assert result.execution_count >= 1


# ---------------------------------------------------------------------------
# Section E: API エンドポイント
# ---------------------------------------------------------------------------

class TestArxivSearchEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/arxiv/search",
                        json={"query": "transformer attention", "max_results": 3},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_papers_list(self, client):
        r = client.post("/v1/arxiv/search",
                        json={"query": "llm", "max_results": 2},
                        headers=_HDR)
        assert isinstance(r.json()["papers"], list)

    def test_paper_has_title(self, client):
        r = client.post("/v1/arxiv/search",
                        json={"query": "neural network", "max_results": 2},
                        headers=_HDR)
        for p in r.json()["papers"]:
            assert "title" in p


class TestArxivPaperEndpoint:
    def test_returns_200(self, client):
        r = client.get("/v1/arxiv/paper/2401.00001", headers=_HDR)
        assert r.status_code == 200

    def test_has_arxiv_id_or_null(self, client):
        r = client.get("/v1/arxiv/paper/2401.99999", headers=_HDR)
        data = r.json()
        assert "paper" in data


class TestDSPyEndpoints:
    def test_predict_returns_200(self, client):
        r = client.post("/v1/dspy/predict",
                        json={
                            "signature_name": "QA",
                            "inputs": {"question": "What is AI?"},
                            "outputs": {"answer": "The answer"},
                        },
                        headers=_HDR)
        assert r.status_code == 200

    def test_predict_has_outputs(self, client):
        r = client.post("/v1/dspy/predict",
                        json={
                            "signature_name": "T",
                            "inputs": {"x": "value"},
                            "outputs": {"y": "result"},
                        },
                        headers=_HDR)
        assert "outputs" in r.json()

    def test_cot_returns_200(self, client):
        r = client.post("/v1/dspy/chain-of-thought",
                        json={
                            "signature_name": "CoT",
                            "inputs": {"premise": "A statement"},
                            "outputs": {"conclusion": "The conclusion"},
                        },
                        headers=_HDR)
        assert r.status_code == 200

    def test_cot_has_prompt(self, client):
        r = client.post("/v1/dspy/chain-of-thought",
                        json={
                            "signature_name": "C",
                            "inputs": {"q": "question"},
                            "outputs": {"a": "answer"},
                        },
                        headers=_HDR)
        assert "prompt" in r.json()
        assert len(r.json()["prompt"]) > 10


class TestWebSearchEndpoints:
    def test_search_returns_200(self, client):
        r = client.post("/v1/search/web",
                        json={"query": "OpenMythos", "max_results": 3},
                        headers=_HDR)
        assert r.status_code == 200

    def test_search_has_results(self, client):
        r = client.post("/v1/search/web",
                        json={"query": "AI research"},
                        headers=_HDR)
        assert isinstance(r.json()["results"], list)

    def test_news_returns_200(self, client):
        r = client.post("/v1/search/news",
                        json={"query": "machine learning"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_news_has_results(self, client):
        r = client.post("/v1/search/news",
                        json={"query": "deep learning"},
                        headers=_HDR)
        assert isinstance(r.json()["results"], list)


class TestJupyterExecuteEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/jupyter/execute",
                        json={"code": "x = 1 + 2"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_success_flag(self, client):
        r = client.post("/v1/jupyter/execute",
                        json={"code": "y = 42"},
                        headers=_HDR)
        assert "success" in r.json()
        assert r.json()["success"] is True

    def test_print_captured(self, client):
        r = client.post("/v1/jupyter/execute",
                        json={"code": "print('sprint47')"},
                        headers=_HDR)
        assert "sprint47" in r.json()["stdout"]

    def test_error_detected(self, client):
        r = client.post("/v1/jupyter/execute",
                        json={"code": "raise RuntimeError('oops')"},
                        headers=_HDR)
        assert r.json()["success"] is False
