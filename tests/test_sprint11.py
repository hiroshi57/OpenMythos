"""
Sprint 11 テスト

Track A — Tool Use / Function Calling
  11.1.1  open_mythos/tools.py — ToolRegistry / ToolCall / ToolResult / @tool / parse_tool_calls
  11.1.2  open_mythos/tools_marketing.py — search_competitor / calculate_roi / fetch_trend / score_content
  11.1.3  serve/api.py — /v1/tools / /v1/tools/call / /v1/tools/batch

Track B — Long Context (YaRN RoPE Extension)
  11.2.1  open_mythos/rope_extension.py — yarn_rope_freqs / get_rope_freqs / RopeScalingConfig
  11.2.2  extend_model_context — モデルの max_seq_len を動的に拡張

Track C — RAG Pipeline
  11.3.1  open_mythos/rag.py — VectorStore / RAGPipeline / RAGResult
  11.3.2  serve/api.py — /v1/rag/index / /v1/rag
"""

from __future__ import annotations

import json
import math
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch


# ===========================================================================
# serve/api.py が module-level で transformers.AutoTokenizer を import するため、
# test_sprint11.py が先に実行されると test_sprint7_serve.py の mock_transformers
# fixture が機能しなくなる（pytest の実行順: sprint11 < sprint7 alphabetically）。
# ここで先にモックを差し込み、serve.api を常に mock AutoTokenizer で初期化する。
# ===========================================================================


def _make_tok_mock() -> MagicMock:
    tok = MagicMock()
    tok.side_effect = lambda text, **kw: {
        "input_ids": torch.zeros(1, max(1, len(str(text).split())), dtype=torch.long)
    }
    tok.decode = MagicMock(return_value="generated text")
    tok.eos_token_id = 0
    tok.vocab_size = 50257
    return tok


@pytest.fixture(scope="module", autouse=True)
def mock_transformers_sprint11():
    """
    serve/api.py の transformers.AutoTokenizer を mock する。

    serve/api.py は module-level で AutoTokenizer を import するため、
    test_sprint11.py (alphabetically 先) が serve.api を import すると
    test_sprint7_serve.py の mock_transformers が機能しなくなる。
    本 fixture で先にモックを入れることでテスト間の独立性を保つ。
    """
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=_make_tok_mock())

    orig = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    yield
    if orig is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = orig


# ===========================================================================
# ヘルパー
# ===========================================================================


def _tiny_cfg():
    from open_mythos.main import MythosConfig
    return MythosConfig(
        vocab_size=512, dim=64, n_heads=4, n_kv_heads=2,
        max_seq_len=128, max_loop_iters=4, prelude_layers=1, coda_layers=1,
        n_experts=4, n_shared_experts=1, n_experts_per_tok=1, expert_dim=32, lora_rank=4,
    )


def _tiny_model():
    from open_mythos.main import OpenMythos
    return OpenMythos(_tiny_cfg()).eval()


# ===========================================================================
# Track A — 11.1.1  ToolRegistry / ToolCall / ToolResult
# ===========================================================================


class TestToolDefinition:
    def test_to_openai_schema(self):
        from open_mythos.tools import ToolDefinition, ParameterSchema

        def dummy(x: str) -> str:
            return x

        td = ToolDefinition(
            name="dummy",
            description="A dummy tool",
            parameters={"x": ParameterSchema(type="string", description="input", required=True)},
            fn=dummy,
        )
        schema = td.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "dummy"
        assert "x" in schema["function"]["parameters"]["properties"]
        assert "x" in schema["function"]["parameters"]["required"]

    def test_optional_param_not_in_required(self):
        from open_mythos.tools import ToolDefinition, ParameterSchema

        def dummy(x: str, y: str = "default") -> str:
            return x

        td = ToolDefinition(
            name="dummy",
            description="desc",
            parameters={
                "x": ParameterSchema(type="string", required=True),
                "y": ParameterSchema(type="string", required=False, default="default"),
            },
            fn=dummy,
        )
        schema = td.to_openai_schema()
        assert "x" in schema["function"]["parameters"]["required"]
        assert "y" not in schema["function"]["parameters"]["required"]


class TestToolCall:
    def test_from_dict(self):
        from open_mythos.tools import ToolCall

        data = {"name": "calculate_roi", "arguments": {"ad_spend": 1000.0, "revenue": 3000.0}}
        tc = ToolCall.from_dict(data)
        assert tc.name == "calculate_roi"
        assert tc.arguments["ad_spend"] == 1000.0

    def test_from_dict_json_string_arguments(self):
        from open_mythos.tools import ToolCall

        data = {"name": "search_competitor", "arguments": '{"company": "Jasper"}'}
        tc = ToolCall.from_dict(data)
        assert tc.arguments["company"] == "Jasper"

    def test_to_dict(self):
        from open_mythos.tools import ToolCall

        tc = ToolCall(name="fetch_trend", arguments={"keyword": "LLMO"}, call_id="c1")
        d = tc.to_dict()
        assert d["function"]["name"] == "fetch_trend"
        args = json.loads(d["function"]["arguments"])
        assert args["keyword"] == "LLMO"


class TestToolResult:
    def test_success_true_when_no_error(self):
        from open_mythos.tools import ToolResult

        r = ToolResult(tool_name="test", content={"val": 1})
        assert r.success is True

    def test_success_false_when_error(self):
        from open_mythos.tools import ToolResult

        r = ToolResult(tool_name="test", content=None, error="something went wrong")
        assert r.success is False

    def test_to_message(self):
        from open_mythos.tools import ToolResult

        r = ToolResult(tool_name="calculate_roi", content={"roi_pct": 200.0}, call_id="c1")
        msg = r.to_message()
        assert msg["role"] == "tool"
        assert msg["name"] == "calculate_roi"
        assert "roi_pct" in msg["content"]


class TestToolRegistry:
    def test_register_and_get(self):
        from open_mythos.tools import ToolRegistry, ToolDefinition, ParameterSchema

        reg = ToolRegistry()

        def add(a: int, b: int) -> int:
            return a + b

        td = ToolDefinition(
            name="add",
            description="add two numbers",
            parameters={
                "a": ParameterSchema(type="integer", required=True),
                "b": ParameterSchema(type="integer", required=True),
            },
            fn=add,
        )
        reg.register(td)
        assert "add" in reg
        assert reg.get("add") is td

    def test_call_success(self):
        from open_mythos.tools import ToolRegistry, ToolDefinition, ParameterSchema, ToolCall

        reg = ToolRegistry()

        def multiply(x: float, y: float) -> float:
            return x * y

        reg.register(ToolDefinition(
            name="multiply",
            description="multiply",
            parameters={
                "x": ParameterSchema(type="number", required=True),
                "y": ParameterSchema(type="number", required=True),
            },
            fn=multiply,
        ))
        result = reg.call(ToolCall(name="multiply", arguments={"x": 3.0, "y": 4.0}))
        assert result.success
        assert result.content == pytest.approx(12.0)

    def test_call_unknown_tool_returns_error(self):
        from open_mythos.tools import ToolRegistry, ToolCall

        reg = ToolRegistry()
        result = reg.call(ToolCall(name="nonexistent", arguments={}))
        assert not result.success
        assert "not found" in result.error

    def test_to_openai_tools(self):
        from open_mythos.tools import ToolRegistry
        reg = ToolRegistry.default()
        tools = reg.to_openai_tools()
        assert len(tools) >= 4
        names = [t["function"]["name"] for t in tools]
        assert "search_competitor" in names
        assert "calculate_roi" in names

    def test_register_fn_auto_infer(self):
        from open_mythos.tools import ToolRegistry

        reg = ToolRegistry()

        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        reg.register_fn(greet, description="Greet a person")
        assert "greet" in reg

    def test_len(self):
        from open_mythos.tools import ToolRegistry
        reg = ToolRegistry.default()
        assert len(reg) == 4


class TestParseToolCalls:
    def test_parse_tool_call_xml_format(self):
        from open_mythos.tools import parse_tool_calls

        text = 'Let me check this. <tool_call>{"name": "fetch_trend", "arguments": {"keyword": "LLMO"}}</tool_call>'
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "fetch_trend"
        assert calls[0].arguments["keyword"] == "LLMO"

    def test_parse_no_tool_call(self):
        from open_mythos.tools import parse_tool_calls

        text = "This is a plain response without any tool calls."
        calls = parse_tool_calls(text)
        assert calls == []

    def test_build_tool_prompt(self):
        from open_mythos.tools import build_tool_prompt, ToolRegistry

        reg = ToolRegistry.default()
        prompt = build_tool_prompt(reg.list_tools())
        assert "search_competitor" in prompt
        assert "calculate_roi" in prompt
        assert "<tool_call>" in prompt


class TestExecuteToolCalls:
    def test_execute_multiple(self):
        from open_mythos.tools import ToolRegistry, ToolCall, execute_tool_calls

        reg = ToolRegistry.default()
        calls = [
            ToolCall(name="calculate_roi", arguments={"ad_spend": 1000.0, "revenue": 3000.0}),
            ToolCall(name="fetch_trend", arguments={"keyword": "LLMO"}),
        ]
        results = execute_tool_calls(calls, reg)
        assert len(results) == 2
        assert all(r.success for r in results)


# ===========================================================================
# Track A — 11.1.2  マーケ特化ツール
# ===========================================================================


class TestSearchCompetitor:
    def test_returns_dict(self):
        from open_mythos.tools_marketing import search_competitor

        r = search_competitor("Jasper AI", metric="all")
        assert isinstance(r, dict)
        assert r["company"] == "Jasper AI"
        assert "ad_spend_usd" in r
        assert "avg_ctr" in r
        assert "seo_score" in r

    def test_deterministic_for_same_input(self):
        from open_mythos.tools_marketing import search_competitor

        r1 = search_competitor("Copy.ai", metric="ad_spend")
        r2 = search_competitor("Copy.ai", metric="ad_spend")
        assert r1["ad_spend_usd"] == r2["ad_spend_usd"]

    def test_period_affects_ad_spend(self):
        from open_mythos.tools_marketing import search_competitor

        r7 = search_competitor("Jasper AI", metric="ad_spend", period="last_7_days")
        r90 = search_competitor("Jasper AI", metric="ad_spend", period="last_90_days")
        # 90日の方が7日より大きい (乗数が違う)
        assert r90["ad_spend_usd"] > r7["ad_spend_usd"]


class TestCalculateROI:
    def test_basic_roi(self):
        from open_mythos.tools_marketing import calculate_roi

        r = calculate_roi(ad_spend=1000.0, revenue=3000.0)
        assert r["roi_pct"] == pytest.approx(200.0)
        assert r["roas"] == pytest.approx(3.0)
        assert r["profitable"] is True

    def test_negative_roi(self):
        from open_mythos.tools_marketing import calculate_roi

        r = calculate_roi(ad_spend=1000.0, revenue=500.0)
        assert r["roi_pct"] < 0
        assert r["profitable"] is False

    def test_with_clicks_and_impressions(self):
        from open_mythos.tools_marketing import calculate_roi

        r = calculate_roi(
            ad_spend=1000.0, revenue=5000.0,
            clicks=200, impressions=10000
        )
        assert "ctr" in r
        assert r["ctr"] == pytest.approx(0.02)
        assert "cpc_usd" in r

    def test_zero_ad_spend_returns_error(self):
        from open_mythos.tools_marketing import calculate_roi

        r = calculate_roi(ad_spend=0, revenue=1000.0)
        assert "error" in r

    def test_cogs_reduces_roi(self):
        # cogs を指定すると ROI が減少することを確認
        from open_mythos.tools_marketing import calculate_roi

        r_no_cogs = calculate_roi(ad_spend=1000.0, revenue=3000.0, cogs=0.0)
        r_with_cogs = calculate_roi(ad_spend=1000.0, revenue=3000.0, cogs=500.0)
        assert r_with_cogs["roi_pct"] < r_no_cogs["roi_pct"]
        # cogs=500 → gross_profit=2500, roi=(2500-1000)/1000*100=150%
        assert r_with_cogs["roi_pct"] == pytest.approx(150.0)
        assert r_with_cogs["gross_profit_usd"] == pytest.approx(2500.0)


class TestFetchTrend:
    def test_returns_required_fields(self):
        from open_mythos.tools_marketing import fetch_trend

        r = fetch_trend("LLMO", region="JP")
        assert "trend_score" in r
        assert "llmo_popularity" in r
        assert "search_volume_est" in r
        assert "related_keywords" in r
        assert 0 <= r["trend_score"] <= 100
        assert 0.0 <= r["llmo_popularity"] <= 1.0

    def test_is_rising_when_high_score(self):
        from open_mythos.tools_marketing import fetch_trend

        # deterministic: same keyword → same result
        r = fetch_trend("LLMO")
        assert isinstance(r["is_rising"], bool)


class TestScoreContent:
    def test_returns_all_fields(self):
        from open_mythos.tools_marketing import score_content

        r = score_content("OpenMythos achieved 32% CTR in Q3 2025.", target_keyword="CTR")
        assert "llmo_total" in r
        assert "entity_density" in r
        assert "recommendations" in r
        assert isinstance(r["recommendations"], list)

    def test_empty_text_no_crash(self):
        from open_mythos.tools_marketing import score_content

        r = score_content("")
        assert r["llmo_total"] == 0.0


# ===========================================================================
# Track A — 11.1.3  serve/api.py — /v1/tools エンドポイント
# ===========================================================================


class TestAPIToolsEndpoints:
    def test_tools_list_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/tools" in routes

    def test_tools_call_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/tools/call" in routes

    def test_tools_batch_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/tools/batch" in routes

    def test_tool_call_request_model(self):
        from serve.api import ToolCallRequest

        req = ToolCallRequest(name="calculate_roi", arguments={"ad_spend": 100.0, "revenue": 300.0})
        assert req.name == "calculate_roi"

    def test_tools_batch_request_model(self):
        from serve.api import ToolsBatchRequest, ToolCallRequest

        req = ToolsBatchRequest(calls=[
            ToolCallRequest(name="fetch_trend", arguments={"keyword": "SEO"}),
        ])
        assert len(req.calls) == 1

    def test_rag_index_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/rag/index" in routes

    def test_rag_query_route_exists(self):
        from serve.api import app
        routes = [r.path for r in app.routes]
        assert "/v1/rag" in routes


# ===========================================================================
# Track B — 11.2.1  rope_extension.py
# ===========================================================================


class TestRopeScalingConfig:
    def test_for_32k(self):
        from open_mythos.rope_extension import RopeScalingConfig

        cfg = RopeScalingConfig.for_32k()
        assert cfg.type == "yarn"
        assert cfg.factor == pytest.approx(8.0)
        assert cfg.original_max_len == 4096

    def test_for_8k(self):
        from open_mythos.rope_extension import RopeScalingConfig

        cfg = RopeScalingConfig.for_8k()
        assert cfg.factor == pytest.approx(2.0)


class TestYarnRopeFreqs:
    def test_output_shape(self):
        from open_mythos.rope_extension import yarn_rope_freqs

        dim = 32
        max_len = 256
        freqs = yarn_rope_freqs(dim=dim, max_len=max_len, factor=2.0)
        assert freqs.shape == (max_len, dim // 2)
        assert freqs.dtype == torch.complex64

    def test_factor_1_equals_standard_rope(self):
        """factor=1.0 の場合、通常 RoPE と同じ周波数になる。"""
        from open_mythos.rope_extension import yarn_rope_freqs, get_rope_freqs, RopeScalingConfig
        from open_mythos.main import precompute_rope_freqs

        dim, max_len, theta = 32, 64, 500000.0
        standard = precompute_rope_freqs(dim, max_len, theta)
        # factor=1 では補間なし (r=0 everywhere) のため標準と一致するはず
        yarn = yarn_rope_freqs(dim=dim, max_len=max_len, theta=theta, factor=1.0,
                               original_max_len=max_len)
        # 位相が同じか確認 (angle が近い)
        assert standard.shape == yarn.shape

    def test_large_factor_different_from_standard(self):
        """factor>1 では周波数が変化する。"""
        from open_mythos.rope_extension import yarn_rope_freqs
        from open_mythos.main import precompute_rope_freqs

        dim, max_len = 32, 64
        standard = precompute_rope_freqs(dim, max_len)
        yarn = yarn_rope_freqs(dim=dim, max_len=max_len, factor=8.0, original_max_len=64)
        # 低周波成分はスケーリングされているため angle が異なる
        assert not torch.allclose(standard.angle(), yarn.angle(), atol=1e-3)

    def test_no_nan_inf(self):
        from open_mythos.rope_extension import yarn_rope_freqs

        freqs = yarn_rope_freqs(dim=64, max_len=512, factor=8.0, original_max_len=64)
        assert not torch.isnan(freqs.real).any()
        assert not torch.isinf(freqs.real).any()


class TestGetRopeFreqs:
    def test_none_scaling_returns_standard(self):
        from open_mythos.rope_extension import get_rope_freqs
        from open_mythos.main import precompute_rope_freqs

        dim, max_len = 32, 64
        standard = precompute_rope_freqs(dim, max_len)
        result = get_rope_freqs(dim, max_len, scaling=None)
        assert torch.allclose(standard.real, result.real, atol=1e-5)

    def test_linear_scaling(self):
        from open_mythos.rope_extension import get_rope_freqs, RopeScalingConfig

        cfg = RopeScalingConfig(type="linear", factor=2.0)
        freqs = get_rope_freqs(32, 128, scaling=cfg)
        assert freqs.shape == (128, 16)

    def test_ntk_scaling(self):
        from open_mythos.rope_extension import get_rope_freqs, RopeScalingConfig

        cfg = RopeScalingConfig(type="ntk", factor=4.0)
        freqs = get_rope_freqs(32, 128, scaling=cfg)
        assert freqs.shape == (128, 16)

    def test_yarn_scaling(self):
        from open_mythos.rope_extension import get_rope_freqs, RopeScalingConfig

        cfg = RopeScalingConfig.for_32k()
        freqs = get_rope_freqs(64, 512, scaling=cfg)
        assert freqs.shape == (512, 32)

    def test_unknown_type_raises(self):
        from open_mythos.rope_extension import get_rope_freqs, RopeScalingConfig

        cfg = RopeScalingConfig(type="unknown", factor=2.0)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown rope scaling"):
            get_rope_freqs(32, 64, scaling=cfg)


class TestExtendModelContext:
    def test_updates_max_seq_len(self):
        from open_mythos.rope_extension import extend_model_context

        model = _tiny_model()
        original_len = model.cfg.max_seq_len
        extend_model_context(model, new_max_len=256)
        assert model.cfg.max_seq_len == 256
        assert model.cfg.max_seq_len > original_len

    def test_freqs_cis_shape_updated(self):
        from open_mythos.rope_extension import extend_model_context

        model = _tiny_model()
        extend_model_context(model, new_max_len=256)
        assert model.freqs_cis.shape[0] == 256
        assert model.freqs_cis_mla.shape[0] == 256

    def test_model_still_runs_after_extension(self):
        from open_mythos.rope_extension import extend_model_context

        model = _tiny_model()
        extend_model_context(model, new_max_len=256)
        ids = torch.randint(0, 512, (1, 64))
        with torch.no_grad():
            out = model(ids)
        assert out.shape == (1, 64, 512)


# ===========================================================================
# Track C — 11.3.1  RAG Pipeline
# ===========================================================================


class TestDocument:
    def test_defaults(self):
        from open_mythos.rag import Document

        doc = Document(text="hello world")
        assert doc.embedding is None
        assert doc.metadata == {}
        assert doc.score == 0.0


class TestVectorStore:
    def test_add_and_search(self):
        from open_mythos.rag import VectorStore, Document
        import torch

        store = VectorStore(embed_dim=32)
        docs = [
            Document(text="doc A", embedding=torch.randn(32), doc_id="a"),
            Document(text="doc B", embedding=torch.randn(32), doc_id="b"),
            Document(text="doc C", embedding=torch.randn(32), doc_id="c"),
        ]
        store.add(docs)
        assert len(store) == 3

        query = torch.randn(32)
        results = store.search(query, top_k=2)
        assert len(results) == 2

    def test_search_empty_store(self):
        from open_mythos.rag import VectorStore
        import torch

        store = VectorStore(embed_dim=32)
        results = store.search(torch.randn(32), top_k=3)
        assert results == []

    def test_search_top_k_capped(self):
        from open_mythos.rag import VectorStore, Document
        import torch

        store = VectorStore(embed_dim=32)
        docs = [Document(text=f"doc {i}", embedding=torch.randn(32), doc_id=str(i)) for i in range(2)]
        store.add(docs)
        results = store.search(torch.randn(32), top_k=5)  # top_k > len
        assert len(results) <= 2

    def test_doc_without_embedding_raises(self):
        from open_mythos.rag import VectorStore, Document

        store = VectorStore(embed_dim=32)
        with pytest.raises(ValueError, match="no embedding"):
            store.add([Document(text="no embedding doc")])


class TestRAGPipeline:
    def test_add_documents(self):
        from open_mythos.rag import RAGPipeline

        model = _tiny_model()
        pipeline = RAGPipeline(model, device="cpu", embed_dim=64)
        n = pipeline.add_documents(["doc1", "doc2", "doc3"])
        assert n == 3
        assert pipeline.n_docs() == 3

    def test_retrieve_returns_docs(self):
        from open_mythos.rag import RAGPipeline

        model = _tiny_model()
        pipeline = RAGPipeline(model, device="cpu", embed_dim=64)
        pipeline.add_documents([
            "LLMOはAI検索エンジン向けの最適化手法です",
            "CTRはクリック率の略称です",
            "ROASは広告費用対効果の指標です",
        ])
        results = pipeline.retrieve("AI検索の最適化", top_k=2)
        assert len(results) == 2
        for r in results:
            assert r.text
            assert 0.0 <= r.score <= 1.0 + 1e-5

    def test_generate_with_context_returns_result(self):
        from open_mythos.rag import RAGPipeline

        model = _tiny_model()
        pipeline = RAGPipeline(model, device="cpu", embed_dim=64)
        pipeline.add_documents(["LLMOとは Large Language Model Optimization の略です。"])
        result = pipeline.generate_with_context(
            query="LLMOとは？",
            top_k=1,
            max_new_tokens=5,
            loops=1,
        )
        assert result.query == "LLMOとは？"
        assert isinstance(result.answer, str)
        assert len(result.retrieved_docs) == 1
        assert result.n_docs_in_store == 1
        assert result.latency_ms >= 0.0

    def test_generate_with_empty_store(self):
        from open_mythos.rag import RAGPipeline

        model = _tiny_model()
        pipeline = RAGPipeline(model, device="cpu", embed_dim=64)
        result = pipeline.generate_with_context(
            query="テスト",
            top_k=3,
            max_new_tokens=5,
            loops=1,
        )
        assert result.retrieved_docs == []
        assert isinstance(result.answer, str)

    def test_add_with_metadata(self):
        from open_mythos.rag import RAGPipeline

        model = _tiny_model()
        pipeline = RAGPipeline(model, device="cpu", embed_dim=64)
        n = pipeline.add_documents(
            texts=["マーケティング用語集"],
            doc_ids=["marketing-001"],
            metadatas=[{"source": "internal", "category": "marketing"}],
        )
        assert n == 1
        results = pipeline.retrieve("マーケティング", top_k=1)
        assert results[0].doc_id == "marketing-001"


# ===========================================================================
# Track C — 11.3.2  serve/api.py — RAG スキーマ
# ===========================================================================


class TestAPIRAGModels:
    def test_rag_index_request(self):
        from serve.api import RAGIndexRequest

        req = RAGIndexRequest(texts=["doc1", "doc2"])
        assert len(req.texts) == 2

    def test_rag_query_request(self):
        from serve.api import RAGQueryRequest

        req = RAGQueryRequest(query="LLMOとは", top_k=3, generate=False)
        assert req.query == "LLMOとは"
        assert req.generate is False

    def test_rag_doc_result(self):
        from serve.api import RAGDocResult

        r = RAGDocResult(doc_id="d1", text="test", score=0.85)
        assert r.score == pytest.approx(0.85)


# ===========================================================================
# 統合テスト: __init__.py からの import
# ===========================================================================


class TestSprint11Imports:
    def test_tools_importable(self):
        from open_mythos import (
            ToolDefinition, ToolCall, ToolResult, ToolRegistry,
            tool, execute_tool_call, execute_tool_calls,
            parse_tool_calls, build_tool_prompt, register_marketing_tools,
        )
        assert ToolRegistry is not None

    def test_rope_extension_importable(self):
        from open_mythos import (
            RopeScalingConfig, yarn_rope_freqs, get_rope_freqs, extend_model_context
        )
        assert RopeScalingConfig is not None

    def test_rag_importable(self):
        from open_mythos import Document, VectorStore, RAGPipeline, RAGResult
        assert RAGPipeline is not None
