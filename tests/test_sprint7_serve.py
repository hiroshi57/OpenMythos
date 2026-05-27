"""
Sprint 7 serve/api.py テスト — /generate / /agent / /health 拡張。

FastAPI TestClient を使い、モデルは nano で実際に動かす。
transformers.AutoTokenizer のみモックして軽量化。
"""

from __future__ import annotations

import sys
import types
import uuid
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# transformers モック（AutoTokenizer のみ）
# ---------------------------------------------------------------------------

def _make_tokenizer_mock():
    """tokenizer(text, ...) → {"input_ids": tensor}、decode → str を返す。"""
    tok = MagicMock()
    # __call__: tokenize
    tok.side_effect = lambda text, **kw: {
        "input_ids": torch.zeros(1, max(1, len(str(text).split())), dtype=torch.long)
    }
    # decode: トークン列 → 固定文字列
    tok.decode = MagicMock(return_value="generated text")
    return tok


@pytest.fixture(scope="module", autouse=True)
def mock_transformers():
    fake_transformers = types.ModuleType("transformers")
    tok_instance = _make_tokenizer_mock()
    fake_transformers.AutoTokenizer = MagicMock()
    fake_transformers.AutoTokenizer.from_pretrained = MagicMock(return_value=tok_instance)
    # 既に import されていれば上書き
    orig = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers
    yield
    if orig is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = orig


# ---------------------------------------------------------------------------
# TASK_LOOPS / _TASK_SYSTEM_PROMPTS の直接検証
# （serve/api.py を直接 import して定数を取得）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_constants():
    """serve/api.py から TASK_LOOPS と _TASK_SYSTEM_PROMPTS を取得。"""
    import ast, pathlib
    src = pathlib.Path("serve/api.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    TARGET_NAMES = {"TASK_LOOPS", "_TASK_SYSTEM_PROMPTS", "DEFAULT_LOOPS"}
    ns: dict = {}

    for node in ast.walk(tree):
        # ast.Assign: x = {...}
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in TARGET_NAMES:
                    try:
                        ns[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
        # ast.AnnAssign: x: Type = {...}
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in TARGET_NAMES:
                if node.value is not None:
                    try:
                        ns[node.target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass

    ns.setdefault("DEFAULT_LOOPS", 4)
    return ns


class TestTaskTables:

    def test_task_loops_has_seo_content(self, api_constants):
        assert "seo_content" in api_constants["TASK_LOOPS"]

    def test_task_loops_has_llmo_optimize(self, api_constants):
        assert "llmo_optimize" in api_constants["TASK_LOOPS"]

    def test_task_loops_has_ad_copy(self, api_constants):
        assert "ad_copy" in api_constants["TASK_LOOPS"]

    def test_task_loops_has_persona_message(self, api_constants):
        assert "persona_message" in api_constants["TASK_LOOPS"]

    def test_task_loops_has_market_summary(self, api_constants):
        assert "market_summary" in api_constants["TASK_LOOPS"]

    def test_llmo_optimize_loops_ge_seo(self, api_constants):
        tl = api_constants["TASK_LOOPS"]
        assert tl["llmo_optimize"] >= tl["seo_content"]

    def test_ad_copy_loops_fast(self, api_constants):
        assert api_constants["TASK_LOOPS"]["ad_copy"] <= 4

    def test_system_prompts_has_seo_content(self, api_constants):
        assert "seo_content" in api_constants["_TASK_SYSTEM_PROMPTS"]

    def test_system_prompts_has_llmo_optimize(self, api_constants):
        assert "llmo_optimize" in api_constants["_TASK_SYSTEM_PROMPTS"]

    def test_system_prompts_has_ad_copy(self, api_constants):
        assert "ad_copy" in api_constants["_TASK_SYSTEM_PROMPTS"]

    def test_system_prompts_has_persona_message(self, api_constants):
        assert "persona_message" in api_constants["_TASK_SYSTEM_PROMPTS"]

    def test_system_prompts_has_market_summary(self, api_constants):
        assert "market_summary" in api_constants["_TASK_SYSTEM_PROMPTS"]

    def test_seo_prompt_mentions_eeat(self, api_constants):
        assert "E-E-A-T" in api_constants["_TASK_SYSTEM_PROMPTS"]["seo_content"]

    def test_llmo_prompt_mentions_ai(self, api_constants):
        p = api_constants["_TASK_SYSTEM_PROMPTS"]["llmo_optimize"]
        assert any(w in p for w in ["AI", "Claude", "ChatGPT", "LLM"])

    def test_ad_copy_prompt_mentions_ctr(self, api_constants):
        p = api_constants["_TASK_SYSTEM_PROMPTS"]["ad_copy"]
        assert any(w in p for w in ["クリック", "コンバージョン", "コピー"])

    def test_all_gen_tasks_have_prompts(self, api_constants):
        gen_tasks = {"seo_content", "llmo_optimize", "ad_copy", "persona_message", "market_summary", "general"}
        for t in gen_tasks:
            assert t in api_constants["_TASK_SYSTEM_PROMPTS"], f"{t} missing"


# ---------------------------------------------------------------------------
# TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from open_mythos.variants import mythos_nano
    from open_mythos.main import OpenMythos
    from open_mythos.agents import OpenMythosLLM, MythosAgent

    cfg = mythos_nano()
    model = OpenMythos(cfg).eval()

    import serve.api as api_module

    tok_mock = _make_tokenizer_mock()
    api_module.state.model = model
    api_module.state.tokenizer = tok_mock
    api_module.state.device = torch.device("cpu")
    api_module.state.n_params = sum(p.numel() for p in model.parameters())
    api_module.state.llm = OpenMythosLLM(
        model=model, device="cpu", max_new_tokens=4,
        temperature=1.0, top_k=10, top_p=0.9,
    )
    api_module.state.agents = {}

    with TestClient(api_module.app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_active_sessions(self, client):
        r = client.get("/health")
        assert "active_sessions" in r.json()

    def test_health_has_endpoints(self, client):
        r = client.get("/health")
        assert "endpoints" in r.json()

    def test_health_endpoints_include_generate(self, client):
        r = client.get("/health")
        assert any("generate" in k for k in r.json()["endpoints"])

    def test_health_endpoints_include_agent(self, client):
        r = client.get("/health")
        assert any("agent" in k for k in r.json()["endpoints"])

    def test_health_has_seo_task(self, client):
        r = client.get("/health")
        assert "seo_content" in r.json()["supported_tasks"]

    def test_health_has_llmo_task(self, client):
        r = client.get("/health")
        assert "llmo_optimize" in r.json()["supported_tasks"]


# ---------------------------------------------------------------------------
# /generate
# ---------------------------------------------------------------------------

class TestGenerateEndpoint:

    def test_generate_returns_200(self, client):
        r = client.post("/generate", json={"prompt": "SEO article", "task": "seo_content"})
        assert r.status_code == 200

    def test_generate_response_has_text(self, client):
        r = client.post("/generate", json={"prompt": "LLMO content", "task": "llmo_optimize"})
        assert "text" in r.json()
        assert isinstance(r.json()["text"], str)

    def test_generate_response_has_task(self, client):
        r = client.post("/generate", json={"prompt": "ad copy", "task": "ad_copy"})
        assert r.json()["task"] == "ad_copy"

    def test_generate_response_has_latency(self, client):
        r = client.post("/generate", json={"prompt": "test", "task": "general"})
        assert r.json()["latency_ms"] >= 0

    def test_generate_persona_message(self, client):
        r = client.post("/generate", json={"prompt": "persona message", "task": "persona_message"})
        assert r.status_code == 200

    def test_generate_market_summary(self, client):
        r = client.post("/generate", json={"prompt": "market summary", "task": "market_summary"})
        assert r.status_code == 200

    def test_generate_custom_system_prompt(self, client):
        r = client.post("/generate", json={
            "prompt": "test",
            "task": "general",
            "system_prompt": "You are a test assistant.",
        })
        assert r.status_code == 200

    def test_generate_prompt_len_positive(self, client):
        r = client.post("/generate", json={"prompt": "hello world", "task": "general"})
        assert r.json()["prompt_len"] >= 0


# ---------------------------------------------------------------------------
# /agent
# ---------------------------------------------------------------------------

class TestAgentEndpoint:

    def test_agent_creates_session(self, client):
        r = client.post("/agent", json={"task_input": "hello", "task": "general"})
        assert r.status_code == 200
        assert "session_id" in r.json()

    def test_agent_returns_response(self, client):
        r = client.post("/agent", json={"task_input": "SEO article structure", "task": "seo_content"})
        assert "response" in r.json()
        assert isinstance(r.json()["response"], str)

    def test_agent_reuses_session(self, client):
        r1 = client.post("/agent", json={"task_input": "first question", "task": "general"})
        sid = r1.json()["session_id"]
        r2 = client.post("/agent", json={"task_input": "follow up", "session_id": sid, "task": "general"})
        assert r2.json()["session_id"] == sid
        assert r2.json()["turn"] == 2

    def test_agent_turn_increments(self, client):
        r1 = client.post("/agent", json={"task_input": "q1", "task": "llmo_optimize"})
        sid = r1.json()["session_id"]
        r2 = client.post("/agent", json={"task_input": "q2", "session_id": sid, "task": "llmo_optimize"})
        assert r2.json()["turn"] > r1.json()["turn"]

    def test_agent_reset_session(self, client):
        r = client.post("/agent", json={"task_input": "test", "task": "ad_copy"})
        sid = r.json()["session_id"]
        r_del = client.delete(f"/agent/{sid}")
        assert r_del.status_code == 200
        assert r_del.json()["ok"] is True

    def test_agent_reset_nonexistent_returns_404(self, client):
        r = client.delete(f"/agent/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_agent_has_latency(self, client):
        r = client.post("/agent", json={"task_input": "latency test", "task": "general"})
        assert r.json()["latency_ms"] >= 0

    def test_agent_seo_task(self, client):
        r = client.post("/agent", json={"task_input": "keyword selection", "task": "seo_content"})
        assert r.status_code == 200

    def test_agent_market_summary_task(self, client):
        r = client.post("/agent", json={"task_input": "AI market overview", "task": "market_summary"})
        assert r.status_code == 200
