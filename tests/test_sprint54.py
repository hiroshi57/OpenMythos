"""
Sprint 54 — OpenAI Assistants API 互換レイヤー テスト

対象:
  - open_mythos/assistant.py:
      AssistantTool / AssistantObject / Thread / MessageContent / Message
      RunUsage / Run / AssistantStore / AssistantRunner
  - serve/api.py:
      POST   /v1/assistants
      GET    /v1/assistants
      GET    /v1/assistants/{assistant_id}
      DELETE /v1/assistants/{assistant_id}
      POST   /v1/threads
      GET    /v1/threads/{thread_id}
      DELETE /v1/threads/{thread_id}
      POST   /v1/threads/{thread_id}/messages
      GET    /v1/threads/{thread_id}/messages
      POST   /v1/threads/{thread_id}/runs
      GET    /v1/threads/{thread_id}/runs/{run_id}
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

from open_mythos.assistant import (
    AssistantTool, AssistantObject, Thread, MessageContent, Message,
    RunUsage, Run, AssistantStore, AssistantRunner,
)


# ---------------------------------------------------------------------------
# Section A: AssistantTool / AssistantObject
# ---------------------------------------------------------------------------

class TestAssistantTool:
    def test_creation(self):
        tool = AssistantTool(type="code_interpreter")
        assert tool.type == "code_interpreter"
        assert tool.function is None

    def test_function_tool(self):
        fn = {"name": "get_weather", "description": "Get weather", "parameters": {}}
        tool = AssistantTool(type="function", function=fn)
        assert tool.function["name"] == "get_weather"


class TestAssistantObject:
    def test_creation(self):
        asst = AssistantObject(id="asst_abc")
        assert asst.id == "asst_abc"
        assert asst.object == "assistant"
        assert asst.model == "openmythos"

    def test_to_dict_has_id(self):
        asst = AssistantObject(id="asst_xyz", name="MyBot")
        d = asst.to_dict()
        assert d["id"] == "asst_xyz"
        assert d["name"] == "MyBot"
        assert d["object"] == "assistant"

    def test_to_dict_has_tools(self):
        asst = AssistantObject(id="asst_t", tools=[AssistantTool(type="retrieval")])
        d = asst.to_dict()
        assert isinstance(d["tools"], list)
        assert d["tools"][0]["type"] == "retrieval"

    def test_created_at_positive(self):
        asst = AssistantObject(id="asst_1")
        assert asst.created_at > 0


# ---------------------------------------------------------------------------
# Section B: Thread / Message / MessageContent
# ---------------------------------------------------------------------------

class TestThread:
    def test_creation(self):
        t = Thread(id="thread_abc")
        assert t.id == "thread_abc"
        assert t.object == "thread"

    def test_to_dict(self):
        t = Thread(id="thread_xyz")
        d = t.to_dict()
        assert d["id"] == "thread_xyz"
        assert d["object"] == "thread"


class TestMessageContent:
    def test_creation(self):
        mc = MessageContent(type="text", text={"value": "hello", "annotations": []})
        assert mc.type == "text"
        assert mc.text["value"] == "hello"

    def test_to_dict(self):
        mc = MessageContent(type="text", text={"value": "test", "annotations": []})
        d = mc.to_dict()
        assert "text" in d
        assert d["type"] == "text"


class TestMessage:
    def test_creation(self):
        msg = Message(id="msg_abc", thread_id="thread_1", role="user")
        assert msg.role == "user"
        assert msg.object == "thread.message"

    def test_text_property(self):
        mc = MessageContent(type="text", text={"value": "Hello world", "annotations": []})
        msg = Message(id="msg_1", thread_id="t1", role="user", content=[mc])
        assert msg.text == "Hello world"

    def test_text_empty_no_content(self):
        msg = Message(id="msg_2", thread_id="t1", role="assistant")
        assert msg.text == ""

    def test_to_dict_has_role(self):
        msg = Message(id="msg_3", thread_id="t1", role="user")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["thread_id"] == "t1"


# ---------------------------------------------------------------------------
# Section C: RunUsage / Run
# ---------------------------------------------------------------------------

class TestRunUsage:
    def test_defaults(self):
        u = RunUsage()
        assert u.prompt_tokens == 0
        assert u.total_tokens == 0

    def test_to_dict(self):
        u = RunUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        d = u.to_dict()
        assert d["total_tokens"] == 30


class TestRun:
    def test_creation(self):
        run = Run(id="run_abc", thread_id="thread_1", assistant_id="asst_1")
        assert run.status == "queued"
        assert run.object == "thread.run"

    def test_to_dict_has_status(self):
        run = Run(id="run_1", thread_id="t1", assistant_id="a1")
        d = run.to_dict()
        assert d["status"] == "queued"
        assert d["thread_id"] == "t1"


# ---------------------------------------------------------------------------
# Section D: AssistantStore
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """テストごとに独立したストアを返す。"""
    return AssistantStore()


class TestAssistantStore:
    def test_create_assistant(self, store):
        asst = store.create_assistant(name="TestBot")
        assert asst.id.startswith("asst_")
        assert asst.name == "TestBot"

    def test_get_assistant(self, store):
        asst = store.create_assistant(name="Bot")
        found = store.get_assistant(asst.id)
        assert found is not None
        assert found.id == asst.id

    def test_get_assistant_missing(self, store):
        assert store.get_assistant("nonexistent") is None

    def test_list_assistants(self, store):
        store.create_assistant(name="A")
        store.create_assistant(name="B")
        items = store.list_assistants()
        assert len(items) >= 2

    def test_delete_assistant(self, store):
        asst = store.create_assistant()
        assert store.delete_assistant(asst.id) is True
        assert store.get_assistant(asst.id) is None

    def test_delete_assistant_missing(self, store):
        assert store.delete_assistant("not_exist") is False

    def test_create_thread(self, store):
        thread = store.create_thread()
        assert thread.id.startswith("thread_")

    def test_get_thread(self, store):
        thread = store.create_thread()
        found = store.get_thread(thread.id)
        assert found is not None

    def test_delete_thread(self, store):
        thread = store.create_thread()
        assert store.delete_thread(thread.id) is True
        assert store.get_thread(thread.id) is None

    def test_add_message(self, store):
        thread = store.create_thread()
        msg = store.add_message(thread.id, role="user", content="Hello")
        assert msg.role == "user"
        assert msg.text == "Hello"

    def test_list_messages(self, store):
        thread = store.create_thread()
        store.add_message(thread.id, role="user", content="Msg 1")
        store.add_message(thread.id, role="user", content="Msg 2")
        msgs = store.list_messages(thread.id)
        assert len(msgs) == 2

    def test_get_message(self, store):
        thread = store.create_thread()
        msg = store.add_message(thread.id, role="user", content="test")
        found = store.get_message(msg.id)
        assert found is not None
        assert found.text == "test"

    def test_create_run(self, store):
        asst = store.create_assistant(name="Bot")
        thread = store.create_thread()
        run = store.create_run(thread.id, asst.id)
        assert run.status == "queued"
        assert run.assistant_id == asst.id

    def test_get_run(self, store):
        asst = store.create_assistant()
        thread = store.create_thread()
        run = store.create_run(thread.id, asst.id)
        found = store.get_run(run.id)
        assert found is not None

    def test_update_run_status_completed(self, store):
        asst = store.create_assistant()
        thread = store.create_thread()
        run = store.create_run(thread.id, asst.id)
        updated = store.update_run_status(run.id, "completed")
        assert updated.status == "completed"
        assert updated.completed_at is not None

    def test_update_run_status_failed(self, store):
        asst = store.create_assistant()
        thread = store.create_thread()
        run = store.create_run(thread.id, asst.id)
        updated = store.update_run_status(run.id, "failed", error="test error")
        assert updated.status == "failed"
        assert updated.last_error is not None

    def test_list_runs(self, store):
        asst = store.create_assistant()
        thread = store.create_thread()
        store.create_run(thread.id, asst.id)
        store.create_run(thread.id, asst.id)
        runs = store.list_runs(thread.id)
        assert len(runs) == 2


# ---------------------------------------------------------------------------
# Section E: AssistantRunner
# ---------------------------------------------------------------------------

class TestAssistantRunner:
    def test_execute_completes(self, store):
        asst = store.create_assistant(instructions="You are helpful.")
        thread = store.create_thread()
        store.add_message(thread.id, role="user", content="Hello!")
        run = store.create_run(thread.id, asst.id)
        runner = AssistantRunner(store)
        result = runner.execute(run)
        assert result.status == "completed"

    def test_execute_adds_assistant_message(self, store):
        asst = store.create_assistant()
        thread = store.create_thread()
        store.add_message(thread.id, role="user", content="What is AI?")
        run = store.create_run(thread.id, asst.id)
        runner = AssistantRunner(store)
        runner.execute(run)
        msgs = store.list_messages(thread.id)
        roles = [m.role for m in msgs]
        assert "assistant" in roles

    def test_execute_records_usage(self, store):
        asst = store.create_assistant()
        thread = store.create_thread()
        store.add_message(thread.id, role="user", content="Hello")
        run = store.create_run(thread.id, asst.id)
        runner = AssistantRunner(store)
        result = runner.execute(run)
        assert result.usage.total_tokens > 0

    def test_execute_with_custom_llm(self, store):
        asst = store.create_assistant()
        thread = store.create_thread()
        store.add_message(thread.id, role="user", content="ping")
        run = store.create_run(thread.id, asst.id)
        runner = AssistantRunner(store, llm_fn=lambda prompt: "pong")
        runner.execute(run)
        msgs = store.list_messages(thread.id)
        asst_msgs = [m for m in msgs if m.role == "assistant"]
        assert asst_msgs[0].text == "pong"

    def test_completed_at_set(self, store):
        asst = store.create_assistant()
        thread = store.create_thread()
        store.add_message(thread.id, role="user", content="hi")
        run = store.create_run(thread.id, asst.id)
        runner = AssistantRunner(store)
        result = runner.execute(run)
        assert result.completed_at is not None


# ---------------------------------------------------------------------------
# Section F: API エンドポイント
# ---------------------------------------------------------------------------

class TestCreateAssistantEndpoint:
    def test_returns_201_or_200(self, client):
        r = client.post("/v1/assistants",
                        json={"name": "TestBot", "model": "openmythos",
                              "instructions": "You are helpful."},
                        headers=_HDR)
        assert r.status_code in (200, 201)

    def test_has_id(self, client):
        r = client.post("/v1/assistants",
                        json={"name": "Bot"},
                        headers=_HDR)
        data = r.json()
        assert "id" in data
        assert data["id"].startswith("asst_")

    def test_has_object_field(self, client):
        r = client.post("/v1/assistants", json={}, headers=_HDR)
        assert r.json()["object"] == "assistant"


class TestListAssistantsEndpoint:
    def test_returns_200(self, client):
        r = client.get("/v1/assistants", headers=_HDR)
        assert r.status_code == 200

    def test_has_data_list(self, client):
        r = client.get("/v1/assistants", headers=_HDR)
        assert "data" in r.json()
        assert isinstance(r.json()["data"], list)


class TestGetAssistantEndpoint:
    def test_returns_200(self, client):
        # まず作成
        create_r = client.post("/v1/assistants", json={"name": "GetBot"}, headers=_HDR)
        asst_id = create_r.json()["id"]
        r = client.get(f"/v1/assistants/{asst_id}", headers=_HDR)
        assert r.status_code == 200

    def test_returns_correct_id(self, client):
        create_r = client.post("/v1/assistants", json={"name": "IDBot"}, headers=_HDR)
        asst_id = create_r.json()["id"]
        r = client.get(f"/v1/assistants/{asst_id}", headers=_HDR)
        assert r.json()["id"] == asst_id

    def test_returns_404_for_unknown(self, client):
        r = client.get("/v1/assistants/asst_nonexistent_xyz", headers=_HDR)
        assert r.status_code == 404


class TestDeleteAssistantEndpoint:
    def test_returns_200(self, client):
        create_r = client.post("/v1/assistants", json={}, headers=_HDR)
        asst_id = create_r.json()["id"]
        r = client.delete(f"/v1/assistants/{asst_id}", headers=_HDR)
        assert r.status_code == 200

    def test_deleted_flag(self, client):
        create_r = client.post("/v1/assistants", json={}, headers=_HDR)
        asst_id = create_r.json()["id"]
        r = client.delete(f"/v1/assistants/{asst_id}", headers=_HDR)
        assert r.json()["deleted"] is True


class TestThreadEndpoints:
    def test_create_thread_200(self, client):
        r = client.post("/v1/threads", json={}, headers=_HDR)
        assert r.status_code in (200, 201)

    def test_create_thread_has_id(self, client):
        r = client.post("/v1/threads", json={}, headers=_HDR)
        assert r.json()["id"].startswith("thread_")

    def test_get_thread_200(self, client):
        create_r = client.post("/v1/threads", json={}, headers=_HDR)
        tid = create_r.json()["id"]
        r = client.get(f"/v1/threads/{tid}", headers=_HDR)
        assert r.status_code == 200

    def test_get_thread_correct_id(self, client):
        create_r = client.post("/v1/threads", json={}, headers=_HDR)
        tid = create_r.json()["id"]
        r = client.get(f"/v1/threads/{tid}", headers=_HDR)
        assert r.json()["id"] == tid

    def test_delete_thread_200(self, client):
        create_r = client.post("/v1/threads", json={}, headers=_HDR)
        tid = create_r.json()["id"]
        r = client.delete(f"/v1/threads/{tid}", headers=_HDR)
        assert r.status_code == 200
        assert r.json()["deleted"] is True


class TestMessageEndpoints:
    def test_add_message_200(self, client):
        tid = client.post("/v1/threads", json={}, headers=_HDR).json()["id"]
        r = client.post(f"/v1/threads/{tid}/messages",
                        json={"role": "user", "content": "Hello!"},
                        headers=_HDR)
        assert r.status_code in (200, 201)

    def test_add_message_has_id(self, client):
        tid = client.post("/v1/threads", json={}, headers=_HDR).json()["id"]
        r = client.post(f"/v1/threads/{tid}/messages",
                        json={"role": "user", "content": "Hi"},
                        headers=_HDR)
        assert r.json()["id"].startswith("msg_")

    def test_list_messages_200(self, client):
        tid = client.post("/v1/threads", json={}, headers=_HDR).json()["id"]
        client.post(f"/v1/threads/{tid}/messages",
                    json={"role": "user", "content": "test"}, headers=_HDR)
        r = client.get(f"/v1/threads/{tid}/messages", headers=_HDR)
        assert r.status_code == 200

    def test_list_messages_has_data(self, client):
        tid = client.post("/v1/threads", json={}, headers=_HDR).json()["id"]
        client.post(f"/v1/threads/{tid}/messages",
                    json={"role": "user", "content": "msg1"}, headers=_HDR)
        r = client.get(f"/v1/threads/{tid}/messages", headers=_HDR)
        assert "data" in r.json()
        assert len(r.json()["data"]) >= 1


class TestRunEndpoints:
    def _setup(self, client):
        """アシスタント + スレッド + メッセージを準備。"""
        asst_id = client.post("/v1/assistants",
                              json={"name": "RunBot", "instructions": "Be helpful."},
                              headers=_HDR).json()["id"]
        tid = client.post("/v1/threads", json={}, headers=_HDR).json()["id"]
        client.post(f"/v1/threads/{tid}/messages",
                    json={"role": "user", "content": "What is 2+2?"}, headers=_HDR)
        return asst_id, tid

    def test_create_run_200(self, client):
        asst_id, tid = self._setup(client)
        r = client.post(f"/v1/threads/{tid}/runs",
                        json={"assistant_id": asst_id},
                        headers=_HDR)
        assert r.status_code in (200, 201)

    def test_create_run_has_id(self, client):
        asst_id, tid = self._setup(client)
        r = client.post(f"/v1/threads/{tid}/runs",
                        json={"assistant_id": asst_id},
                        headers=_HDR)
        assert r.json()["id"].startswith("run_")

    def test_create_run_status_completed(self, client):
        asst_id, tid = self._setup(client)
        r = client.post(f"/v1/threads/{tid}/runs",
                        json={"assistant_id": asst_id},
                        headers=_HDR)
        assert r.json()["status"] == "completed"

    def test_get_run_200(self, client):
        asst_id, tid = self._setup(client)
        run_r = client.post(f"/v1/threads/{tid}/runs",
                            json={"assistant_id": asst_id},
                            headers=_HDR)
        run_id = run_r.json()["id"]
        r = client.get(f"/v1/threads/{tid}/runs/{run_id}", headers=_HDR)
        assert r.status_code == 200

    def test_get_run_correct_status(self, client):
        asst_id, tid = self._setup(client)
        run_r = client.post(f"/v1/threads/{tid}/runs",
                            json={"assistant_id": asst_id},
                            headers=_HDR)
        run_id = run_r.json()["id"]
        r = client.get(f"/v1/threads/{tid}/runs/{run_id}", headers=_HDR)
        assert r.json()["status"] == "completed"

    def test_run_adds_assistant_message(self, client):
        asst_id, tid = self._setup(client)
        client.post(f"/v1/threads/{tid}/runs",
                    json={"assistant_id": asst_id},
                    headers=_HDR)
        msgs = client.get(f"/v1/threads/{tid}/messages", headers=_HDR).json()["data"]
        roles = [m["role"] for m in msgs]
        assert "assistant" in roles
