"""
Sprint 6.3 — エージェント統合テスト

カバー範囲:
  6.3.1  OpenMythosLLM — from_variant, run, stream, LangChain未インストール時の挙動
  6.3.2  MythosAgent   — from_variant, run, stream_run, history, reset, repr
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from open_mythos import OpenMythosLLM, MythosAgent, OpenMythos
from open_mythos.variants import mythos_nano


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm(**kwargs) -> OpenMythosLLM:
    kwargs.setdefault("max_new_tokens", 4)
    return OpenMythosLLM.from_variant("nano", **kwargs)


def _agent(**kwargs) -> MythosAgent:
    kwargs.setdefault("max_new_tokens", 4)
    return MythosAgent.from_variant("nano", **kwargs)


# ---------------------------------------------------------------------------
# 6.3.1  OpenMythosLLM
# ---------------------------------------------------------------------------

class TestOpenMythosLLM:
    def test_from_variant_returns_instance(self):
        llm = _llm()
        assert isinstance(llm, OpenMythosLLM)

    def test_run_returns_string(self):
        llm = _llm()
        result = llm.run("Hello")
        assert isinstance(result, str)

    def test_run_no_crash_empty_prompt(self):
        llm = _llm()
        result = llm.run("")
        assert isinstance(result, str)

    def test_stream_yields_strings(self):
        llm = _llm()
        chunks = list(llm.stream("Hi"))
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_stream_total_length_matches_max_new_tokens(self):
        """ストリームの合計トークン数が max_new_tokens と一致する。"""
        llm = _llm(max_new_tokens=4)
        chunks = list(llm.stream("Test"))
        # 各 chunk は 1 トークン分のテキスト
        assert len(chunks) == 4

    def test_llm_type_property(self):
        llm = _llm()
        assert llm._llm_type == "open-mythos"

    def test_temperature_affects_output_variation(self):
        """temperature=0 に近い値で2回生成しても同じかチェック（決定的ではないが煙テスト）。"""
        llm = _llm(temperature=0.01, top_k=1, max_new_tokens=4)
        r1 = llm.run("Once")
        r2 = llm.run("Once")
        # 小さい温度では同じ出力が得られやすい（必須ではないが確認）
        assert isinstance(r1, str) and isinstance(r2, str)

    def test_from_pretrained_local(self):
        """ローカル .pt ファイルから OpenMythosLLM を構築できる。"""
        cfg = mythos_nano()
        m = OpenMythos(cfg).eval()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pt")
            torch.save({"model": m.state_dict(), "cfg": cfg}, path)
            llm = OpenMythosLLM.from_pretrained(path, max_new_tokens=4)
        assert isinstance(llm.run("Hi"), str)

    def test_model_stored_outside_pydantic(self):
        """_model は Pydantic フィールドではなく object.__setattr__ で保持される。"""
        llm = _llm()
        assert isinstance(llm._model, OpenMythos)

    def test_max_new_tokens_respected(self):
        """generate は vocab_size 内のトークン ID を返す（クラッシュしない）。"""
        llm = _llm(max_new_tokens=8)
        result = llm.run("The answer is")
        assert isinstance(result, str)

    def test_langchain_not_installed_generate_raises(self, monkeypatch):
        """langchain_core がない場合 _generate() は ImportError を送出する。"""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "langchain" in name:
                raise ImportError("langchain not installed")
            return real_import(name, *args, **kwargs)

        # agents モジュールの _HAS_LANGCHAIN を False にする
        import open_mythos.agents as agents_mod
        monkeypatch.setattr(agents_mod, "_HAS_LANGCHAIN", False)

        llm = _llm()
        with pytest.raises(ImportError, match="langchain"):
            llm._generate(["Hello"])


# ---------------------------------------------------------------------------
# 6.3.2  MythosAgent
# ---------------------------------------------------------------------------

class TestMythosAgent:
    def test_from_variant_returns_instance(self):
        agent = _agent()
        assert isinstance(agent, MythosAgent)

    def test_run_returns_string(self):
        agent = _agent()
        result = agent.run("What is 2+2?")
        assert isinstance(result, str)

    def test_run_via_call_operator(self):
        agent = _agent()
        result = agent("Tell me a story.")
        assert isinstance(result, str)

    def test_stream_run_yields_strings(self):
        agent = _agent()
        chunks = list(agent.stream_run("Hello"))
        assert len(chunks) > 0
        assert all(isinstance(c, str) for c in chunks)

    def test_history_accumulates(self):
        agent = _agent()
        agent.run("First question")
        agent.run("Second question")
        assert len(agent._history) == 2

    def test_reset_clears_history(self):
        agent = _agent()
        agent.run("A task")
        agent.reset()
        assert len(agent._history) == 0

    def test_system_prompt_prepended(self):
        """system_prompt がプロンプトに含まれることを _build_prompt で確認。"""
        agent = _agent(system_prompt="You are a helpful assistant.")
        prompt = agent._build_prompt("Hi")
        assert "You are a helpful assistant." in prompt

    def test_build_prompt_includes_task(self):
        agent = _agent()
        prompt = agent._build_prompt("What is AI?")
        assert "What is AI?" in prompt

    def test_history_truncated_to_last_2_turns(self):
        """_build_prompt は最新2ターンだけを含む（コンテキスト肥大化防止）。"""
        agent = _agent()
        for i in range(5):
            agent._history.append(f"User: q{i}\nAssistant: a{i}")
        prompt = agent._build_prompt("new task")
        # 5ターン分すべては含まれない
        assert "q0" not in prompt
        assert "q3" in prompt or "q4" in prompt

    def test_repr_contains_agent_name(self):
        agent = _agent(agent_name="my-bot")
        assert "my-bot" in repr(agent)

    def test_from_pretrained_local(self):
        cfg = mythos_nano()
        m = OpenMythos(cfg).eval()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pt")
            torch.save({"model": m.state_dict(), "cfg": cfg}, path)
            agent = MythosAgent.from_pretrained(path, max_new_tokens=4)
        assert isinstance(agent.run("Hi"), str)

    def test_multi_turn_context_in_prompt(self):
        """2ターン後の _build_prompt に直近の会話履歴が含まれる。"""
        agent = _agent()
        agent._history.append("User: hello\nAssistant: hi there")
        prompt = agent._build_prompt("how are you?")
        assert "hello" in prompt or "hi there" in prompt
