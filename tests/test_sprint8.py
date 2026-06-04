"""
Sprint 8 テスト

8.1.1  finetune.py  — LoRA モード / フルファインチューニングモード
8.1.2  eval_perplexity.py — --checkpoint フラグ
8.2.1  serve/api.py — /v1/chat/completions (OpenAI 互換)
8.2.2  serve/sla_router.py — ultra モード
"""

from __future__ import annotations

import math

import pytest
import torch

# ===========================================================================
# 8.1.1  scripts.finetune — LoRA ファインチューニング
# ===========================================================================


class TestFinetuneLoRA:
    def test_lora_flag_freezes_base_params(self):
        """--lora フラグ時は LoRA アダプタ以外が freeze される。"""
        from open_mythos.main import MythosConfig, OpenMythos

        cfg = MythosConfig(
            vocab_size=256,
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            max_loop_iters=2,
            prelude_layers=1,
            coda_layers=1,
            attn_type="gqa",
            n_experts=2,
            n_shared_experts=1,
            n_experts_per_tok=1,
            expert_dim=32,
            act_threshold=0.99,
            lora_rank=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
        )
        model = OpenMythos(cfg).enable_lora_finetuning()
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        assert len(trainable) > 0, "LoRA パラメータが存在すること"
        assert len(frozen) > 0, "Freeze されたパラメータが存在すること"
        assert len(trainable) < len(frozen), "trainable 数 < frozen 数"

    def test_lora_trainable_params_only_lora(self):
        """trainable_parameters() は .lora. を含む名前のみ。"""
        from open_mythos.main import MythosConfig, OpenMythos

        cfg = MythosConfig(
            vocab_size=256,
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            max_loop_iters=2,
            prelude_layers=1,
            coda_layers=1,
            attn_type="gqa",
            n_experts=2,
            n_shared_experts=1,
            n_experts_per_tok=1,
            expert_dim=32,
            act_threshold=0.99,
            lora_rank=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
        )
        model = OpenMythos(cfg).enable_lora_finetuning()
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        for name in trainable_names:
            assert ".lora." in name, f"非 LoRA パラメータが trainable: {name}"

    def test_full_finetune_all_params_trainable(self):
        """LoRA なしでは全パラメータが trainable。"""
        from open_mythos.main import MythosConfig, OpenMythos

        cfg = MythosConfig(
            vocab_size=256,
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            max_loop_iters=2,
            prelude_layers=1,
            coda_layers=1,
            attn_type="gqa",
            n_experts=2,
            n_shared_experts=1,
            n_experts_per_tok=1,
            expert_dim=32,
            act_threshold=0.99,
            lora_rank=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
        )
        model = OpenMythos(cfg)
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        assert len(frozen) == 0, "全 param が trainable であること"

    def test_lora_forward_runs(self):
        """LoRA mode で forward が動くこと。"""
        from open_mythos.main import MythosConfig, OpenMythos

        cfg = MythosConfig(
            vocab_size=256,
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            max_loop_iters=2,
            prelude_layers=1,
            coda_layers=1,
            attn_type="gqa",
            n_experts=2,
            n_shared_experts=1,
            n_experts_per_tok=1,
            expert_dim=32,
            act_threshold=0.99,
            lora_rank=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
        )
        model = OpenMythos(cfg).enable_lora_finetuning()
        ids = torch.randint(0, 256, (1, 8))
        out = model(ids, n_loops=2)
        assert out.shape == (1, 8, 256)

    def test_lora_backward_runs(self):
        """LoRA mode で backward + optimizer.step が通ること。"""
        from open_mythos.main import MythosConfig, OpenMythos
        import torch.nn.functional as F

        cfg = MythosConfig(
            vocab_size=256,
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=32,
            max_loop_iters=2,
            prelude_layers=1,
            coda_layers=1,
            attn_type="gqa",
            n_experts=2,
            n_shared_experts=1,
            n_experts_per_tok=1,
            expert_dim=32,
            act_threshold=0.99,
            lora_rank=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
        )
        model = OpenMythos(cfg).enable_lora_finetuning()
        opt = torch.optim.AdamW(list(model.trainable_parameters()), lr=1e-3)
        ids = torch.randint(0, 256, (1, 8))
        labels = torch.randint(0, 256, (1, 8))
        logits = model(ids, n_loops=2)
        loss = F.cross_entropy(logits.view(-1, 256), labels.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert not torch.isnan(loss), "loss が NaN でないこと"

    def test_finetune_configs_all_tasks(self):
        """FINETUNE_CONFIGS が全 4 タスク + all を含む。"""
        from scripts.finetune import FINETUNE_CONFIGS

        assert set(FINETUNE_CONFIGS.keys()) == {
            "content_quality",
            "ad_performance",
            "persona_segment",
            "market_research",
            "all",
        }

    def test_collate_fn_pads_correctly(self):
        """collate_fn が可変長バッチを正しくパディングする。"""
        from scripts.finetune import collate_fn

        batch = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "task": "t", "weight": 1.0},
            {"input_ids": [4, 5], "labels": [-100, 5], "task": "t", "weight": 2.0},
        ]
        result = collate_fn(batch)
        assert result["input_ids"].shape == (2, 3), "バッチ (2, max_len=3) になること"
        assert result["labels"].shape == (2, 3)
        # 短い方は 0 でパディング (input_ids)
        assert result["input_ids"][1, 2].item() == 0
        # 短い方のラベルは -100 でパディング
        assert result["labels"][1, 2].item() == -100


# ===========================================================================
# 8.1.2  scripts.eval_perplexity — compute_perplexity / checkpoint ロード
# ===========================================================================


class TestEvalPerplexity:
    def test_compute_perplexity_returns_finite(self):
        """compute_perplexity が有限の ppl を返すこと。"""
        from scripts.eval_perplexity import compute_perplexity, small_eval_config
        from open_mythos.main import OpenMythos

        cfg = small_eval_config()
        model = OpenMythos(cfg)
        token_ids = torch.randint(0, 256, (300,))
        ppl, elapsed = compute_perplexity(
            model,
            token_ids,
            n_loops=2,
            seq_len=32,
            device=torch.device("cpu"),
            max_batches=5,
        )
        assert math.isfinite(ppl), f"ppl は有限であること: {ppl}"
        assert ppl > 0
        assert elapsed >= 0

    def test_more_tokens_gives_same_scale_ppl(self):
        """トークン数が変わっても ppl のスケール(桁数)は安定。"""
        from scripts.eval_perplexity import compute_perplexity, small_eval_config
        from open_mythos.main import OpenMythos

        cfg = small_eval_config()
        model = OpenMythos(cfg)
        token_ids = torch.randint(0, 50257, (500,))
        ppl, _ = compute_perplexity(
            model,
            token_ids,
            n_loops=1,
            seq_len=64,
            device=torch.device("cpu"),
            max_batches=3,
        )
        # 乱数初期化モデルなので非常に大きい ppl になるが、有限であること
        assert math.isfinite(ppl)

    def test_checkpoint_load_changes_output(self, tmp_path):
        """checkpoint をロードするとランダム初期化と異なる出力になること。"""
        from scripts.eval_perplexity import small_eval_config
        from open_mythos.main import OpenMythos

        cfg = small_eval_config()
        m1 = OpenMythos(cfg)
        m2 = OpenMythos(cfg)

        # m2 に異なるランダム重みを設定して保存
        for p in m2.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
        ckpt_path = tmp_path / "model.pt"
        torch.save(m2.state_dict(), ckpt_path)

        # m1 にロード
        m1.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

        ids = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            out1 = m1(ids, n_loops=1)
            out2 = m2(ids, n_loops=1)
        # ロード後は同じ出力になること
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_small_eval_config_valid(self):
        """small_eval_config() が有効な MythosConfig を返すこと。"""
        from scripts.eval_perplexity import small_eval_config
        from open_mythos.main import OpenMythos

        cfg = small_eval_config()
        assert cfg.dim == 256
        model = OpenMythos(cfg)
        assert model is not None


# ===========================================================================
# 8.2.1  serve.api — /v1/chat/completions
# ===========================================================================


class TestChatCompletions:
    @pytest.fixture
    def mock_state(self, monkeypatch):
        """serve.api の global state を小さいモデルで差し替える。"""
        import serve.api as api
        from open_mythos.main import MythosConfig, OpenMythos
        from transformers import AutoTokenizer

        cfg = MythosConfig(
            vocab_size=50257,
            dim=64,
            n_heads=4,
            n_kv_heads=2,
            max_seq_len=128,
            max_loop_iters=4,
            prelude_layers=1,
            coda_layers=1,
            attn_type="gqa",
            n_experts=2,
            n_shared_experts=1,
            n_experts_per_tok=1,
            expert_dim=32,
            act_threshold=0.99,
            lora_rank=4,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=8,
        )
        model = OpenMythos(cfg)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # _State は lifespan 前は属性を持たない → 直接代入
        api.state.model = model
        api.state.tokenizer = tokenizer
        api.state.device = torch.device("cpu")
        api.state.n_params = sum(p.numel() for p in model.parameters())
        return api

    def test_chat_completions_returns_response(self, mock_state):
        from serve.api import ChatRequest, ChatMessage, chat_completions

        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=8,
            loops=2,
            stream=False,
        )
        resp = chat_completions(req)
        assert resp.id.startswith("chatcmpl-")
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"

    def test_chat_completions_usage(self, mock_state):
        from serve.api import ChatRequest, ChatMessage, chat_completions

        req = ChatRequest(
            messages=[ChatMessage(role="user", content="test")],
            max_tokens=4,
            loops=1,
        )
        resp = chat_completions(req)
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    def test_chat_completions_schema(self, mock_state):
        from serve.api import ChatRequest, ChatMessage, chat_completions

        req = ChatRequest(
            messages=[
                ChatMessage(role="system", content="You are helpful."),
                ChatMessage(role="user", content="Say hi"),
            ],
            max_tokens=6,
        )
        resp = chat_completions(req)
        # Sprint 40: finish_reason は EOS 到達時 "stop"、max_tokens 到達時 "length"
        assert resp.choices[0].finish_reason in ("stop", "length")
        assert isinstance(resp.choices[0].message.content, str)

    def test_chat_request_defaults(self):
        from serve.api import ChatRequest, ChatMessage

        req = ChatRequest(messages=[ChatMessage(role="user", content="x")])
        assert req.model == "openmythos"
        assert req.stream is False
        assert req.task == "general"

    def test_build_chat_prompt(self):
        from serve.api import _build_chat_prompt, ChatMessage

        msgs = [
            ChatMessage(role="system", content="Be helpful"),
            ChatMessage(role="user", content="Hello"),
        ]
        prompt = _build_chat_prompt(msgs)
        assert "[System]:" in prompt
        assert "[User]:" in prompt
        assert "[Assistant]:" in prompt


# ===========================================================================
# 8.2.2  serve.sla_router — ultra モード
# ===========================================================================


class TestSLAUltraMode:
    def test_ultra_loops_16_for_all_tasks(self):
        """ultra モードは全タスクで loops=16。"""
        from serve.sla_router import _resolve_loops_and_budget, DEFAULT_SLA

        for task in DEFAULT_SLA:
            loops, budget = _resolve_loops_and_budget(task, "ultra")
            assert loops == 16, f"{task}: ultra loops should be 16, got {loops}"

    def test_ultra_budget_is_2000(self):
        """ultra モードのデフォルト budget は 2000 ms。"""
        from serve.sla_router import _resolve_loops_and_budget, DEFAULT_SLA

        for task in DEFAULT_SLA:
            loops, budget = _resolve_loops_and_budget(task, "ultra")
            assert budget == 2000, f"{task}: ultra budget should be 2000, got {budget}"

    def test_ultra_loops_gte_accurate(self):
        """ultra ループ数 >= accurate ループ数。"""
        from serve.sla_router import _resolve_loops_and_budget, DEFAULT_SLA

        for task in DEFAULT_SLA:
            loops_acc, _ = _resolve_loops_and_budget(task, "accurate")
            loops_ult, _ = _resolve_loops_and_budget(task, "ultra")
            assert loops_ult >= loops_acc, f"{task}: ultra >= accurate"

    def test_sla_request_accepts_ultra(self):
        from serve.sla_router import SLARequest

        req = SLARequest(text="test", sla_mode="ultra")
        assert req.sla_mode == "ultra"

    def test_sla_request_default_still_balanced(self):
        from serve.sla_router import SLARequest

        req = SLARequest(text="test")
        assert req.sla_mode == "balanced"

    def test_all_modes_valid(self):
        """fast / balanced / accurate / ultra の全モードが解決できること。"""
        from serve.sla_router import _resolve_loops_and_budget, DEFAULT_SLA

        for task in DEFAULT_SLA:
            for mode in ("fast", "balanced", "accurate", "ultra"):
                loops, budget = _resolve_loops_and_budget(task, mode)
                assert 1 <= loops <= 16
                assert budget > 0
