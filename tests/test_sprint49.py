"""
Sprint 49 — 訓練最適化統合 テスト

対象:
  - open_mythos/skills/training_optimization.py:
      LightningTrainerConfig / LightningTrainResult / LightningTrainer
      FSDPConfig / FSDPModelInfo / FSDPWrapper
      SimPOConfig / SimPOTrainResult / SimPOTrainer
      SAEConfig / SAETrainResult / SparseAutoencoder
  - serve/api.py:
      POST /v1/training/lightning/fit
      POST /v1/training/lightning/config
      POST /v1/training/fsdp/estimate
      POST /v1/training/simpo/compute-loss
      POST /v1/training/simpo/train-step
      POST /v1/training/sae/forward
      POST /v1/training/sae/estimate-config
"""
from __future__ import annotations

import sys
import math
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

from open_mythos.skills.training_optimization import (
    LightningTrainerConfig, LightningTrainResult, LightningTrainer,
    FSDPConfig, FSDPModelInfo, FSDPWrapper,
    SimPOConfig, SimPOTrainResult, SimPOTrainer,
    SAEConfig, SAETrainResult, SparseAutoencoder,
)


# ---------------------------------------------------------------------------
# Section A: LightningTrainer
# ---------------------------------------------------------------------------

class TestLightningTrainerConfig:
    def test_defaults(self):
        cfg = LightningTrainerConfig()
        assert cfg.max_epochs == 3
        assert cfg.accelerator == "auto"
        assert cfg.precision == "32-true"

    def test_custom(self):
        cfg = LightningTrainerConfig(max_epochs=10, accelerator="gpu", precision="bf16-mixed")
        assert cfg.max_epochs == 10
        assert cfg.precision == "bf16-mixed"


class TestLightningTrainer:
    def test_is_native_bool(self):
        tr = LightningTrainer(LightningTrainerConfig())
        assert isinstance(tr.is_native, bool)

    def test_fit_returns_result(self):
        tr = LightningTrainer(LightningTrainerConfig(max_epochs=2))
        result = tr.fit(model=None)
        assert isinstance(result, LightningTrainResult)

    def test_fit_loss_positive(self):
        tr = LightningTrainer(LightningTrainerConfig(max_epochs=3))
        result = tr.fit(model=None)
        assert result.final_loss > 0.0

    def test_fit_epochs_trained(self):
        tr = LightningTrainer(LightningTrainerConfig(max_epochs=5))
        result = tr.fit(model=None)
        assert result.epochs_trained == 5

    def test_fit_train_time_nonneg(self):
        tr = LightningTrainer(LightningTrainerConfig())
        result = tr.fit(model=None)
        assert result.train_time_s >= 0.0

    def test_build_config_dict(self):
        cfg = LightningTrainerConfig(max_epochs=7)
        tr = LightningTrainer(cfg)
        d = tr.build_config_dict()
        assert d["max_epochs"] == 7
        assert "accelerator" in d


class TestLightningTrainResult:
    def test_creation(self):
        r = LightningTrainResult(final_loss=0.5, epochs_trained=3, best_val_loss=0.4, train_time_s=10.0)
        assert r.final_loss == 0.5
        assert r.callbacks_used == []


# ---------------------------------------------------------------------------
# Section B: FSDPWrapper
# ---------------------------------------------------------------------------

class TestFSDPConfig:
    def test_defaults(self):
        cfg = FSDPConfig()
        assert cfg.sharding_strategy == "FULL_SHARD"
        assert cfg.mixed_precision is True
        assert cfg.cpu_offload is False

    def test_custom(self):
        cfg = FSDPConfig(sharding_strategy="NO_SHARD", cpu_offload=True)
        assert cfg.sharding_strategy == "NO_SHARD"
        assert cfg.cpu_offload is True


class TestFSDPWrapper:
    def test_is_native_bool(self):
        fw = FSDPWrapper(FSDPConfig())
        assert isinstance(fw.is_native, bool)

    def test_wrap_single_gpu_returns_model(self):
        fw = FSDPWrapper(FSDPConfig())
        mock_model = object()
        result = fw.wrap(mock_model, world_size=1)
        assert result is mock_model

    def test_estimate_memory_keys(self):
        fw = FSDPWrapper(FSDPConfig())
        info = fw.estimate_memory(total_params=1_000_000, world_size=4)
        assert isinstance(info, FSDPModelInfo)
        assert info.shard_count == 4
        assert info.total_params == 1_000_000

    def test_estimate_memory_sharding(self):
        fw = FSDPWrapper(FSDPConfig())
        info = fw.estimate_memory(total_params=4_000_000, world_size=4)
        assert info.local_params == 1_000_000

    def test_estimate_memory_mb_positive(self):
        fw = FSDPWrapper(FSDPConfig())
        info = fw.estimate_memory(total_params=1_000_000, world_size=1)
        assert info.memory_per_shard_mb > 0.0


class TestFSDPModelInfo:
    def test_creation(self):
        info = FSDPModelInfo(shard_count=8, local_params=125000, total_params=1000000, memory_per_shard_mb=0.25)
        assert info.shard_count == 8


# ---------------------------------------------------------------------------
# Section C: SimPOTrainer
# ---------------------------------------------------------------------------

class TestSimPOConfig:
    def test_defaults(self):
        cfg = SimPOConfig()
        assert cfg.beta == 2.0
        assert cfg.gamma == 0.5
        assert cfg.reference_free is True

    def test_custom(self):
        cfg = SimPOConfig(beta=1.0, gamma=0.3)
        assert cfg.beta == 1.0


class TestSimPOTrainer:
    def test_is_native_bool(self):
        tr = SimPOTrainer(SimPOConfig())
        assert isinstance(tr.is_native, bool)

    def test_compute_loss_positive(self):
        tr = SimPOTrainer(SimPOConfig())
        loss = tr.compute_loss([-0.5, -0.6], [-1.0, -1.2])
        assert loss >= 0.0

    def test_compute_loss_is_float(self):
        tr = SimPOTrainer(SimPOConfig())
        loss = tr.compute_loss([-0.3], [-0.8])
        assert isinstance(loss, float)

    def test_train_step_keys(self):
        tr = SimPOTrainer(SimPOConfig())
        result = tr.train_step([-0.4, -0.5], [-0.9, -1.0])
        assert "loss" in result
        assert "chosen_reward" in result
        assert "rejected_reward" in result
        assert "reward_margin" in result

    def test_train_step_reward_margin(self):
        tr = SimPOTrainer(SimPOConfig())
        result = tr.train_step([-0.2, -0.3], [-0.8, -0.9])
        # chosen が高いので margin > 0
        assert result["reward_margin"] > 0.0


class TestSimPOTrainResult:
    def test_creation(self):
        r = SimPOTrainResult(final_loss=0.3, chosen_reward=0.7, rejected_reward=0.2,
                              reward_margin=0.5, steps=100)
        assert r.steps == 100


# ---------------------------------------------------------------------------
# Section D: SparseAutoencoder
# ---------------------------------------------------------------------------

class TestSAEConfig:
    def test_defaults(self):
        cfg = SAEConfig()
        assert cfg.d_in == 512
        assert cfg.d_sae == 2048
        assert cfg.k == 32

    def test_custom(self):
        cfg = SAEConfig(d_in=256, d_sae=1024, k=16)
        assert cfg.d_in == 256


class TestSparseAutoencoder:
    def test_is_native_bool(self):
        sae = SparseAutoencoder(SAEConfig(d_in=16, d_sae=64, k=4))
        assert isinstance(sae.is_native, bool)

    def test_encode_output_shape(self):
        sae = SparseAutoencoder(SAEConfig(d_in=16, d_sae=64, k=4))
        x = torch.randn(8, 16)
        z = sae.encode(x)
        assert z.shape == (8, 64)

    def test_encode_sparsity(self):
        sae = SparseAutoencoder(SAEConfig(d_in=16, d_sae=64, k=4))
        x = torch.randn(8, 16)
        z = sae.encode(x)
        # TopK なので各行のゼロでない要素は k 以下
        nonzero_per_row = (z != 0).sum(dim=-1).float()
        assert (nonzero_per_row <= 4).all()

    def test_forward_has_recon_loss(self):
        sae = SparseAutoencoder(SAEConfig(d_in=16, d_sae=64, k=4))
        x = torch.randn(4, 16)
        result = sae.forward(x)
        assert "recon_loss" in result
        assert isinstance(result["recon_loss"], float)

    def test_forward_has_sparsity(self):
        sae = SparseAutoencoder(SAEConfig(d_in=16, d_sae=64, k=4))
        x = torch.randn(4, 16)
        result = sae.forward(x)
        assert "l0_sparsity" in result
        assert "l1_sparsity" in result

    def test_estimate_config(self):
        sae = SparseAutoencoder(SAEConfig(d_in=16, d_sae=64))
        cfg = sae.estimate_config(model_dim=512, expansion=4)
        assert cfg.d_in == 512
        assert cfg.d_sae == 2048


# ---------------------------------------------------------------------------
# Section E: API エンドポイント
# ---------------------------------------------------------------------------

class TestLightningFitEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/training/lightning/fit",
                        json={"max_epochs": 2},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_final_loss(self, client):
        r = client.post("/v1/training/lightning/fit",
                        json={"max_epochs": 3},
                        headers=_HDR)
        assert "final_loss" in r.json()
        assert r.json()["final_loss"] > 0.0

    def test_has_epochs_trained(self, client):
        r = client.post("/v1/training/lightning/fit",
                        json={"max_epochs": 4},
                        headers=_HDR)
        assert r.json()["epochs_trained"] == 4


class TestLightningConfigEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/training/lightning/config",
                        json={"max_epochs": 5, "accelerator": "cpu"},
                        headers=_HDR)
        assert r.status_code == 200

    def test_has_config(self, client):
        r = client.post("/v1/training/lightning/config",
                        json={"max_epochs": 3},
                        headers=_HDR)
        assert r.json()["max_epochs"] == 3


class TestFSDPEstimateEndpoint:
    def test_returns_200(self, client):
        r = client.post("/v1/training/fsdp/estimate",
                        json={"total_params": 1000000, "world_size": 4},
                        headers=_HDR)
        assert r.status_code == 200

    def test_shard_count(self, client):
        r = client.post("/v1/training/fsdp/estimate",
                        json={"total_params": 1000000, "world_size": 8},
                        headers=_HDR)
        assert r.json()["shard_count"] == 8

    def test_memory_positive(self, client):
        r = client.post("/v1/training/fsdp/estimate",
                        json={"total_params": 500000, "world_size": 2},
                        headers=_HDR)
        assert r.json()["memory_per_shard_mb"] > 0.0


class TestSimPOEndpoints:
    def test_compute_loss_200(self, client):
        r = client.post("/v1/training/simpo/compute-loss",
                        json={"chosen_logprobs": [-0.5, -0.6], "rejected_logprobs": [-1.0, -1.2]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_compute_loss_value(self, client):
        r = client.post("/v1/training/simpo/compute-loss",
                        json={"chosen_logprobs": [-0.3], "rejected_logprobs": [-0.9]},
                        headers=_HDR)
        assert "loss" in r.json()
        assert r.json()["loss"] >= 0.0

    def test_train_step_200(self, client):
        r = client.post("/v1/training/simpo/train-step",
                        json={"chosen_logprobs": [-0.4, -0.5], "rejected_logprobs": [-0.8, -0.9]},
                        headers=_HDR)
        assert r.status_code == 200

    def test_train_step_has_margin(self, client):
        r = client.post("/v1/training/simpo/train-step",
                        json={"chosen_logprobs": [-0.2], "rejected_logprobs": [-0.7]},
                        headers=_HDR)
        assert "reward_margin" in r.json()


class TestSAEEndpoints:
    def test_forward_200(self, client):
        r = client.post("/v1/training/sae/forward",
                        json={"activations": [[0.1] * 16], "d_in": 16, "d_sae": 64, "k": 4},
                        headers=_HDR)
        assert r.status_code == 200

    def test_forward_has_recon_loss(self, client):
        r = client.post("/v1/training/sae/forward",
                        json={"activations": [[0.5] * 16], "d_in": 16, "d_sae": 64, "k": 4},
                        headers=_HDR)
        assert "recon_loss" in r.json()

    def test_estimate_config_200(self, client):
        r = client.post("/v1/training/sae/estimate-config",
                        json={"model_dim": 512, "expansion": 4},
                        headers=_HDR)
        assert r.status_code == 200

    def test_estimate_config_values(self, client):
        r = client.post("/v1/training/sae/estimate-config",
                        json={"model_dim": 256, "expansion": 8},
                        headers=_HDR)
        assert r.json()["d_in"] == 256
        assert r.json()["d_sae"] == 2048
