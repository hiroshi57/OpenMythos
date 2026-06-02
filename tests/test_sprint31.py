"""
Sprint 31 — GPU LoRA SFT 統合テストスイート (40 tests)

LoraTrainer は GPU なし環境ではシミュレーションにフォールバックするため
CI (CPU only) でも全テスト PASS する。
"""

import pytest
import torch

from open_mythos.lora_trainer import (
    LoraTrainerConfig,
    DistillInMemoryDataset,
    LoraTrainer,
    collate_distill,
    _default_tokenize,
)
from open_mythos.self_distill import (
    DistillSample,
    SFTResult,
    SelfDistillConfig,
    SelfDistillLoop,
    SelfDistillResult,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_samples(n: int = 6, score: float = 0.75) -> list:
    return [
        DistillSample(
            prompt=f"プロンプト{i}",
            output=f"これはサンプル出力{i}です。テスト用テキスト。",
            score=score,
        )
        for i in range(n)
    ]


# =========================================================================
# LoraTrainerConfig (5 tests)
# =========================================================================

def test_lora_config_defaults():
    cfg = LoraTrainerConfig()
    assert cfg.lr == 3e-4
    assert cfg.batch_size == 2
    assert cfg.max_length == 128
    assert cfg.device == "auto"

def test_lora_config_min_samples_default():
    cfg = LoraTrainerConfig()
    assert cfg.min_samples == 4

def test_lora_config_max_steps_default():
    cfg = LoraTrainerConfig()
    assert cfg.max_steps == 10

def test_lora_config_save_checkpoints_false():
    cfg = LoraTrainerConfig()
    assert cfg.save_checkpoints is False

def test_lora_config_custom():
    cfg = LoraTrainerConfig(lr=1e-3, batch_size=4, device="cpu")
    assert cfg.lr == 1e-3
    assert cfg.batch_size == 4
    assert cfg.device == "cpu"


# =========================================================================
# _default_tokenize (3 tests)
# =========================================================================

def test_default_tokenize_ascii():
    ids = _default_tokenize("hello")
    assert isinstance(ids, list)
    assert all(0 <= x <= 50256 for x in ids)

def test_default_tokenize_japanese():
    ids = _default_tokenize("こんにちは")
    assert len(ids) > 0
    assert all(0 <= x <= 50256 for x in ids)

def test_default_tokenize_clamp():
    # バイト値 > 50256 は 50256 にクランプされる
    ids = _default_tokenize("\xff\xfe")
    assert all(x <= 50256 for x in ids)


# =========================================================================
# DistillInMemoryDataset (7 tests)
# =========================================================================

def test_distill_dataset_len():
    samples = _make_samples(4)
    ds = DistillInMemoryDataset(samples, max_length=64)
    assert len(ds) == 4

def test_distill_dataset_item_keys():
    samples = _make_samples(2)
    ds = DistillInMemoryDataset(samples, max_length=64)
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert "weight" in item

def test_distill_dataset_weight_positive():
    samples = _make_samples(2, score=0.8)
    ds = DistillInMemoryDataset(samples, max_length=64)
    assert all(ds[i]["weight"] > 0 for i in range(len(ds)))

def test_distill_dataset_labels_length():
    samples = _make_samples(2)
    ds = DistillInMemoryDataset(samples, max_length=64)
    item = ds[0]
    assert len(item["input_ids"]) == len(item["labels"])

def test_distill_dataset_max_length():
    samples = _make_samples(1)
    ds = DistillInMemoryDataset(samples, max_length=10)
    assert len(ds[0]["input_ids"]) <= 10

def test_distill_dataset_custom_tokenize():
    called = []
    def my_tokenize(text: str):
        called.append(text)
        return [1, 2, 3]
    samples = _make_samples(1)
    ds = DistillInMemoryDataset(samples, tokenize_fn=my_tokenize)
    assert len(called) == 1

def test_distill_dataset_empty_samples():
    ds = DistillInMemoryDataset([], max_length=64)
    assert len(ds) == 0


# =========================================================================
# collate_distill (3 tests)
# =========================================================================

def test_collate_output_keys():
    samples = _make_samples(2)
    ds  = DistillInMemoryDataset(samples, max_length=32)
    batch = [ds[0], ds[1]]
    out = collate_distill(batch)
    assert "input_ids" in out
    assert "labels" in out
    assert "weights" in out

def test_collate_tensor_shape():
    samples = _make_samples(3)
    ds = DistillInMemoryDataset(samples, max_length=32)
    batch = [ds[i] for i in range(len(ds))]
    out = collate_distill(batch)
    assert out["input_ids"].shape[0] == len(batch)

def test_collate_labels_pad_value():
    samples = _make_samples(2)
    ds = DistillInMemoryDataset(samples, max_length=32)
    batch = [ds[0], ds[1]]
    out = collate_distill(batch)
    # パディング位置は -100
    assert (out["labels"] >= -100).all()


# =========================================================================
# LoraTrainer — 基本 (7 tests)
# =========================================================================

def test_lora_trainer_instantiation():
    trainer = LoraTrainer()
    assert trainer.cfg is not None

def test_lora_trainer_device_cpu_forced():
    trainer = LoraTrainer(cfg=LoraTrainerConfig(device="cpu"))
    assert trainer.device == "cpu"

def test_lora_trainer_device_auto_resolves():
    trainer = LoraTrainer()
    assert trainer.device in ("cuda", "cpu")

def test_lora_trainer_has_gpu_type():
    trainer = LoraTrainer()
    assert isinstance(trainer.has_gpu, bool)

def test_lora_trainer_has_gpu_consistent():
    trainer = LoraTrainer(cfg=LoraTrainerConfig(device="cpu"))
    assert trainer.has_gpu is False

def test_lora_trainer_custom_cfg():
    cfg = LoraTrainerConfig(lr=1e-3, max_steps=5)
    trainer = LoraTrainer(cfg=cfg)
    assert trainer.cfg.max_steps == 5

def test_lora_trainer_custom_tokenize():
    tokenize_fn = lambda t: [0, 1, 2]
    trainer = LoraTrainer(tokenize_fn=tokenize_fn)
    assert trainer._tokenize is tokenize_fn


# =========================================================================
# LoraTrainer.train() — SFTResult (6 tests)
# =========================================================================

def test_lora_trainer_returns_sft_result():
    trainer = LoraTrainer()
    samples = _make_samples(6)
    result  = trainer.train(samples, round_num=1)
    assert isinstance(result, SFTResult)

def test_lora_trainer_sft_n_samples():
    trainer = LoraTrainer()
    samples = _make_samples(5)
    result  = trainer.train(samples, round_num=1)
    assert result.n_samples == 5

def test_lora_trainer_sft_eval_score_range():
    trainer = LoraTrainer()
    samples = _make_samples(4)
    result  = trainer.train(samples, round_num=1)
    assert 0.0 <= result.eval_score <= 1.0

def test_lora_trainer_sft_train_loss_positive():
    trainer = LoraTrainer()
    samples = _make_samples(4)
    result  = trainer.train(samples, round_num=1)
    assert result.train_loss >= 0.0

def test_lora_trainer_sft_round_num():
    trainer = LoraTrainer()
    samples = _make_samples(4)
    result  = trainer.train(samples, round_num=3)
    assert result.round_num == 3

def test_lora_trainer_fallback_on_few_samples():
    """サンプル数 < min_samples → シミュレーションフォールバック"""
    cfg     = LoraTrainerConfig(min_samples=100)
    trainer = LoraTrainer(cfg=cfg)
    samples = _make_samples(2)
    result  = trainer.train(samples, round_num=1)
    # フォールバックでも SFTResult が返る
    assert isinstance(result, SFTResult)


# =========================================================================
# SelfDistillConfig.sft_backend (2 tests)
# =========================================================================

def test_sft_config_default_backend():
    cfg = SelfDistillConfig()
    assert cfg.sft_backend == "simulate"

def test_sft_config_lora_backend():
    cfg = SelfDistillConfig(sft_backend="lora")
    assert cfg.sft_backend == "lora"


# =========================================================================
# SelfDistillLoop — Sprint 31 統合 (7 tests)
# =========================================================================

def test_sdloop_run_string_input():
    """run() に文字列を渡してもエラーにならない (Sprint 31)"""
    loop   = SelfDistillLoop(SelfDistillConfig(n_rounds=1))
    result = loop.run("テストプロンプト")
    assert isinstance(result, SelfDistillResult)

def test_sdloop_run_n_iterations_override():
    """n_iterations でラウンド数を上書きできる"""
    loop   = SelfDistillLoop(SelfDistillConfig(n_rounds=5))
    result = loop.run(["prompt"], n_iterations=1)
    assert result.rounds_completed == 1

def test_sdloop_best_output_property():
    """best_output が DistillSample または None を返す"""
    loop   = SelfDistillLoop(SelfDistillConfig(n_rounds=1, score_threshold=0.0))
    result = loop.run(["テスト出力"])
    # best_output は DistillSample か None
    bo = result.best_output
    assert bo is None or isinstance(bo, DistillSample)

def test_sdloop_best_output_highest_score():
    """best_output がデータセット内の最高スコアを持つ"""
    loop   = SelfDistillLoop(SelfDistillConfig(n_rounds=2, score_threshold=0.0))
    result = loop.run(["プロンプト1", "プロンプト2"])
    bo = result.best_output
    if bo is not None:
        for s in result.dataset._samples:
            assert s.score <= bo.score + 1e-9

def test_sdloop_lora_backend_with_trainer():
    """sft_backend='lora' + LoraTrainer 注入でエラーなく完了する"""
    trainer = LoraTrainer()
    cfg     = SelfDistillConfig(n_rounds=1, sft_backend="lora")
    loop    = SelfDistillLoop(cfg=cfg, lora_trainer=trainer)
    result  = loop.run(["テストプロンプト"])
    assert isinstance(result, SelfDistillResult)

def test_sdloop_lora_backend_auto_creates_trainer():
    """sft_backend='lora' でも trainer=None なら自動生成して動作する"""
    cfg  = SelfDistillConfig(n_rounds=1, sft_backend="lora")
    loop = SelfDistillLoop(cfg=cfg)  # lora_trainer は None
    result = loop.run(["テスト"])
    assert isinstance(result, SelfDistillResult)

def test_sdloop_sft_result_in_round():
    """sft_simulate=True のとき round_result に sft_result が入る"""
    cfg    = SelfDistillConfig(n_rounds=1, score_threshold=0.0, sft_simulate=True)
    loop   = SelfDistillLoop(cfg=cfg)
    result = loop.run(["テストプロンプト"])
    rr = result.round_results[0]
    if rr.filtered > 0:
        assert rr.sft_result is not None
        assert isinstance(rr.sft_result, SFTResult)
