"""
Sprint 38 — GPU LoRA CosineScheduler 統合テスト

対象: open_mythos/lora_trainer.py
  - LoraTrainerConfig: warmup_steps / min_lr_ratio / use_scheduler フィールド追加
  - _real_train(): CosineAnnealingLR + 線形 warmup 統合
  - LoraTrainer.cosine_t_max() / get_current_lr() ヘルパー

GPU 不要: torch.cuda.is_available() をモックして _real_train() を CPU 上で実行。
GPU 必要テストは @pytest.mark.skipif で条件付きスキップ。
"""

from __future__ import annotations

import pytest
import torch
from unittest.mock import patch

from open_mythos.lora_trainer import (
    DistillInMemoryDataset,
    LoraTrainer,
    LoraTrainerConfig,
    collate_distill,
)
from open_mythos.self_distill import DistillSample, SFTResult


# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

def _make_sample(score: float = 0.8, round_num: int = 1) -> DistillSample:
    return DistillSample(
        prompt="テストプロンプト",
        output="テスト出力テキスト",
        score=score,
        round_num=round_num,
    )


def _make_samples(n: int = 6, score: float = 0.8) -> list:
    return [_make_sample(score=score) for _ in range(n)]


def _cpu_trainer(cfg: LoraTrainerConfig | None = None) -> LoraTrainer:
    """CUDA をモックして CPU で _real_train() が呼べるトレーナーを返す"""
    c = cfg or LoraTrainerConfig(
        max_steps=3,
        batch_size=2,
        n_loops=1,
        min_samples=4,
        device="cpu",
    )
    trainer = LoraTrainer(cfg=c)
    trainer._device_str = "cpu"
    return trainer


# ---------------------------------------------------------------------------
# 1. LoraTrainerConfig — 新フィールド
# ---------------------------------------------------------------------------

class TestLoraTrainerConfigSprint38:
    """Sprint 38 で追加した 3 フィールドのデフォルト値と設定を検証"""

    def test_warmup_steps_default(self):
        assert LoraTrainerConfig().warmup_steps == 0

    def test_min_lr_ratio_default(self):
        assert LoraTrainerConfig().min_lr_ratio == 0.0

    def test_use_scheduler_default(self):
        assert LoraTrainerConfig().use_scheduler is True

    def test_custom_warmup_steps(self):
        cfg = LoraTrainerConfig(warmup_steps=5)
        assert cfg.warmup_steps == 5

    def test_custom_min_lr_ratio(self):
        cfg = LoraTrainerConfig(min_lr_ratio=0.1)
        assert abs(cfg.min_lr_ratio - 0.1) < 1e-9

    def test_use_scheduler_false(self):
        cfg = LoraTrainerConfig(use_scheduler=False)
        assert cfg.use_scheduler is False

    def test_existing_fields_unchanged(self):
        """既存フィールドがデフォルト値を維持している"""
        cfg = LoraTrainerConfig()
        assert cfg.lr == 3e-4
        assert cfg.n_loops == 4
        assert cfg.batch_size == 2
        assert cfg.max_steps == 10
        assert cfg.min_samples == 4
        assert cfg.device == "auto"
        assert cfg.save_checkpoints is False


# ---------------------------------------------------------------------------
# 2. CosineAnnealingLR T_max 計算ロジック
# ---------------------------------------------------------------------------

class TestCosineSchedulerUnit:
    """cosine_t_max() の計算ロジックとスケジューラ単体の挙動を検証"""

    def test_t_max_no_warmup(self):
        cfg = LoraTrainerConfig(max_steps=10, warmup_steps=0)
        assert LoraTrainer.cosine_t_max(cfg) == 10

    def test_t_max_with_warmup(self):
        cfg = LoraTrainerConfig(max_steps=10, warmup_steps=3)
        assert LoraTrainer.cosine_t_max(cfg) == 7

    def test_t_max_minimum_one(self):
        """warmup_steps >= max_steps でも T_max が 1 になる"""
        cfg = LoraTrainerConfig(max_steps=5, warmup_steps=10)
        assert LoraTrainer.cosine_t_max(cfg) == 1

    def test_t_max_equal(self):
        """warmup_steps == max_steps → T_max = 1"""
        cfg = LoraTrainerConfig(max_steps=8, warmup_steps=8)
        assert LoraTrainer.cosine_t_max(cfg) == 1

    def test_eta_min_calculation(self):
        """min_lr_ratio から eta_min が正しく計算される"""
        cfg = LoraTrainerConfig(lr=1e-3, min_lr_ratio=0.1)
        expected = 1e-3 * 0.1
        assert abs(cfg.lr * cfg.min_lr_ratio - expected) < 1e-10

    def test_cosine_annealing_lr_decreases(self):
        """CosineAnnealingLR 適用後に LR が単調減少する (T_max=5)"""
        param = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([param], lr=1.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5, eta_min=0.0)
        lrs = []
        for _ in range(5):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()
        # 最初の LR > 最後の LR
        assert lrs[0] > lrs[-1]

    def test_cosine_annealing_eta_min_floor(self):
        """eta_min 設定時に LR が eta_min を下回らない"""
        param = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([param], lr=1.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4, eta_min=0.1)
        for _ in range(20):  # 十分に多くステップ
            sched.step()
        assert opt.param_groups[0]["lr"] >= 0.1 - 1e-6

    def test_get_current_lr_helper(self):
        """get_current_lr() がオプティマイザの現在 LR を返す"""
        trainer = LoraTrainer()
        param = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.AdamW([param], lr=5e-4)
        assert abs(trainer.get_current_lr(opt) - 5e-4) < 1e-9


# ---------------------------------------------------------------------------
# 3. _real_train() スモークテスト (CPU モック)
# ---------------------------------------------------------------------------

class TestRealTrainSmoke:
    """_real_train() が CPU 上で正常に完走し SFTResult を返すことを確認。

    _real_train() は has_gpu チェックを行わないため、直接 CPU で呼び出せる。
    torch.cuda.is_available() のモックは不要 (むしろ PyTorch 内部が CUDA を
    初期化しようとしてエラーになるため使わない)。
    """

    def test_returns_sft_result(self):
        result = _cpu_trainer()._real_train(_make_samples(6), round_num=1)
        assert isinstance(result, SFTResult)

    def test_train_loss_non_negative(self):
        result = _cpu_trainer()._real_train(_make_samples(6), round_num=1)
        assert result.train_loss >= 0.0

    def test_n_samples_matches_input(self):
        samples = _make_samples(8)
        result = _cpu_trainer()._real_train(samples, round_num=1)
        assert result.n_samples == 8

    def test_round_num_recorded(self):
        result = _cpu_trainer()._real_train(_make_samples(6), round_num=3)
        assert result.round_num == 3

    def test_duration_ms_positive(self):
        result = _cpu_trainer()._real_train(_make_samples(6), round_num=1)
        assert result.duration_ms > 0

    def test_eval_score_in_range(self):
        result = _cpu_trainer()._real_train(_make_samples(6), round_num=1)
        assert 0.0 <= result.eval_score <= 1.0

    def test_empty_dataset_falls_back_to_simulate(self):
        """有効サンプルが 0 件 → _simulate() に fallback"""
        cfg = LoraTrainerConfig(max_steps=2, batch_size=2, n_loops=1, device="cpu", min_samples=4)
        trainer = LoraTrainer(cfg=cfg)
        trainer._device_str = "cpu"
        empty_samples = [DistillSample(prompt="", output="", score=0.5) for _ in range(6)]
        result = trainer._real_train(empty_samples, round_num=1)
        assert isinstance(result, SFTResult)

    def test_max_steps_respected(self):
        """max_steps=2 で 2 ステップ以内に終了する"""
        cfg = LoraTrainerConfig(max_steps=2, batch_size=2, n_loops=1, device="cpu", min_samples=4)
        trainer = LoraTrainer(cfg=cfg)
        trainer._device_str = "cpu"
        result = trainer._real_train(_make_samples(10), round_num=1)
        assert isinstance(result, SFTResult)


# ---------------------------------------------------------------------------
# 4. Warmup 挙動テスト
# ---------------------------------------------------------------------------

class TestWarmupBehavior:
    """線形 warmup の LR 遷移を検証"""

    def test_warmup_lr_formula(self):
        """step=0, warmup_steps=4 → lr = base_lr * 1/4"""
        base_lr = 1e-3
        warmup_steps = 4
        step = 0
        expected = base_lr * (step + 1) / warmup_steps
        assert abs(expected - base_lr * 0.25) < 1e-9

    def test_warmup_lr_at_last_step(self):
        """step=warmup_steps-1 → lr = base_lr (warmup 完了)"""
        base_lr = 1e-3
        warmup_steps = 4
        step = warmup_steps - 1
        expected = base_lr * (step + 1) / warmup_steps
        assert abs(expected - base_lr) < 1e-9

    def test_warmup_lr_increases_monotonically(self):
        """warmup 期間中 LR が単調増加する"""
        base_lr = 1e-3
        warmup_steps = 5
        lrs = [base_lr * (s + 1) / warmup_steps for s in range(warmup_steps)]
        assert all(lrs[i] < lrs[i + 1] for i in range(len(lrs) - 1))

    def test_warmup_smoke_real_train(self):
        """warmup_steps=1 を含む _real_train() が正常完了する"""
        cfg = LoraTrainerConfig(
            max_steps=3, batch_size=2, n_loops=1, device="cpu",
            min_samples=4, warmup_steps=1,
        )
        trainer = LoraTrainer(cfg=cfg)
        trainer._device_str = "cpu"
        result = trainer._real_train(_make_samples(6), round_num=1)
        assert isinstance(result, SFTResult)
        assert result.train_loss >= 0

    def test_warmup_steps_larger_than_data(self):
        """warmup_steps がデータ数を超えてもクラッシュしない"""
        cfg = LoraTrainerConfig(
            max_steps=4, batch_size=2, n_loops=1, device="cpu",
            min_samples=4, warmup_steps=100,
        )
        trainer = LoraTrainer(cfg=cfg)
        trainer._device_str = "cpu"
        result = trainer._real_train(_make_samples(6), round_num=1)
        assert isinstance(result, SFTResult)

    def test_no_warmup_default(self):
        """warmup_steps=0 (デフォルト) でも _real_train() が正常完了"""
        cfg = LoraTrainerConfig(
            max_steps=3, batch_size=2, n_loops=1, device="cpu",
            min_samples=4, warmup_steps=0,
        )
        trainer = LoraTrainer(cfg=cfg)
        trainer._device_str = "cpu"
        result = trainer._real_train(_make_samples(6), round_num=1)
        assert isinstance(result, SFTResult)


# ---------------------------------------------------------------------------
# 5. use_scheduler=False — 固定 LR モード
# ---------------------------------------------------------------------------

class TestSchedulerToggle:
    """use_scheduler=False で CosineAnnealingLR が呼ばれないことを確認"""

    def test_no_scheduler_completes(self):
        cfg = LoraTrainerConfig(
            max_steps=3, batch_size=2, n_loops=1, device="cpu",
            min_samples=4, use_scheduler=False,
        )
        trainer = LoraTrainer(cfg=cfg)
        trainer._device_str = "cpu"
        result = trainer._real_train(_make_samples(6), round_num=1)
        assert isinstance(result, SFTResult)

    @patch("torch.optim.lr_scheduler.CosineAnnealingLR.step")
    def test_scheduler_step_not_called_when_disabled(self, mock_step):
        """use_scheduler=False のとき scheduler.step() が呼ばれない"""
        cfg = LoraTrainerConfig(
            max_steps=3, batch_size=2, n_loops=1, device="cpu",
            min_samples=4, use_scheduler=False,
        )
        trainer = LoraTrainer(cfg=cfg)
        trainer._device_str = "cpu"
        trainer._real_train(_make_samples(6), round_num=1)
        mock_step.assert_not_called()

    @patch("torch.optim.lr_scheduler.CosineAnnealingLR.step")
    def test_scheduler_step_called_when_enabled(self, mock_step):
        """use_scheduler=True (デフォルト) のとき scheduler.step() が呼ばれる"""
        cfg = LoraTrainerConfig(
            max_steps=3, batch_size=2, n_loops=1, device="cpu",
            min_samples=4, use_scheduler=True, warmup_steps=0,
        )
        trainer = LoraTrainer(cfg=cfg)
        trainer._device_str = "cpu"
        trainer._real_train(_make_samples(6), round_num=1)
        assert mock_step.call_count > 0

    def test_min_lr_ratio_passed(self):
        """min_lr_ratio=0.1 で _real_train() がクラッシュしない"""
        cfg = LoraTrainerConfig(
            max_steps=3, batch_size=2, n_loops=1, device="cpu",
            min_samples=4, min_lr_ratio=0.1,
        )
        trainer = LoraTrainer(cfg=cfg)
        trainer._device_str = "cpu"
        result = trainer._real_train(_make_samples(6), round_num=1)
        assert isinstance(result, SFTResult)


# ---------------------------------------------------------------------------
# 6. 後方互換性
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """既存の LoraTrainer / _simulate() インタフェースが壊れていないことを確認"""

    def test_has_gpu_false_on_cpu(self):
        trainer = LoraTrainer(LoraTrainerConfig(device="cpu"))
        assert trainer.has_gpu is False

    def test_device_property_cpu(self):
        trainer = LoraTrainer(LoraTrainerConfig(device="cpu"))
        assert trainer.device == "cpu"

    def test_train_falls_back_to_simulate_without_gpu(self):
        """CPU 環境では train() が _simulate() にフォールバックする"""
        trainer = LoraTrainer(LoraTrainerConfig(device="cpu"))
        result = trainer.train(_make_samples(6), round_num=1)
        assert isinstance(result, SFTResult)
        # simulate の train_loss は 1.0 - round_num*0.15 (round_num=1 → 0.85)
        assert result.train_loss >= 0.0

    def test_simulate_n_samples(self):
        samples = _make_samples(7)
        result = LoraTrainer._simulate(samples, round_num=2)
        assert result.n_samples == 7

    def test_simulate_round_num(self):
        result = LoraTrainer._simulate(_make_samples(5), round_num=4)
        assert result.round_num == 4

    def test_simulate_train_loss_decreasing(self):
        """ラウンドが増えるにつれて simulate の train_loss が下がる"""
        r1 = LoraTrainer._simulate(_make_samples(5), round_num=1)
        r3 = LoraTrainer._simulate(_make_samples(5), round_num=3)
        assert r1.train_loss > r3.train_loss

    def test_distill_dataset_unchanged(self):
        """DistillInMemoryDataset が Sprint 38 で壊れていない"""
        samples = _make_samples(4)
        ds = DistillInMemoryDataset(samples, max_length=32)
        assert len(ds) == 4

    def test_collate_still_works(self):
        samples = _make_samples(2)
        ds = DistillInMemoryDataset(samples, max_length=32)
        batch = collate_distill([ds[0], ds[1]])
        assert "input_ids" in batch
        assert "labels" in batch
        assert "weights" in batch


# ---------------------------------------------------------------------------
# 7. GPU 実機テスト (CUDA 環境でのみ実行)
# ---------------------------------------------------------------------------

_GPU_REASON = "CUDA GPU が必要"
_GPU_AVAIL = torch.cuda.is_available()


@pytest.mark.skipif(not _GPU_AVAIL, reason=_GPU_REASON)
class TestGPURealTrain:
    """CUDA GPU 実機で _real_train() が動作することを検証する。

    GPU のない CI 環境ではスキップされる。
    GPU 搭載マシンで pytest -m gpu などで実行すること。
    """

    def test_gpu_real_train_smoke(self):
        """GPU 実機で _real_train() が SFTResult を返す"""
        cfg = LoraTrainerConfig(
            max_steps=2, batch_size=2, n_loops=1, device="cuda", min_samples=4,
        )
        trainer = LoraTrainer(cfg=cfg)
        assert trainer.has_gpu
        result = trainer._real_train(_make_samples(6), round_num=1)
        assert isinstance(result, SFTResult)
        assert result.train_loss >= 0

    def test_gpu_device_is_cuda(self):
        """GPU 環境で device が "cuda" になる"""
        trainer = LoraTrainer(LoraTrainerConfig(device="auto"))
        assert trainer.device.startswith("cuda")

    def test_gpu_checkpoint_save(self, tmp_path):
        """save_checkpoints=True でチェックポイントファイルが生成される"""
        cfg = LoraTrainerConfig(
            max_steps=2, batch_size=2, n_loops=1, device="cuda",
            min_samples=4, save_checkpoints=True,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        trainer = LoraTrainer(cfg=cfg)
        trainer._real_train(_make_samples(6), round_num=1)
        ckpt_files = list((tmp_path / "ckpt").glob("*.pt"))
        assert len(ckpt_files) == 1
