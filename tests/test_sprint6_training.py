"""
Sprint 6.2 — 訓練スクリプト高度化テスト

カバー範囲:
  6.2.1  TrainLogger — none / wandb-missing / mlflow-missing / tensorboard-missing
         全て graceful fallback → "none" で動作すること
  6.2.2  training script の argparse (_parse_args) — デフォルト値と上書き確認
"""

from __future__ import annotations

import sys
import warnings

import pytest

from open_mythos.logger_utils import TrainLogger


# ---------------------------------------------------------------------------
# TrainLogger — none backend
# ---------------------------------------------------------------------------

class TestTrainLoggerNone:
    def test_init_none_no_crash(self):
        log = TrainLogger(backend="none")
        assert log.backend == "none"

    def test_log_none_no_crash(self):
        log = TrainLogger(backend="none")
        log.log({"loss": 1.23, "lr": 3e-4}, step=1)

    def test_log_artifact_none_no_crash(self):
        log = TrainLogger(backend="none")
        log.log_artifact("/nonexistent/path.pt")  # must not raise

    def test_finish_none_no_crash(self):
        log = TrainLogger(backend="none")
        log.finish()

    def test_context_manager(self):
        with TrainLogger(backend="none") as log:
            log.log({"x": 1.0}, step=0)

    def test_unknown_backend_falls_back_to_none(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log = TrainLogger(backend="nonexistent_backend")
        assert log.backend == "none"
        assert any("nonexistent_backend" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# TrainLogger — wandb missing → graceful fallback
# ---------------------------------------------------------------------------

class TestTrainLoggerWandbMissing:
    def test_wandb_missing_falls_back(self, monkeypatch):
        """wandb がインストールされていない環境でも fallback する。"""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "wandb":
                raise ImportError("wandb not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log = TrainLogger(backend="wandb")
        assert log.backend == "none"
        assert any("wandb" in str(warning.message).lower() for warning in w)

    def test_log_after_wandb_fallback(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "wandb":
                raise ImportError
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            log = TrainLogger(backend="wandb")
        log.log({"loss": 0.5}, step=1)  # must not raise
        log.finish()


# ---------------------------------------------------------------------------
# TrainLogger — mlflow missing → graceful fallback
# ---------------------------------------------------------------------------

class TestTrainLoggerMlflowMissing:
    def test_mlflow_missing_falls_back(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("mlflow not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log = TrainLogger(backend="mlflow")
        assert log.backend == "none"

    def test_log_after_mlflow_fallback(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            log = TrainLogger(backend="mlflow")
        log.log({"loss": 0.8}, step=2)
        log.finish()


# ---------------------------------------------------------------------------
# TrainLogger — tensorboard missing → graceful fallback
# ---------------------------------------------------------------------------

class TestTrainLoggerTensorboardMissing:
    def test_tensorboard_missing_falls_back(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("torch.utils.tensorboard", "tensorboardX"):
                raise ImportError
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # torch.utils.tensorboard はサブモジュールとして import されるため
        # monkeypatch.setitem で sys.modules から除外する
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)  # type: ignore
        monkeypatch.setitem(sys.modules, "tensorboardX", None)  # type: ignore

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log = TrainLogger(backend="tensorboard")
        assert log.backend == "none"


# ---------------------------------------------------------------------------
# training script argparse
# ---------------------------------------------------------------------------

class TestTrainingArgparse:
    """_parse_args のデフォルト値と CLI オーバーライドを検証。"""

    @pytest.fixture(autouse=True)
    def _import_parse_args(self):
        """Training script の _parse_args を ast 経由で安全にインポート。"""
        import ast, importlib.util, pathlib, textwrap

        src_path = pathlib.Path(__file__).parents[1] / "training" / "3b_fine_web_edu.py"
        src = src_path.read_text(encoding="utf-8")
        tree = ast.parse(src)

        # _parse_args 関数だけを抽出してコンパイル
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_parse_args":
                fn_src = ast.get_source_segment(src, node)
                ns: dict = {"argparse": __import__("argparse")}
                exec(compile(textwrap.dedent(fn_src), "<_parse_args>", "exec"), ns)
                self.parse = ns["_parse_args"]
                return
        pytest.skip("_parse_args not found in training script")

    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train"])
        args = self.parse()
        assert args.seq_len == 2048
        assert args.micro_batch == 4
        assert abs(args.lr - 3e-4) < 1e-8
        assert args.logger == "none"
        assert args.ckpt_dir == "checkpoints"
        assert args.resume is None

    def test_override_logger(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train", "--logger", "wandb"])
        args = self.parse()
        assert args.logger == "wandb"

    def test_override_ckpt_dir(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train", "--ckpt-dir", "/tmp/ckpts"])
        args = self.parse()
        assert args.ckpt_dir == "/tmp/ckpts"

    def test_override_resume(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train", "--resume", "/tmp/step_0001000.pt"])
        args = self.parse()
        assert args.resume == "/tmp/step_0001000.pt"

    def test_override_lr_and_seq_len(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train", "--lr", "1e-3", "--seq-len", "1024"])
        args = self.parse()
        assert abs(args.lr - 1e-3) < 1e-9
        assert args.seq_len == 1024

    def test_no_grad_ckpt_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train", "--no-grad-ckpt"])
        args = self.parse()
        assert args.no_grad_ckpt is True

    def test_run_name_default_none(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train"])
        args = self.parse()
        assert args.run_name is None

    def test_run_name_override(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["train", "--run-name", "exp-001"])
        args = self.parse()
        assert args.run_name == "exp-001"
