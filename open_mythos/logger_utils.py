"""
OpenMythos training logger — unified interface for WandB / MLflow / TensorBoard / none.

Usage in training scripts::

    from open_mythos.logger_utils import TrainLogger
    log = TrainLogger(backend="wandb", run_name="3b-run-001", config=cfg_dict)
    log.log({"loss": 1.23, "lr": 3e-4}, step=100)
    log.finish()

Backends
--------
- ``"wandb"``       -- Weights & Biases (requires ``pip install wandb``)
- ``"mlflow"``      -- MLflow (requires ``pip install mlflow``)
- ``"tensorboard"`` -- TensorBoardX / torch.utils.tensorboard
- ``"none"``        -- no-op; all calls are silent no-ops

Any missing optional dependency degrades gracefully to ``"none"`` with a
warning rather than crashing the training run.
"""

from __future__ import annotations

import warnings
from typing import Any


class TrainLogger:
    """Unified training logger with pluggable backend.

    Args:
        backend  -- one of ``"wandb"``, ``"mlflow"``, ``"tensorboard"``, ``"none"``
        run_name -- human-readable run identifier
        project  -- project / experiment name (WandB / MLflow)
        config   -- dict of hyperparameters to record at run start
        log_dir  -- directory for TensorBoard event files (ignored by other backends)
    """

    def __init__(
        self,
        backend: str = "none",
        run_name: str = "mythos-run",
        project: str = "open-mythos",
        config: dict | None = None,
        log_dir: str = "runs",
    ) -> None:
        self.backend = backend.lower()
        self._run = None
        self._writer = None

        if self.backend == "wandb":
            self._init_wandb(run_name, project, config or {})
        elif self.backend == "mlflow":
            self._init_mlflow(run_name, project, config or {})
        elif self.backend == "tensorboard":
            self._init_tensorboard(log_dir, run_name)
        elif self.backend == "none":
            pass
        else:
            warnings.warn(
                f"Unknown logger backend '{backend}'; falling back to 'none'.",
                stacklevel=2,
            )
            self.backend = "none"

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------

    def _init_wandb(self, run_name: str, project: str, config: dict) -> None:
        try:
            import wandb
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                resume="allow",
            )
        except ImportError:
            warnings.warn(
                "wandb not installed (pip install wandb). Falling back to 'none'.",
                stacklevel=3,
            )
            self.backend = "none"
        except Exception as exc:
            warnings.warn(
                f"wandb.init() failed: {exc}. Falling back to 'none'.",
                stacklevel=3,
            )
            self.backend = "none"

    def _init_mlflow(self, run_name: str, experiment: str, config: dict) -> None:
        try:
            import mlflow
            mlflow.set_experiment(experiment)
            self._run = mlflow.start_run(run_name=run_name)
            if config:
                mlflow.log_params(config)
        except ImportError:
            warnings.warn(
                "mlflow not installed (pip install mlflow). Falling back to 'none'.",
                stacklevel=3,
            )
            self.backend = "none"
        except Exception as exc:
            warnings.warn(
                f"mlflow.start_run() failed: {exc}. Falling back to 'none'.",
                stacklevel=3,
            )
            self.backend = "none"

    def _init_tensorboard(self, log_dir: str, run_name: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")
        except ImportError:
            try:
                from tensorboardX import SummaryWriter  # type: ignore[no-redef]
                self._writer = SummaryWriter(log_dir=f"{log_dir}/{run_name}")
            except ImportError:
                warnings.warn(
                    "tensorboard / tensorboardX not installed. Falling back to 'none'.",
                    stacklevel=3,
                )
                self.backend = "none"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dict of scalar metrics at the given step.

        Args:
            metrics -- ``{"loss": 1.23, "lr": 3e-4, ...}``
            step    -- global optimizer step; used as x-axis in plots
        """
        if self.backend == "wandb" and self._run is not None:
            import wandb
            wandb.log(metrics, step=step)

        elif self.backend == "mlflow" and self._run is not None:
            import mlflow
            mlflow.log_metrics(metrics, step=step)

        elif self.backend == "tensorboard" and self._writer is not None:
            for key, val in metrics.items():
                try:
                    self._writer.add_scalar(key, float(val), global_step=step)
                except (TypeError, ValueError):
                    pass  # skip non-scalar values silently

    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        """Upload a file as an artifact (WandB / MLflow only).

        Args:
            path          -- local file path to upload
            artifact_type -- artifact type label (e.g. ``"model"``, ``"dataset"``)
        """
        if self.backend == "wandb" and self._run is not None:
            import wandb
            artifact = wandb.Artifact(name=artifact_type, type=artifact_type)
            artifact.add_file(path)
            self._run.log_artifact(artifact)

        elif self.backend == "mlflow" and self._run is not None:
            import mlflow
            mlflow.log_artifact(path)

    def finish(self) -> None:
        """Finalise and close the logging run / writer."""
        if self.backend == "wandb" and self._run is not None:
            import wandb
            wandb.finish()

        elif self.backend == "mlflow" and self._run is not None:
            import mlflow
            mlflow.end_run()

        elif self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "TrainLogger":
        return self

    def __exit__(self, *_) -> None:
        self.finish()
