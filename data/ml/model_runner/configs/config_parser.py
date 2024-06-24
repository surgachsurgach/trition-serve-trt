from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, Optional, Union

import gin
from loguru import logger

from data.ml.model_runner.utils import sagemaker_utils
from data.ml.utils import file_utils


@dataclasses.dataclass
class BaseConfig:
    accelerator: Optional[str] = dataclasses.field(default="cpu")
    batch_size: Optional[int] = dataclasses.field(default=32)
    devices: Optional[Union[int, str]] = dataclasses.field(default="auto")
    num_workers: int = dataclasses.field(default=0)
    pin_memory: bool = dataclasses.field(default=False)
    strategy: Optional[str] = dataclasses.field(default="ddp")

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        if config:
            return cls(**config)

        return None


@gin.configurable
@dataclasses.dataclass
class CheckpointConfig:
    monitor: str = dataclasses.field(default="loss")
    dirpath: str | None = dataclasses.field(default=None)
    filename: str = dataclasses.field(default="{epoch:03d}")
    mode: str = dataclasses.field(default="min")
    every_n_epochs: int | None = dataclasses.field(default=None)
    save_last: bool = dataclasses.field(default=True)

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None):
        if config:
            return cls(**config)

        return None


@gin.configurable
@dataclasses.dataclass
class EarlyStoppingConfig:
    monitor: str = dataclasses.field(default="loss")
    min_delta: float = dataclasses.field(default=0.0)
    mode: str = dataclasses.field(default="min")
    patience: int = dataclasses.field(default=3)
    verbose: bool = dataclasses.field(default=True)

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None):
        if config:
            return cls(**config)
        return None


@gin.configurable
@dataclasses.dataclass
class TrainConfig(BaseConfig):
    checkpoints: list[CheckpointConfig] | None = dataclasses.field(default=None)
    early_stopping: Optional[EarlyStoppingConfig] = dataclasses.field(default=None)
    log_dir: Optional[str] = dataclasses.field(default=None)
    max_epoch: int = dataclasses.field(default=50)
    min_epoch: int = dataclasses.field(default=1)
    log_every_n_steps: Optional[int] = dataclasses.field(default=None)

    def __post_init__(self):
        # TODO(hyesung): Remove this when we have a better way to handle this.
        if self.log_dir is None and sagemaker_utils.is_in_sagemaker_env():
            logger.info("Override log_dir with sagemaker tensorboard.")
            self.log_dir = sagemaker_utils.get_sagemaker_tensorboard_path()

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> TrainConfig:
        checkpoint_list = config.pop("checkpoints", None)
        early_stopping_dict = config.pop("early_stopping", None)

        if checkpoint_list:
            checkpoints = []
            for checkpoint_dict in checkpoint_list:
                ckpt = CheckpointConfig.from_dict(checkpoint_dict)
                if ckpt:
                    checkpoints.append(ckpt)
        else:
            checkpoints = None

        return cls(
            checkpoints=checkpoints,
            early_stopping=EarlyStoppingConfig.from_dict(early_stopping_dict),
            **config,
        )


@gin.configurable
@dataclasses.dataclass
class PredictConfig(BaseConfig):
    checkpoint_dir: Optional[str] = dataclasses.field(default=None)
    output_path: Optional[str] = dataclasses.field(default=None)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        if config:
            return cls(**config)
        return None

    def get_best_checkpoint_path(self) -> str:
        assert self.checkpoint_dir
        if has_best_model(self.checkpoint_dir):
            best_ckpt_path = best_checkpoint_path(self.checkpoint_dir)
            logger.info(f"Restoring the best checkpoint from {best_ckpt_path}")
            return best_ckpt_path

        fs = file_utils.get_filesystem(self.checkpoint_dir)
        last_ckpt_path = f"{self.checkpoint_dir}/last.ckpt"
        if fs.exists(last_ckpt_path):
            return last_ckpt_path

        raise RuntimeError(f"No checkpoint is available in {self.checkpoint_dir}.")


@gin.configurable
@dataclasses.dataclass
class RecsysPredictConfig(PredictConfig):
    top_k: int = dataclasses.field(default=20)
    exclude_inputs: bool = dataclasses.field(default=True)
    database: str | None = dataclasses.field(default=None)
    table_name: str | None = dataclasses.field(default=None)
    partition: dict | None = dataclasses.field(default=None)

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        if config:
            return cls(**config)
        return None

    def get_best_checkpoint_path(self) -> str:
        assert self.checkpoint_dir

        if has_best_model(self.checkpoint_dir):  # pylint: disable=protected-access
            best_ckpt_path = best_checkpoint_path(self.checkpoint_dir)  # pylint: disable=protected-access
            logger.info(f"Restoring the best checkpoint from {best_ckpt_path}")
            return best_ckpt_path

        fs = file_utils.get_filesystem(self.checkpoint_dir)
        last_ckpt_path = f"{self.checkpoint_dir}/last.ckpt"
        if fs.exists(last_ckpt_path):
            return last_ckpt_path

        raise RuntimeError(f"No checkpoint is available in {self.checkpoint_dir}.")


def best_checkpoint_path(model_dir: str) -> str:
    best_checkpoint = os.path.join(model_dir, "best_model.ckpt")
    logger.info(f"best checkpoint path: {best_checkpoint}")
    return best_checkpoint


def has_best_model(model_dir: str) -> bool:
    ckpt_path = best_checkpoint_path(model_dir)
    fs = file_utils.get_filesystem(ckpt_path)
    is_exists = fs.exists(ckpt_path)
    logger.debug(f"best checkpoint exists: {is_exists}")
    return is_exists
