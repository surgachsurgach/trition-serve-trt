import abc
import dataclasses
import datetime
import os

import fsspec
import gin
from lightning import pytorch as pl
from loguru import logger
import pytz
from torch.utils import data as td
import yaml

from data.ml.model_runner.base import loggers
from data.ml.model_runner.base import model_base
from data.ml.model_runner.configs import config_parser
from data.ml.model_runner.datasets import collate
from data.ml.model_runner.utils import sagemaker_utils
from data.ml.utils import file_utils


def _remove_path_if_exists(filesystem: fsspec.AbstractFileSystem, path: str, recursive: bool = False):
    try:
        if filesystem.exists(path):
            filesystem.rm(path, recursive=recursive)
    except (FileNotFoundError, OSError):
        # It is possible that the file is already removed by other process, especially in multi-gpu training.
        logger.warning("Failed to remove path: %s", path)
        logger.warning("device might be currently being used by some process.")
        pass


def _export_train_config(model: model_base.ModelBase, config: config_parser.TrainConfig):
    if config.checkpoints is None or len(config.checkpoints) < 1:
        logger.info("Skip exporting configuration yaml file because checkpoint configuration is not provided.")
        return

    if config.checkpoints[0].dirpath:
        train_config_path = os.path.join(config.checkpoints[0].dirpath, "config.yaml")

        fs = file_utils.get_filesystem(train_config_path)
        if fs.exists(train_config_path):
            _remove_path_if_exists(fs, train_config_path)
        else:
            fs.makedirs(os.path.dirname(train_config_path), exist_ok=True)

        with fs.open(train_config_path, "w") as ofile:
            yaml.dump_all([{"model": model.hparams}, config], ofile)
    else:
        logger.info("Skip exporting configuration yaml file since config.checkpoints[0].dirpath is None.")


def _get_callbacks(config: config_parser.TrainConfig) -> list[pl.Callback]:
    callbacks = []

    if config.checkpoints:
        for ckpt in config.checkpoints:
            callbacks.append(pl.callbacks.ModelCheckpoint(**dataclasses.asdict(ckpt)))

    if config.early_stopping:
        callbacks.append(pl.callbacks.EarlyStopping(**dataclasses.asdict(config.early_stopping)))

    return callbacks


def _sagemaker_checkpoint(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        if sagemaker_utils.is_in_sagemaker_env():
            # pylint: disable=protected-access
            self._trainer.save_checkpoint(sagemaker_utils.get_sagemaker_checkpoint())

    return wrapper


@gin.configurable
class TrainerBase(abc.ABC):
    def __init__(
        self,
        model: model_base.ModelBase,
        config: config_parser.TrainConfig,
        train_dataset: td.Dataset,
        dev_dataset: td.Dataset | None,
        train_collator: collate.Collator | None = None,
        dev_collator: collate.Collator | None = None,
    ):
        self._config = config
        self._model = model
        self._train_dataset = train_dataset
        self._dev_dataset = dev_dataset
        self._train_collator = train_collator
        self._dev_collator = dev_collator

        self._log_version = "_".join(
            [datetime.datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y_%m_%d_%H_%M_%S"), self._model.get_name()]
        )
        self._trainer = None

    def _clear_checkpoints_paths(self):
        if self._config.checkpoints:
            for ckpt in self._config.checkpoints:
                fs = file_utils.get_filesystem(ckpt.dirpath)
                _remove_path_if_exists(fs, ckpt.dirpath, recursive=True)

    @_sagemaker_checkpoint
    def train(self):
        with sagemaker_utils.init_run() as run:
            self._model.run = run
            run.log_parameters(
                {
                    "model": sagemaker_utils.guard_none(self._model.hparams),
                    "train_config": sagemaker_utils.dataclass_to_dict(self._config),
                },
            )
            _export_train_config(self._model, self._config)

            self._clear_checkpoints_paths()
            self._preprocess()

            train_data_loader = td.DataLoader(
                self._train_dataset,
                batch_size=self._config.batch_size,
                num_workers=self._config.num_workers,
                collate_fn=self._train_collator.collate if self._train_collator else None,
                pin_memory=self._config.pin_memory,
            )

            if self._dev_dataset:
                # TODO(swkang): change num_workers when dev dataset contains more than 1 file.
                dev_data_loader = td.DataLoader(
                    self._dev_dataset,
                    batch_size=self._config.batch_size,
                    num_workers=self._config.num_workers,
                    collate_fn=self._dev_collator.collate if self._dev_collator else None,
                    pin_memory=self._config.pin_memory,
                )
            else:
                dev_data_loader = None

            trainer = pl.Trainer(
                max_epochs=self._config.max_epoch,
                min_epochs=self._config.min_epoch,
                accelerator=self._config.accelerator,
                devices=self._config.devices if self._config.accelerator != "cpu" else "auto",
                logger=loggers.FastTensorboardLogger(self._config.log_dir, version=self._log_version),
                callbacks=[pl.callbacks.RichProgressBar(leave=True)] + _get_callbacks(self._config),
                enable_progress_bar=True,
                strategy=self._config.strategy if isinstance(self._config.devices, int) and self._config.devices > 1 else "auto",
                log_every_n_steps=self._config.log_every_n_steps,
            )
            self._trainer = trainer

            trainer.fit(self._model, train_data_loader, dev_data_loader)

            self._postprocess()

    @abc.abstractmethod
    def _preprocess(self):
        pass

    @abc.abstractmethod
    def _postprocess(self):
        pass
