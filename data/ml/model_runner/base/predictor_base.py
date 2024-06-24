import abc
from typing import Any

import gin
from lightning import pytorch as pl
import pandas as pd
from torch.utils import data as td

from data.ml.model_runner.base import model_base
from data.ml.model_runner.configs import config_parser
from data.ml.model_runner.datasets import collate


@gin.configurable
class PredictorBase(abc.ABC):
    def __init__(
        self,
        model: model_base.ModelBase,
        config: config_parser.PredictConfig,
        test_dataset: td.Dataset,
        test_collator: collate.Collator | None = None,
    ):
        self._config = config
        self._model = model
        self._dataset = test_dataset
        self._collator = test_collator

    def _load_model(self, ckpt_path: str) -> model_base.ModelBase:
        return self._model.load_from_checkpoint(ckpt_path)

    def predict(self):
        ckpt_path = self._config.get_best_checkpoint_path()
        model = self._load_model(ckpt_path)

        self._preprocess()

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            accelerator=self._config.accelerator,
            devices=self._config.devices if self._config.accelerator != "cpu" else "auto",
            callbacks=[pl.callbacks.RichProgressBar(leave=True)],
            enable_progress_bar=True,
            strategy=self._config.strategy if isinstance(self._config.devices, int) and self._config.devices > 1 else "auto",
        )

        data_loader = td.DataLoader(
            self._dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.num_workers,
            collate_fn=self._collator.collate if self._collator else None,
        )

        outputs = trainer.predict(model, data_loader)
        assert outputs, "`outputs` must not be empty."

        dframe = self._postprocess(outputs)
        self._save(dframe)

    @abc.abstractmethod
    def _preprocess(self):
        pass

    @abc.abstractmethod
    def _postprocess(self, outputs: list[Any] | list[list[Any]]) -> pd.DataFrame:
        """Save, print or do other processes with predicted outputs."""

    @abc.abstractmethod
    def _save(self, dframe: pd.DataFrame) -> None:
        """Save predictions to a file."""
