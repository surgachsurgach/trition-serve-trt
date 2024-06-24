import math

import gin

from data.ml.model_runner.base import trainer_base
from data.ml.model_runner.datasets import parquet_dataset


@gin.configurable
class RecsysTrainer(trainer_base.TrainerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if isinstance(self._train_dataset, parquet_dataset.ParquetIterableDataSet):
            if self._config.devices > 1 and self._train_dataset.seed is None:
                raise ValueError("seed must be provided when using multiple devices")

    def _preprocess(self):
        super()._preprocess()

        self._model.total_training_steps = int(math.ceil(len(self._train_dataset) / self._config.batch_size) * self._config.max_epoch)
        self._model.train_steps_per_epoch = int(math.ceil(len(self._train_dataset) / self._config.batch_size))
        if self._dev_dataset:
            self._model.validation_steps_per_epoch = int(math.ceil(len(self._dev_dataset) / self._config.batch_size))

    def _postprocess(self):
        pass
