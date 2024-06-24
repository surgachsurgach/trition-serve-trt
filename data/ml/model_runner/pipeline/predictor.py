import os
from typing import Any

import gin
from loguru import logger
import pandas as pd

from data.ml.model_runner.base import model_base
from data.ml.model_runner.base import predictor_base
from data.ml.model_runner.configs import config_parser
from data.ml.utils import file_utils
from data.ml.utils import metadata as meta


@gin.configurable
class RecsysPredictor(predictor_base.PredictorBase):
    def __init__(self, metadata: meta.Meta, **kwargs):
        super().__init__(**kwargs)
        self._metadata = metadata

    def _load_model(self, ckpt_path: str) -> model_base.ModelBase:
        if not isinstance(self._config, config_parser.RecsysPredictConfig):
            assert False

        return self._model.__class__.load_from_checkpoint(
            ckpt_path,
            generator=self._model._generator,  # pylint: disable=protected-access
            predict_top_k=self._config.top_k,
            exclude_inputs_from_predictions=self._config.exclude_inputs,
        )

    def _preprocess(self):
        pass

    def _postprocess(self, outputs: list[Any] | list[list[Any]]) -> pd.DataFrame:
        dframe = pd.concat(outputs)
        return dframe.reset_index(drop=True)

    def _save(self, dframe: pd.DataFrame) -> None:
        """Save predictions to a file or a table.

        - Save as parquet file if `output_path` is specified. (Will be deprecated)
        - Save as a hive table if `table_name` is specified.
        """
        if self._config.output_path:
            logger.info(f"Writing predictions to {self._config.output_path}")

            fs = file_utils.get_filesystem(self._config.output_path)
            fs.makedirs(os.path.dirname(self._config.output_path), exist_ok=True)

            dframe.to_parquet(self._config.output_path, index=False, compression="snappy")
