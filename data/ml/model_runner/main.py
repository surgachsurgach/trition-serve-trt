import os

import gin
from loguru import logger
import torch

from data.ml.configs import config_parser
from data.ml.model_runner import configs  # pylint: disable=unused-import
from data.ml.model_runner import datasets  # pylint: disable=unused-import
from data.ml.model_runner import models  # pylint: disable=unused-import
from data.ml.model_runner import pipeline  # pylint: disable=unused-import
from data.ml.model_runner.base import predictor_base
from data.ml.model_runner.base import trainer_base
from data.ml.model_runner.utils import gin_utils as model_gin_utils
from data.ml.utils import gin_utils  # pylint: disable=unused-import
from data.ml.utils import metadata  # pylint: disable=unused-import
from data.pylib import watcher

try:
    # Import to enable tensorboard to write logs to s3.
    import tensorflow_io as tfio  # pylint: disable=unused-import
except ImportError:
    logger.warning("tensorflow-io is not installed. Some file systems may not be supported for tensorboard logging.")


def _train(trainer: trainer_base.TrainerBase):
    logger.info("starts training.")
    trainer.train()


def _predict(predictor: predictor_base.PredictorBase):
    logger.info("starts predicting.")
    predictor.predict()


@gin.configurable
def _run(
    do_train: bool = False,
    trainer: trainer_base.TrainerBase | None = None,
    do_predict: bool = False,
    predictor: predictor_base.PredictorBase | None = None,
):
    if not do_train and not do_predict:
        logger.error("Neither do_train nor do_predict is set. Doing nothing.")
        return

    if do_train:
        assert trainer, "`trainer` is not provided."
        _train(trainer)

    if do_predict:
        assert predictor, "`predictor` is not provided."
        _predict(predictor)


@watcher.report_error
def main():
    model_gin_utils.resolve_gin_path()

    # check file existence at GIN_FILE
    logger.error(f"Files in the directory: {os.listdir(os.path.dirname(os.getenv('GIN_FILE')))}")

    config_parser.gin_parse_config(
        gin_utils.parse_gin_file(os.getenv("GIN_FILE")),
        gin_utils.parse_gin_params(os.getenv("GIN_PARAMS")),
    )

    _run()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
