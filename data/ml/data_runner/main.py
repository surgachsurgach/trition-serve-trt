"""Main script for generating data for model."""
import os
from typing import Optional

import gin
from loguru import logger

from data.ml.configs import config_parser
from data.ml.data_runner.data_processors import data_processor as dp
from data.ml.data_runner.utils import gin_utils as data_gin_utils
from data.ml.utils import gin_utils  # pylint: disable=unused-import
from data.pylib import watcher


@gin.configurable
def _run(data_processor: Optional[dp.DataProcessor] = None, do_train: bool = False, do_predict: bool = False):
    assert data_processor

    if not do_train and not do_predict:
        logger.error("Neither do_train nor do_predict is set. Doing nothing.")
        return

    if do_train:
        data_processor.write_data()
    if do_predict:
        data_processor.write_data(is_test_split=True)


@watcher.report_error
def main():
    data_gin_utils.resolve_gin_path()

    config_parser.gin_parse_config(
        gin_utils.parse_gin_file(os.getenv("GIN_FILE")),
        gin_utils.parse_gin_params(os.environ.get("GIN_PARAMS")),
    )

    _run()


if __name__ == "__main__":
    main()
