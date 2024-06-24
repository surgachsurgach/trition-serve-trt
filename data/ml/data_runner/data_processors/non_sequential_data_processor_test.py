import os
import tempfile
from typing import Dict, Optional

from absl.testing import absltest
from pyspark.sql import functions as F
from pyspark.sql import types as T

from data.pylib.constant import recsys as common
from data.ml.data_runner.data_processors import data_processor
from data.ml.data_runner.data_processors import non_sequential_data_processor
from data.ml.data_runner.data_processors import test_utils
from data.ml.utils import metadata
from data.pylib.spark_test import spark_test_base
from data.pylib.spark_test import spark_test_utils

_USER_ID_COL = common.USER_ID_COL
_USER_INPUTS_COL = common.USER_INPUTS_COL
_ITEM_ID_COL = common.ITEM_ID_COL
_ITEM_INPUTS_COL = common.ITEM_INPUTS_COL
_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL


def _id2index_udf(id2idx: Dict[int, int]) -> F.udf:
    def _udf(id_: int) -> Optional[int]:
        return id2idx[id_] if id_ in id2idx else None

    return F.udf(_udf, returnType=T.IntegerType())


class DataProcessorTest(spark_test_base.SparkTestBase):
    def test_autoencoder_data_processor(self):
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as tmpdirname:
            train_dev_processor = non_sequential_data_processor.AutoEncoderDataProcessor(
                min_item_interaction=3,
                max_item_interaction=15,
                min_user_interaction=2,
                max_user_interaction=10,
                input_files=test_utils.get_testdata_path("purchase_events.snappy.parquet"),
                output_path=tmpdirname,
                meta_filename="meta.json",
                dev_split_size=0.3,
                random_seed=123456,
                input_target_interaction_col="event_type",
            )
            train_dev_processor.write_data()

            train_expected = self.spark.createDataFrame(
                [
                    # fmt: off
                    {_INPUTS_COL: [10, 8, 9, 7, 6, 5, 1, 2, 4, 0, 3], _TARGETS_COL: [10, 8, 9, 7, 6, 5, 1, 2, 4, 0, 3], },
                    {_INPUTS_COL: [10, 8, 9, 7, 6, 5, 1, 2, 4, 0, 3], _TARGETS_COL: [10, 8, 9, 7, 6, 5, 1, 2, 4, 0, 3], },
                    # fmt: on
                ]
            )
            train_actual = self.spark.read.load(os.path.join(train_dev_processor.output_path, data_processor._TRAIN))
            spark_test_utils.assert_dataframe_equals(train_expected, train_actual)

            dev_expected = self.spark.createDataFrame(
                [
                    # fmt: off
                    {_INPUTS_COL: [1, 2, 4, 0], _TARGETS_COL: [3], }
                    # fmt: on
                ]
            )
            dev_actual = self.spark.read.load(os.path.join(train_dev_processor.output_path, data_processor._DEV))

            spark_test_utils.assert_dataframe_equals(dev_expected, dev_actual)

            meta = metadata.Meta.load(train_dev_processor._output_path)
            assert meta.get_meta(_ITEM_ID_COL, int) == [123, 2123, 2562, 4234, 69054, 37891, 583, 65634, 8932, 9471, 9852]

            test_processor = non_sequential_data_processor.AutoEncoderDataProcessor(
                min_item_interaction=0,
                min_user_interaction=0,
                input_files=test_utils.get_testdata_path("purchase_events.snappy.parquet"),
                output_path=tmpdirname,
                meta_filename="meta.json",
                random_seed=123456,
                input_target_interaction_col="event_type",
            )
            test_processor.write_data(is_test_split=True)

            test_expected = self.spark.createDataFrame(
                [
                    # fmt: off
                    {_USER_ID_COL: 1232, _INPUTS_COL: [0, 1, 2, 3, 4], },
                    {_USER_ID_COL: 1233, _INPUTS_COL: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], },
                    {_USER_ID_COL: 1234, _INPUTS_COL: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], },
                    # fmt: on
                ]
            )

            test_actual = self.spark.read.load(os.path.join(test_processor.output_path, data_processor._TEST)).withColumn(
                _INPUTS_COL, F.array_sort(_INPUTS_COL)
            )

            spark_test_utils.assert_dataframe_equals(test_expected, test_actual, _USER_ID_COL)

    def test_keyword_autoencoder_data_processor(self):
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_proc = non_sequential_data_processor.KeywordAutoEncoderDataProcessor(
                min_item_interaction=1,
                min_user_interaction=1,
                input_files=test_utils.get_testdata_path("keyword_events.snappy.parquet"),
                output_path=tmpdirname,
                meta_filename="meta.json",
                dev_split_size=0.3,
                random_seed=123456,
                input_user_id_col="u_idx",
                input_item_id_col="item.tags",
            )

            data_proc.write_data()

            train_actual = self.spark.read.load(os.path.join(data_proc.output_path, data_processor._TRAIN))
            train_expected = self.spark.createDataFrame(
                [
                    {_INPUTS_COL: [5, 4, 1], _TARGETS_COL: [5, 4, 1]},
                    {_INPUTS_COL: [2, 3, 10], _TARGETS_COL: [2, 3, 10]},
                    {_INPUTS_COL: [6, 9, 11], _TARGETS_COL: [6, 9, 11]},
                ]
            )

            spark_test_utils.assert_dataframe_equals(train_expected, train_actual, order_by=[_INPUTS_COL])

            dev_actual = self.spark.read.load(os.path.join(data_proc.output_path, data_processor._DEV))
            dev_expected = self.spark.createDataFrame(
                [
                    {_INPUTS_COL: [7], _TARGETS_COL: [0]},
                    {_INPUTS_COL: [8], _TARGETS_COL: [0]},
                ]
            )

            spark_test_utils.assert_dataframe_equals(dev_expected, dev_actual, order_by=[_INPUTS_COL])

            meta = metadata.Meta.load(data_proc._output_path)
            assert meta.get_meta(_ITEM_ID_COL, int) == [2201, 1277, 1322, 1405, 1688, 2109, 2198, 2199, 2200, 3316, 3322, 3684]

    def test_purchase_classifier_data_processor(self):
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as tmpdirname:
            train_dev_processor = non_sequential_data_processor.ClassificationDataProcessor(
                min_item_interaction=3,
                max_item_interaction=15,
                min_user_interaction=2,
                max_user_interaction=10,
                input_files=test_utils.get_testdata_path("purchase_events.snappy.parquet"),
                output_path=tmpdirname,
                meta_filename="meta.json",
                dev_split_size=0.3,
                negative_sample_ratio=1.0,
                shuffle_items=False,
                random_seed=123456,
                input_target_interaction_col="event_type",
            )
            train_dev_processor.write_data()

            train_expected = (
                self.spark.createDataFrame(
                    [
                        # fmt: off
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 123, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 37891, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 4234, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 2562, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 2123, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 69054, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 2123, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 69054, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 123, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 4234, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 8932, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 2562, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 9852, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 9471, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 65634, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 583, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 37891, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 2123, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 69054, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 123, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 4234, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 8932, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 2562, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 9852, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 9471, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 65634, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 583, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 37891, _TARGETS_COL: 1},
                        # fmt: on
                    ]
                )
                .withColumn(
                    _USER_INPUTS_COL,
                    _id2index_udf(train_dev_processor.get_meta().get_id_to_idx(_USER_ID_COL, int))(F.col(_USER_INPUTS_COL)),
                )
                .withColumn(
                    _ITEM_INPUTS_COL,
                    _id2index_udf(train_dev_processor.get_meta().get_id_to_idx(_ITEM_ID_COL, int))(F.col(_ITEM_INPUTS_COL)),
                )
            )

            train_actual = self.spark.read.load(os.path.join(train_dev_processor.output_path, data_processor._TRAIN))
            spark_test_utils.assert_dataframe_equals(train_expected, train_actual, [_USER_INPUTS_COL, _ITEM_INPUTS_COL, _TARGETS_COL])

            dev_expected = (
                self.spark.createDataFrame(
                    [
                        # fmt: off
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 2562, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 2123, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 123, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 4234, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 2562, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 37891, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 583, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 65634, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 2123, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 69054, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 123, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 4234, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 2562, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 37891, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 583, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 65634, _TARGETS_COL: 0},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 2123, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 69054, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 123, _TARGETS_COL: 1},
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 4234, _TARGETS_COL: 1},
                        # fmt: on
                    ]
                )
                .withColumn(
                    _USER_INPUTS_COL,
                    _id2index_udf(train_dev_processor.get_meta().get_id_to_idx(_USER_ID_COL, int))(F.col(_USER_INPUTS_COL)),
                )
                .withColumn(
                    _ITEM_INPUTS_COL,
                    _id2index_udf(train_dev_processor.get_meta().get_id_to_idx(_ITEM_ID_COL, int))(F.col(_ITEM_INPUTS_COL)),
                )
            )
            dev_actual = self.spark.read.load(os.path.join(train_dev_processor.output_path, data_processor._DEV))

            spark_test_utils.assert_dataframe_equals(dev_expected, dev_actual, [_USER_INPUTS_COL, _ITEM_INPUTS_COL, _TARGETS_COL])

            meta = metadata.Meta.load(train_dev_processor._output_path)

            assert meta._metadata == {
                _ITEM_ID_COL: ["123", "2123", "2562", "4234", "69054", "37891", "583", "65634", "8932", "9471", "9852"],
                "item_id_prefix": ["", "", "", "", "", "", "", "", "", "", ""],
                _USER_ID_COL: ["1233", "1234", "1232"],
            }

            test_processor = non_sequential_data_processor.ClassificationDataProcessor(
                min_item_interaction=3,
                max_item_interaction=15,
                min_user_interaction=2,
                max_user_interaction=10,
                input_files=test_utils.get_testdata_path("purchase_events.snappy.parquet"),
                output_path=tmpdirname,
                meta_filename="meta.json",
                dev_split_size=0.5,
                negative_sample_ratio=1.0,
                shuffle_items=False,
                random_seed=123456,
                input_target_interaction_col="event_type",
            )
            test_processor.write_data(is_test_split=True)

            test_expected = (
                self.spark.createDataFrame(
                    [
                        # fmt: off
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 2562, },
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 2123, },
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 69054, },
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 123, },
                        {_USER_INPUTS_COL: 1232, _ITEM_INPUTS_COL: 4234, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 8932, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 2562, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 9852, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 9471, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 65634, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 583, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 37891, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 2123, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 69054, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 123, },
                        {_USER_INPUTS_COL: 1233, _ITEM_INPUTS_COL: 4234, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 8932, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 2562, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 9852, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 9471, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 65634, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 583, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 37891, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 2123, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 69054, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 123, },
                        {_USER_INPUTS_COL: 1234, _ITEM_INPUTS_COL: 4234, },
                        # fmt: on
                    ]
                )
                .withColumn(
                    _USER_INPUTS_COL,
                    _id2index_udf(train_dev_processor.get_meta().get_id_to_idx(_USER_ID_COL, int))(F.col(_USER_INPUTS_COL)),
                )
                .withColumn(
                    _ITEM_INPUTS_COL,
                    _id2index_udf(train_dev_processor.get_meta().get_id_to_idx(_ITEM_ID_COL, int))(F.col(_ITEM_INPUTS_COL)),
                )
            )

            test_actual = self.spark.read.load(os.path.join(test_processor.output_path, data_processor._TEST))
            spark_test_utils.assert_dataframe_equals(test_expected, test_actual, [_USER_INPUTS_COL, _ITEM_INPUTS_COL])


if __name__ == "__main__":
    absltest.main()
