import os
import tempfile

from absl.testing import absltest

from data.pylib.constant import recsys as common
from data.ml.data_runner.data_processors import configs
from data.ml.data_runner.data_processors import sequential_data_processor_v2 as sequential_data_processor
from data.ml.data_runner.data_processors import test_utils
from data.pylib.spark_test import spark_test_base
from data.pylib.spark_test import spark_test_utils

_USER_ID_COL = common.USER_ID_COL
_ITEM_ID_COL = common.ITEM_ID_COL
_TIMESTAMP_COL = common.TIMESTAMP_COL
_TARGET_INTERACTION_INPUT_COL = "event_name"

_ITEM_IDX_COL = common.ITEM_IDX_COL
_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL

_TRAIN_DATASET = common.TRAIN_DATASET_NAME
_DEV_DATASET = common.DEV_DATASET_NAME
_TEST_DATASET = common.TEST_DATASET_NAME

_TRAIN_META_FILENAME = common.TRAIN_META_FILENAME
_TEST_META_FILENAME = common.TEST_META_FILENAME


class NonSequentialDataProcessorV2Test(spark_test_base.SparkTestBase):
    def test_user_sequential_data_processor(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "purchase_events.snappy.parquet")
            test_utils.make_parquet_file(
                [
                    {
                        _USER_ID_COL: 1,
                        _ITEM_ID_COL: "1",
                        _TIMESTAMP_COL: 1671345491,
                        _TARGET_INTERACTION_INPUT_COL: "paid",
                    },
                    {
                        _USER_ID_COL: 1,
                        _ITEM_ID_COL: "2",
                        _TIMESTAMP_COL: 1671345492,
                        _TARGET_INTERACTION_INPUT_COL: "paid",
                    },
                    {
                        _USER_ID_COL: 1,
                        _ITEM_ID_COL: "3",
                        _TIMESTAMP_COL: 1671345493,
                        _TARGET_INTERACTION_INPUT_COL: "paid",
                    },
                    {
                        _USER_ID_COL: 1,
                        _ITEM_ID_COL: "4",
                        _TIMESTAMP_COL: 1671345494,
                        _TARGET_INTERACTION_INPUT_COL: "paid",
                    },
                    {
                        _USER_ID_COL: 2,
                        _ITEM_ID_COL: "1",
                        _TIMESTAMP_COL: 1671345495,
                        _TARGET_INTERACTION_INPUT_COL: "paid",
                    },
                    {
                        _USER_ID_COL: 2,
                        _ITEM_ID_COL: "2",
                        _TIMESTAMP_COL: 1671345496,
                        _TARGET_INTERACTION_INPUT_COL: "free",
                    },
                    {
                        _USER_ID_COL: 3,
                        _ITEM_ID_COL: "2",
                        _TIMESTAMP_COL: 1671345497,
                        _TARGET_INTERACTION_INPUT_COL: "paid",
                    },
                    {
                        _USER_ID_COL: 3,
                        _ITEM_ID_COL: "3",
                        _TIMESTAMP_COL: 1671345498,
                        _TARGET_INTERACTION_INPUT_COL: "paid",
                    },
                    {
                        _USER_ID_COL: 3,
                        _ITEM_ID_COL: "4",
                        _TIMESTAMP_COL: 1671345499,
                        _TARGET_INTERACTION_INPUT_COL: "paid",
                    },
                    {
                        _USER_ID_COL: 4,
                        _ITEM_ID_COL: "2",
                        _TIMESTAMP_COL: 1671345500,
                        _TARGET_INTERACTION_INPUT_COL: "paid",
                    },
                    {
                        _USER_ID_COL: 4,
                        _ITEM_ID_COL: "3",
                        _TIMESTAMP_COL: 1671345510,
                        _TARGET_INTERACTION_INPUT_COL: "paid",
                    },
                ],
                filepath,
            )
            processor = sequential_data_processor.UserSequentialDataProcessor(
                config=configs.DataProcessConfig(
                    input_files=filepath,
                    output_path=tmpdirname,
                    dev_split_size=0.3,
                ),
                schema_config=configs.InteractionDataSchemaConfig(
                    input_user_id_col=_USER_ID_COL,
                    input_item_id_col=_ITEM_ID_COL,
                    input_target_interaction_col=_TARGET_INTERACTION_INPUT_COL,
                ),
                train_process_config=configs.InteractionDataProcessConfig(
                    target_interactions=["paid"],
                    user_interaction_range=configs.InteractionRange(min=2, max=4),
                    item_interaction_range=configs.InteractionRange(min=2, max=5),
                ),
                test_process_config=configs.InteractionDataProcessConfig(
                    target_interactions=["paid", "free"],
                    user_interaction_range=configs.InteractionRange(min=1, max=5),
                    item_interaction_range=configs.InteractionRange(min=2, max=5),
                ),
                random_seed=123456,
            )

            processor.write_data()

            train_actual = self.spark.read.load(os.path.join(processor.output_path, _TRAIN_DATASET))
            train_expected = self.spark.createDataFrame(
                [
                    {
                        _USER_ID_COL: 3,
                        _INPUTS_COL: [0, 1, 2],
                    },
                    {
                        _USER_ID_COL: 4,
                        _INPUTS_COL: [0, 1],
                    },
                ]
            )
            train_actual.show(truncate=False)
            spark_test_utils.assert_dataframe_equals(train_actual, train_expected, order_by=[_USER_ID_COL])

            dev_actual = self.spark.read.load(os.path.join(processor.output_path, _DEV_DATASET))
            dev_expected = self.spark.createDataFrame(
                [
                    {
                        _USER_ID_COL: 1,
                        _INPUTS_COL: [3, 0, 1],
                        _TARGETS_COL: [2],
                    }
                ]
            )
            spark_test_utils.assert_dataframe_equals(dev_actual, dev_expected, order_by=[_USER_ID_COL])

            processor.write_data(is_test_split=True)
            test_actual = self.spark.read.load(os.path.join(processor.output_path, _TEST_DATASET))
            test_expected = self.spark.createDataFrame(
                [
                    {
                        _USER_ID_COL: 1,
                        _INPUTS_COL: [3, 0, 1, 2],
                    },
                    {
                        _USER_ID_COL: 2,
                        _INPUTS_COL: [3, 0],
                    },
                    {
                        _USER_ID_COL: 3,
                        _INPUTS_COL: [0, 1, 2],
                    },
                    {
                        _USER_ID_COL: 4,
                        _INPUTS_COL: [0, 1],
                    },
                ],
                schema=test_actual.schema,
            )
            spark_test_utils.assert_dataframe_equals(test_actual, test_expected, order_by=[_USER_ID_COL])

            # Test `max_seq_len` is not null.
            processor = sequential_data_processor.UserSequentialDataProcessor(
                config=configs.DataProcessConfig(
                    input_files=filepath,
                    output_path=tmpdirname,
                    dev_split_size=0.2,
                ),
                schema_config=configs.InteractionDataSchemaConfig(
                    input_user_id_col=_USER_ID_COL,
                    input_item_id_col=_ITEM_ID_COL,
                    input_target_interaction_col=_TARGET_INTERACTION_INPUT_COL,
                ),
                train_process_config=configs.InteractionDataProcessConfig(
                    target_interactions=["paid"],
                    user_interaction_range=configs.InteractionRange(min=2, max=4),
                    item_interaction_range=configs.InteractionRange(min=2, max=5),
                ),
                test_process_config=configs.InteractionDataProcessConfig(
                    target_interactions=["paid", "free"],
                    user_interaction_range=configs.InteractionRange(min=1, max=5),
                    item_interaction_range=configs.InteractionRange(min=2, max=5),
                ),
                random_seed=123456,
                max_seq_len=3,
            )

            processor.write_data()

            train_actual = self.spark.read.load(os.path.join(processor.output_path, _TRAIN_DATASET))
            train_expected = self.spark.createDataFrame(
                [
                    {
                        _USER_ID_COL: 3,
                        _INPUTS_COL: [0, 1, 2],
                    },
                    {
                        _USER_ID_COL: 4,
                        _INPUTS_COL: [0, 1],
                    },
                ]
            )
            spark_test_utils.assert_dataframe_equals(train_actual, train_expected, order_by=[_USER_ID_COL])

            dev_actual = self.spark.read.load(os.path.join(processor.output_path, _DEV_DATASET))
            dev_expected = self.spark.createDataFrame(
                [
                    {
                        _USER_ID_COL: 1,
                        _INPUTS_COL: [0, 1],
                        _TARGETS_COL: [2],
                    }
                ]
            )
            spark_test_utils.assert_dataframe_equals(dev_actual, dev_expected, order_by=[_USER_ID_COL])

            processor.write_data(is_test_split=True)
            test_actual = self.spark.read.load(os.path.join(processor.output_path, _TEST_DATASET))
            test_expected = self.spark.createDataFrame(
                [
                    {
                        _USER_ID_COL: 1,
                        _INPUTS_COL: [0, 1, 2],
                    },
                    {
                        _USER_ID_COL: 2,
                        _INPUTS_COL: [3, 0],
                    },
                    {
                        _USER_ID_COL: 3,
                        _INPUTS_COL: [0, 1, 2],
                    },
                    {
                        _USER_ID_COL: 4,
                        _INPUTS_COL: [0, 1],
                    },
                ],
            )
            spark_test_utils.assert_dataframe_equals(test_actual, test_expected, order_by=[_USER_ID_COL])


if __name__ == "__main__":
    absltest.main()
