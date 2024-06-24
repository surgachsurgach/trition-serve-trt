import os
import tempfile

from absl.testing import absltest

from data.pylib.constant import recsys as common
from data.ml.data_runner.data_processors import configs
from data.ml.data_runner.data_processors import non_sequential_data_processor_v2 as non_sequential_data_processor
from data.ml.data_runner.data_processors import test_utils
from data.ml.utils import metadata
from data.pylib.spark_test import spark_test_base
from data.pylib.spark_test import spark_test_utils

_USER_ID_COL = common.USER_ID_COL
_ITEM_ID_COL = common.ITEM_ID_COL
_ITEM_IDX_COL = common.ITEM_IDX_COL
_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_WEIGHT_COL = common.WEIGHT_COL
_INPUT_WEIGHTS_COL = common.INPUT_WEIGHTS_COL
_TARGET_WEIGHTS_COL = common.TARGET_WEIGHTS_COL
_TARGET_INTERACTION_INPUT_COL = "event_name"

_POSITIVE_CONTEXTS_COL = common.POSITIVE_CONTEXTS_COL
_NEGATIVE_CONTEXTS_COL = common.NEGATIVE_CONTEXTS_COL

_GENDER_COL = common.GENDER_COL
_GENDER_IDX_COL = common.GENDER_IDX_COL
_GENERATION_COL = common.GENERATION_COL
_GENERATION_IDX_COL = common.GENERATION_IDX_COL

_TRAIN_DATASET = common.TRAIN_DATASET_NAME
_DEV_DATASET = common.DEV_DATASET_NAME
_TEST_DATASET = common.TEST_DATASET_NAME

_TRAIN_META_FILENAME = common.TRAIN_META_FILENAME
_TEST_META_FILENAME = common.TEST_META_FILENAME


class NonSequentialDataProcessorV2Test(spark_test_base.SparkTestBase):
    def test_user_autoencoder_data_processor(self):
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_utils.make_parquet_file(
                [
                    {_USER_ID_COL: 1, _ITEM_ID_COL: "1", _WEIGHT_COL: 1.0, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 1, _ITEM_ID_COL: "2", _WEIGHT_COL: 2.0, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 1, _ITEM_ID_COL: "3", _WEIGHT_COL: 1.5, _TARGET_INTERACTION_INPUT_COL: "free"},
                    {_USER_ID_COL: 2, _ITEM_ID_COL: "1", _WEIGHT_COL: 1.0, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 2, _ITEM_ID_COL: "2", _WEIGHT_COL: 1.1, _TARGET_INTERACTION_INPUT_COL: "free"},
                    {_USER_ID_COL: 3, _ITEM_ID_COL: "2", _WEIGHT_COL: 1.2, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 3, _ITEM_ID_COL: "3", _WEIGHT_COL: 1.3, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 3, _ITEM_ID_COL: "4", _WEIGHT_COL: 1.4, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 4, _ITEM_ID_COL: "3", _WEIGHT_COL: 0.5, _TARGET_INTERACTION_INPUT_COL: "paid"},
                ],
                os.path.join(tmpdirname, "purchase_events.snappy.parquet"),
            )
            data_proc = non_sequential_data_processor.UserAutoEncoderDataProcessor(
                config=configs.DataProcessConfig(
                    input_files=os.path.join(tmpdirname, "purchase_events.snappy.parquet"),
                    output_path=tmpdirname,
                    dev_split_size=0.3,
                ),
                schema_config=configs.InteractionDataSchemaConfig(
                    input_user_id_col=_USER_ID_COL,
                    input_item_id_col=_ITEM_ID_COL,
                    input_weight_col=_WEIGHT_COL,
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
            data_proc.write_data()

            train_actual = self.spark.read.load(os.path.join(data_proc.output_path, _TRAIN_DATASET))
            train_expected = self.spark.createDataFrame(
                [
                    {
                        _INPUTS_COL: [2, 0],
                        _TARGETS_COL: [2, 0],
                        _INPUT_WEIGHTS_COL: [1.3, 1.2],
                        _TARGET_WEIGHTS_COL: [1.3, 1.2],
                    },
                ]
            )
            spark_test_utils.assert_dataframe_equals(train_actual, train_expected, order_by=[_INPUTS_COL])
            dev_actual = self.spark.read.load(os.path.join(data_proc.output_path, _DEV_DATASET))
            dev_expected = self.spark.createDataFrame(
                [
                    {
                        _INPUTS_COL: [1],
                        _TARGETS_COL: [0],
                        _INPUT_WEIGHTS_COL: [1.0],
                        _TARGET_WEIGHTS_COL: [2.0],
                    },
                ]
            )
            spark_test_utils.assert_dataframe_equals(dev_actual, dev_expected, order_by=[_INPUTS_COL])

            data_proc.write_data(is_test_split=True)
            test_actual = self.spark.read.load(os.path.join(data_proc.output_path, _TEST_DATASET))
            test_expected = self.spark.createDataFrame(
                [
                    {_USER_ID_COL: 1, _INPUTS_COL: [1, 0, 2], _INPUT_WEIGHTS_COL: [1.0, 2.0, 1.5]},
                    {_USER_ID_COL: 2, _INPUTS_COL: [1, 0], _INPUT_WEIGHTS_COL: [1.0, 1.1]},
                    {_USER_ID_COL: 3, _INPUTS_COL: [0, 2], _INPUT_WEIGHTS_COL: [1.2, 1.3]},
                    {_USER_ID_COL: 4, _INPUTS_COL: [2], _INPUT_WEIGHTS_COL: [0.5]},
                ],
                schema=test_actual.schema,
            )
            spark_test_utils.assert_dataframe_equals(test_actual, test_expected, order_by=[_USER_ID_COL])

            train_meta = metadata.Meta.load(data_proc.output_path, data_proc._train_meta_filename)
            assert train_meta._metadata == {
                "num_users": 2,
                "num_items": 3,
                "num_interactions": 4,
                "item_id": ["2", "1", "3"],
                "item_id_prefix": ["", "", ""],
            }

            test_meta = metadata.Meta.load(data_proc.output_path, data_proc._test_meta_filename)

            assert test_meta._metadata == {"num_users": 4, "num_items": 3, "num_interactions": 8}

    def test_user_conditional_autoencoder_data_processor(self):
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_utils.make_parquet_file(
                [
                    {_USER_ID_COL: 1, _ITEM_ID_COL: "1", _WEIGHT_COL: 1.0, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 1, _ITEM_ID_COL: "2", _WEIGHT_COL: 2.0, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 1, _ITEM_ID_COL: "3", _WEIGHT_COL: 1.5, _TARGET_INTERACTION_INPUT_COL: "free"},
                    {_USER_ID_COL: 2, _ITEM_ID_COL: "1", _WEIGHT_COL: 1.0, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 2, _ITEM_ID_COL: "2", _WEIGHT_COL: 1.1, _TARGET_INTERACTION_INPUT_COL: "free"},
                    {_USER_ID_COL: 3, _ITEM_ID_COL: "2", _WEIGHT_COL: 1.2, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 3, _ITEM_ID_COL: "3", _WEIGHT_COL: 1.3, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 3, _ITEM_ID_COL: "4", _WEIGHT_COL: 1.4, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 4, _ITEM_ID_COL: "3", _WEIGHT_COL: 0.5, _TARGET_INTERACTION_INPUT_COL: "paid"},
                ],
                os.path.join(tmpdirname, "purchase_events.snappy.parquet"),
            )
            test_utils.make_parquet_file(
                [
                    {_USER_ID_COL: 1, _GENDER_COL: "M", _GENERATION_COL: "10"},
                    {_USER_ID_COL: 2, _GENDER_COL: "F", _GENERATION_COL: "20"},
                    {_USER_ID_COL: 3, _GENDER_COL: None, _GENERATION_COL: None},
                    {_USER_ID_COL: 4, _GENDER_COL: None, _GENERATION_COL: "40"},
                ],
                os.path.join(tmpdirname, "user_data.snappy.parquet"),
            )
            data_proc = non_sequential_data_processor.UserConditionalAutoEncoderDataProcessor(
                config=configs.DataProcessConfig(
                    input_files=os.path.join(tmpdirname, "purchase_events.snappy.parquet"),
                    output_path=tmpdirname,
                    dev_split_size=0.3,
                ),
                schema_config=configs.InteractionDataSchemaConfig(
                    input_user_id_col=_USER_ID_COL,
                    input_item_id_col=_ITEM_ID_COL,
                    input_weight_col=_WEIGHT_COL,
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
                user_data_config=configs.UserConditionalDataConfig(
                    input_files=os.path.join(tmpdirname, "user_data.snappy.parquet"),
                    input_user_id_col=_USER_ID_COL,
                    input_gender_col=_GENDER_COL,
                    input_generation_col=_GENERATION_COL,
                ),
                random_seed=123456,
            )
            data_proc.write_data()

            train_actual = self.spark.read.load(os.path.join(data_proc.output_path, _TRAIN_DATASET))
            train_expected = self.spark.createDataFrame(
                [
                    {
                        _INPUTS_COL: [2, 0],
                        _TARGETS_COL: [2, 0],
                        _INPUT_WEIGHTS_COL: [1.3, 1.2],
                        _TARGET_WEIGHTS_COL: [1.3, 1.2],
                        _GENDER_IDX_COL: 0,
                        _GENERATION_IDX_COL: 3,
                    },
                ]
            )
            spark_test_utils.assert_dataframe_equals(train_actual, train_expected, order_by=[_INPUTS_COL])
            dev_actual = self.spark.read.load(os.path.join(data_proc.output_path, _DEV_DATASET))
            dev_expected = self.spark.createDataFrame(
                [
                    {
                        _INPUTS_COL: [1],
                        _TARGETS_COL: [0],
                        _INPUT_WEIGHTS_COL: [1.0],
                        _TARGET_WEIGHTS_COL: [2.0],
                        _GENDER_IDX_COL: 2,
                        _GENERATION_IDX_COL: 0,
                    },
                ]
            )
            spark_test_utils.assert_dataframe_equals(dev_actual, dev_expected, order_by=[_INPUTS_COL])

            data_proc.write_data(is_test_split=True)
            test_actual = self.spark.read.load(os.path.join(data_proc.output_path, _TEST_DATASET))
            test_expected = self.spark.createDataFrame(
                [
                    {
                        _USER_ID_COL: 1,
                        _INPUTS_COL: [1, 0, 2],
                        _INPUT_WEIGHTS_COL: [1.0, 2.0, 1.5],
                        _GENDER_IDX_COL: 2,
                        _GENERATION_IDX_COL: 0,
                    },
                    {_USER_ID_COL: 2, _INPUTS_COL: [1, 0], _INPUT_WEIGHTS_COL: [1.0, 1.1], _GENDER_IDX_COL: 1, _GENERATION_IDX_COL: 1},
                    {_USER_ID_COL: 3, _INPUTS_COL: [0, 2], _INPUT_WEIGHTS_COL: [1.2, 1.3], _GENDER_IDX_COL: 0, _GENERATION_IDX_COL: 3},
                    {_USER_ID_COL: 4, _INPUTS_COL: [2], _INPUT_WEIGHTS_COL: [0.5], _GENDER_IDX_COL: 0, _GENERATION_IDX_COL: 2},
                ],
                schema=test_actual.schema,
            )
            spark_test_utils.assert_dataframe_equals(test_actual, test_expected, order_by=[_USER_ID_COL])

            train_meta = metadata.Meta.load(data_proc.output_path, data_proc._train_meta_filename)
            assert train_meta._metadata == {
                "num_users": 2,
                "num_items": 3,
                "num_interactions": 4,
                "item_id": ["2", "1", "3"],
                "gender": ["unknown", "F", "M"],
                "generation": ["10", "20", "40", "unknown"],
                "item_id_prefix": ["", "", ""],
            }

            test_meta = metadata.Meta.load(data_proc.output_path, data_proc._test_meta_filename)

            assert test_meta._metadata == {"num_users": 4, "num_items": 3, "num_interactions": 8}

    def test_item_embedding_data_processor(self):
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as tmpdirname:
            test_utils.make_parquet_file(
                [
                    {_USER_ID_COL: 1, _ITEM_ID_COL: "1", _WEIGHT_COL: 1.0, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 1, _ITEM_ID_COL: "2", _WEIGHT_COL: 2.0, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 1, _ITEM_ID_COL: "3", _WEIGHT_COL: 1.5, _TARGET_INTERACTION_INPUT_COL: "free"},
                    {_USER_ID_COL: 2, _ITEM_ID_COL: "1", _WEIGHT_COL: 1.0, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 2, _ITEM_ID_COL: "2", _WEIGHT_COL: 1.1, _TARGET_INTERACTION_INPUT_COL: "free"},
                    {_USER_ID_COL: 3, _ITEM_ID_COL: "2", _WEIGHT_COL: 1.2, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 3, _ITEM_ID_COL: "3", _WEIGHT_COL: 1.3, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 3, _ITEM_ID_COL: "4", _WEIGHT_COL: 1.4, _TARGET_INTERACTION_INPUT_COL: "paid"},
                    {_USER_ID_COL: 4, _ITEM_ID_COL: "3", _WEIGHT_COL: 0.5, _TARGET_INTERACTION_INPUT_COL: "paid"},
                ],
                os.path.join(tmpdirname, "purchase_events.snappy.parquet"),
            )

            data_proc = non_sequential_data_processor.ItemEmbeddingDataProcessor(
                config=configs.DataProcessConfig(
                    input_files=os.path.join(tmpdirname, "purchase_events.snappy.parquet"),
                    output_path=tmpdirname,
                    dev_split_size=0.3,
                ),
                schema_config=configs.InteractionDataSchemaConfig(
                    input_user_id_col=_USER_ID_COL,
                    input_item_id_col=_ITEM_ID_COL,
                    input_weight_col=_WEIGHT_COL,
                    input_target_interaction_col=_TARGET_INTERACTION_INPUT_COL,
                ),
                train_process_config=configs.InteractionDataProcessConfig(
                    user_interaction_range=configs.InteractionRange(min=2, max=4),
                    item_interaction_range=configs.InteractionRange(min=2, max=5),
                ),
                test_process_config=configs.InteractionDataProcessConfig(
                    user_interaction_range=configs.InteractionRange(min=1, max=5),
                    item_interaction_range=configs.InteractionRange(min=2, max=5),
                ),
                random_seed=123456,
            )
            data_proc.write_data()

            train_actual = self.spark.read.load(os.path.join(data_proc.output_path, _TRAIN_DATASET))
            train_expected = self.spark.createDataFrame(
                [
                    {_INPUTS_COL: [1, 0]},
                    {_INPUTS_COL: [2, 0]},
                ]
            )
            spark_test_utils.assert_dataframe_equals(train_actual, train_expected, order_by=[_INPUTS_COL])

            dev_actual = self.spark.read.load(os.path.join(data_proc.output_path, _DEV_DATASET))
            dev_expected = self.spark.createDataFrame(
                [
                    {_INPUTS_COL: [2, 1, 0]},
                ]
            )
            spark_test_utils.assert_dataframe_equals(dev_actual, dev_expected, order_by=[_INPUTS_COL])

            data_proc.write_data(is_test_split=True)

            test_actual = self.spark.read.load(os.path.join(data_proc.output_path, _TEST_DATASET))
            test_expected = self.spark.createDataFrame(
                [
                    {_ITEM_IDX_COL: 0},
                    {_ITEM_IDX_COL: 1},
                    {_ITEM_IDX_COL: 2},
                ]
            )
            spark_test_utils.assert_dataframe_equals(test_actual, test_expected, order_by=[_ITEM_IDX_COL])

            train_meta = metadata.Meta.load(data_proc.output_path, data_proc._train_meta_filename)
            assert train_meta._metadata == {
                "num_users": 3,
                "num_items": 3,
                "num_interactions": 7,
                "item_frequency": [3, 2, 2],
                "item_id": ["2", "1", "3"],
                "item_id_prefix": ["", "", ""],
            }

            test_meta = metadata.Meta.load(data_proc.output_path, data_proc._test_meta_filename)
            assert test_meta._metadata == {"num_users": 4, "num_items": 3, "num_interactions": 8}


if __name__ == "__main__":
    absltest.main()
