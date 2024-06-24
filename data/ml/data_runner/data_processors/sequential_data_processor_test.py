import json
import os
import tempfile

from absl.testing import absltest

from data.pylib.constant import recsys as common
from data.ml.data_runner.data_processors import data_processor
from data.ml.data_runner.data_processors import sequential_data_processor
from data.ml.data_runner.data_processors import test_utils
from data.ml.utils import metadata
from data.pylib.spark_test import spark_test_base
from data.pylib.spark_test import spark_test_utils

_USER_ID_COL = common.USER_ID_COL
_ITEM_ID_COL = common.ITEM_ID_COL


class SequentialDataProcessorTest(spark_test_base.SparkTestBase):
    def test_sequential_data_processor(self):
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as tmpdirname:
            train_dev_processor = sequential_data_processor.SequentialDataProcessor(
                side_features=["category", "author"],
                min_item_interaction=1,
                min_user_interaction=1,
                input_files=test_utils.get_testdata_path("purchase_events_with_meta.snappy.parquet"),
                output_path=tmpdirname,
                meta_filename="meta.json",
                dev_split_size=0.3,
                random_seed=0,
                input_user_id_col="u_idx",
                input_item_id_col="item.id",
            )

            train_dev_processor.write_data()

            output_cols = [_USER_ID_COL, "timestamp", _ITEM_ID_COL, "item_category", "item_author"]
            train_expected = self.spark.createDataFrame(
                [
                    (
                        "1039939",
                        [1671374995, 1671374995, 1671374995, 1671374995, 1671374995, 1671374995, 1671374995, 1671374995],
                        [0, 2, 3, 4, 5, 6, 7, 8],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [6, 1, 4, 8, 7, 0, 3, 2],
                    )
                ],
                schema=output_cols,
            )

            train_actual = self.spark.read.load(os.path.join(train_dev_processor.output_path, data_processor._TRAIN))
            spark_test_utils.assert_dataframe_equals(train_expected, train_actual, _USER_ID_COL)

            dev_expected = self.spark.createDataFrame(
                [
                    ("1037713", [1671348246], [1], [0], [5]),
                ],
                schema=output_cols,
            )

            dev_actual = self.spark.read.load(os.path.join(train_dev_processor.output_path, data_processor._DEV))
            spark_test_utils.assert_dataframe_equals(dev_expected, dev_actual, _USER_ID_COL)

            test_processor = sequential_data_processor.SequentialDataProcessor(
                side_features=["category", "author"],
                min_item_interaction=1,
                min_user_interaction=1,
                input_files=test_utils.get_testdata_path("purchase_events_with_meta.snappy.parquet"),
                output_path=tmpdirname,
                meta_filename="meta.json",
                dev_split_size=0.3,
                random_seed=0,
                input_user_id_col="u_idx",
                input_item_id_col="item.id",
            )

            test_processor.write_data(is_test_split=True)

            test_expected = self.spark.createDataFrame(
                [
                    (
                        "1039939",
                        [1671374995, 1671374995, 1671374995, 1671374995, 1671374995, 1671374995, 1671374995, 1671374995],
                        [0, 2, 3, 4, 5, 6, 7, 8],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [6, 1, 4, 8, 7, 0, 3, 2],
                    ),
                    ("1037713", [1671348246], [1], [0], [5]),
                ],
                schema=output_cols,
            )

            test_actual = self.spark.read.load(os.path.join(test_processor.output_path, data_processor._TEST))
            spark_test_utils.assert_dataframe_equals(test_expected, test_actual, _USER_ID_COL)

            with open(os.path.join(tmpdirname, "meta.json"), encoding="utf-8") as meta_file:
                meta = json.load(meta_file)

            meta = metadata.Meta.load(train_dev_processor._output_path)
            assert meta.get_meta(_ITEM_ID_COL) == [
                "1173002290",
                "2093066897",
                "2093071667",
                "2213010207",
                "2630000104",
                "3302000898",
                "3586021885",
                "961014800",
                "961038072",
            ]

            assert meta.get_meta("item_id_prefix") == ["romance" for _ in range(9)]


if __name__ == "__main__":
    absltest.main()
