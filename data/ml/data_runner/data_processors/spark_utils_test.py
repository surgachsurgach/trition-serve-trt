from absl.testing import absltest

from data.ml.data_runner.data_processors import spark_utils
from data.pylib.spark_test import spark_test_base
from data.pylib.spark_test import spark_test_utils


class SpakrUtilsTest(spark_test_base.SparkTestBase):
    def test_split_randomly(self):
        dframe = self.spark.createDataFrame(
            [
                {"col1": 10, "col2": 1},
                {"col1": 10, "col2": 2},
                {"col1": 10, "col2": 3},
                {"col1": 11, "col2": 4},
                {"col1": 11, "col2": 6},
            ]
        )

        train_actual, dev_actual = spark_utils.split_randomly(dframe, "col1", 0.7, seed=12345)
        train_expected = self.spark.createDataFrame(
            [
                {"col1": 10, "col2": 2},
                {"col1": 10, "col2": 3},
                {"col1": 11, "col2": 4},
            ]
        )
        dev_expected = self.spark.createDataFrame(
            [
                {"col1": 10, "col2": 1},
                {"col1": 11, "col2": 6},
            ]
        )

        spark_test_utils.assert_dataframe_equals(train_actual, train_expected, ["col1", "col2"])
        spark_test_utils.assert_dataframe_equals(dev_actual, dev_expected, ["col1", "col2"])

    def test_split_chronologically(self):
        dframe = self.spark.createDataFrame(
            [
                {"col1": 10, "col2": 1, "timestamp": 1},
                {"col1": 10, "col2": 2, "timestamp": 2},
                {"col1": 10, "col2": 3, "timestamp": 3},
                {"col1": 11, "col2": 4, "timestamp": 2},
                {"col1": 11, "col2": 6, "timestamp": 5},
            ]
        )

        train_actual, dev_actual = spark_utils.split_chronologically(dframe, "col1", "timestamp", 0.7, seed=12345)
        train_expected = self.spark.createDataFrame(
            [
                {"col1": 10, "col2": 1, "timestamp": 1},
                {"col1": 11, "col2": 4, "timestamp": 2},
                {"col1": 10, "col2": 2, "timestamp": 2},
            ]
        )
        dev_expected = self.spark.createDataFrame(
            [
                {"col1": 11, "col2": 6, "timestamp": 5},
                {"col1": 10, "col2": 3, "timestamp": 3},
            ]
        )

        spark_test_utils.assert_dataframe_equals(train_actual, train_expected, ["col1", "col2", "timestamp"])
        spark_test_utils.assert_dataframe_equals(dev_actual, dev_expected, ["col1", "col2", "timestamp"])

    def test_negative_sampling(self):
        dframe = self.spark.createDataFrame(
            [
                {"col1": 10, "col2": 1},
                {"col1": 10, "col2": 2},
                {"col1": 10, "col2": 3},
                {"col1": 11, "col2": 4},
                {"col1": 11, "col2": 6},
            ]
        )

        item_ids = list(range(10))

        negative_sampling = spark_utils.negative_sampling("col1", "col2", "targets", item_ids, 1.0, False, seed=12345)

        actual = negative_sampling(dframe)
        expected = self.spark.createDataFrame(
            [
                {"col1": 10, "col2": 1, "targets": 1},
                {"col1": 10, "col2": 2, "targets": 1},
                {"col1": 10, "col2": 3, "targets": 1},
                {"col1": 10, "col2": 0, "targets": 0},
                {"col1": 10, "col2": 4, "targets": 0},
                {"col1": 10, "col2": 5, "targets": 0},
                {"col1": 11, "col2": 4, "targets": 1},
                {"col1": 11, "col2": 6, "targets": 1},
                {"col1": 11, "col2": 0, "targets": 0},
                {"col1": 11, "col2": 1, "targets": 0},
            ]
        )

        spark_test_utils.assert_dataframe_equals(actual, expected, order_by=["col1", "col2"])


if __name__ == "__main__":
    absltest.main()
