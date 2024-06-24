from absl.testing import absltest

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

        spark_test_utils.assert_dataframe_equals(dframe, dframe, ["col1", "col2"])


if __name__ == "__main__":
    absltest.main()
