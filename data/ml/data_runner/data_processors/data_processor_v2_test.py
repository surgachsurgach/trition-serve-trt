import os
import tempfile

from absl.testing import absltest
from pyspark import ml
from pyspark import sql
from pyspark.ml import feature

from data.pylib.constant import recsys as common
from data.ml.data_runner.data_processors import configs
from data.ml.data_runner.data_processors import data_processor_v2 as data_processor
from data.ml.utils import metadata
from data.pylib.spark_test import spark_test_base

_TRAIN_META_FILENAME = common.TRAIN_META_FILENAME
_TEST_META_FILENAME = common.TEST_META_FILENAME


class _TestDataProcessor(data_processor.DataProcessor):
    def _filter_train_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        return dframe

    def _filter_test_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        return dframe

    def _generate_train_dev_data(self, dframe):
        indexer = feature.StringIndexer(inputCols=["a", "b"], outputCols=["a1", "b1"])
        pipeline = ml.Pipeline(stages=[indexer])
        preprocessor = pipeline.fit(dframe)
        self._save_preprocessor(preprocessor)

        return data_processor.DataSplits(train=dframe, dev=dframe)

    def _generate_test_data(self, dframe):
        return data_processor.DataSplits(test=dframe)

    def _read_data(self, spark):
        data = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
            {"a": 7, "b": 8, "c": 9},
        ]
        return spark.createDataFrame(data)


class DataProcessorTest(spark_test_base.SparkTestBase):
    def test_empty_input_files(self):
        # This test expects any kinds of exception is not raised.
        with tempfile.TemporaryDirectory() as tmpdirname:
            processor = _TestDataProcessor(configs.DataProcessConfig(input_files="", output_path=tmpdirname))
            assert processor.write_data() is None

    def test_write_train_dev_data(self):
        # pylint: disable=protected-access
        with tempfile.NamedTemporaryFile() as tmpfilename, tempfile.TemporaryDirectory() as tmpdirname:
            processor = _TestDataProcessor(
                configs.DataProcessConfig(
                    input_files=tmpfilename.name,
                    output_path=tmpdirname,
                )
            )
            processor.write_data()

            meta = metadata.Meta.load(processor.output_path, _TRAIN_META_FILENAME)

            train_expected = self.spark.createDataFrame(
                [
                    {"a": 1, "b": 2, "c": 3},
                    {"a": 4, "b": 5, "c": 6},
                    {"a": 7, "b": 8, "c": 9},
                ]
            )

            dev_expected = self.spark.createDataFrame(
                [
                    {"a": 1, "b": 2, "c": 3},
                    {"a": 4, "b": 5, "c": 6},
                    {"a": 7, "b": 8, "c": 9},
                ]
            )

            train_actual = self.spark.read.parquet(os.path.join(processor.output_path, data_processor._TRAIN_DATASET))

            assert train_expected.orderBy("a", "b", "c").collect() == train_actual.orderBy("a", "b", "c").collect()

            dev_actual = self.spark.read.parquet(os.path.join(processor.output_path, data_processor._TRAIN_DATASET))

            assert dev_expected.orderBy("a", "b", "c").collect() == dev_actual.orderBy("a", "b", "c").collect()
            assert {"a": ["1", "4", "7"], "b": ["2", "5", "8"]} == meta._metadata, meta._metadata

    def test_write_test_data(self):
        # pylint: disable=protected-access
        with tempfile.NamedTemporaryFile() as tmpfilename, tempfile.TemporaryDirectory() as tmpdirname:
            test_processor = _TestDataProcessor(
                configs.DataProcessConfig(
                    input_files=tmpfilename.name,
                    output_path=tmpdirname,
                )
            )
            test_processor.write_data(is_test_split=True)

            test_expected = self.spark.createDataFrame(
                [
                    {"a": 1, "b": 2, "c": 3},
                    {"a": 4, "b": 5, "c": 6},
                    {"a": 7, "b": 8, "c": 9},
                ]
            )

            test_actual = self.spark.read.parquet(os.path.join(test_processor.output_path, data_processor._TEST_DATASET))

            assert test_expected.orderBy("a", "b", "c").collect() == test_actual.orderBy("a", "b", "c").collect()


if __name__ == "__main__":
    absltest.main()
