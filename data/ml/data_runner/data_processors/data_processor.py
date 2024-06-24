import abc
import collections
import dataclasses
import math
import os
from typing import Any, final, Iterable, List, Optional, Union

from loguru import logger
from pyspark import conf
from pyspark import ml
from pyspark import sql
from pyspark.ml import feature
from pyspark.sql import functions as F

from data.ml.utils import file_utils
from data.ml.utils import metadata
from data.pylib.constant import recsys as common

_TRAIN = "train"
_DEV = "dev"
_TEST = "test"

_PREPROCESSOR = "preprocessor"


@dataclasses.dataclass
class DataSplits:
    train: Optional[sql.DataFrame] = dataclasses.field(default=None)
    dev: Optional[sql.DataFrame] = dataclasses.field(default=None)
    test: Optional[sql.DataFrame] = dataclasses.field(default=None)


def _ensure_list(value: Any) -> List:
    if not isinstance(value, list):
        return [value]
    return value


def _ensure_output_path_exists(dirname: str) -> str:
    fs = file_utils.get_filesystem(dirname)
    fs.makedirs(dirname, exist_ok=True)
    return dirname


def _create_spark(app_name: str, run_locally: bool, scheme: str):
    spark_builder = sql.SparkSession.builder
    spark_conf = conf.SparkConf()
    if run_locally:
        logger.info("Run PySpark locally.")
        spark_builder = spark_builder.appName(app_name)
        spark_conf.set("spark.master", "local[*]")
        spark_conf.set("spark.driver.host", "localhost")

    spark_builder = spark_builder.config(conf=spark_conf)
    return spark_builder.getOrCreate()


TrainDevData = collections.namedtuple("TrainDevData", ["preprocessor", "data_splits"])


class DataProcessor(abc.ABC):
    """Abstract class for data processors."""

    def __init__(
        self,
        input_files: Union[str, Iterable[str]],
        output_path: str,
        meta_filename="meta.json",
        dev_split_size: float = 0.1,
        run_locally: bool = False,
        num_partitions: int = 20,
    ):
        """Constructs DataProcessor instance.

        DataProcessor reads data files from local or S3.
        Subclasses of DataProcessor should implement `_read_data` and `_generate_data` methods.
        `_read_data` is responsible for reading raw data files from local or S3 in its proper data format such as
        CSV and parquets. `_read_data` should return the raw data as Spark DataFrame.
        `_generate_data` converts a plain DataFrame into processed DataFrame. For example,
        it can filter out unnecessary rows.

        Args:
            input_files: string or a list of strings. A list of input files. Input files can be local files or
                S3 objects. For example, ['data/file1.csv', 'data/file2.csv'] or ['s3://bucket/file1.csv',
                's3://bucket/file2.csv']. All input files must have the same scheme.
            output_path: Output directory path. It can be local path or S3.
            meta_filename: Filename of metadata. This file contains the number of items,
                item-to-index mappings, the number of users, and user-to-index mappings.
            dev_split_size: The proportion of the dataset to include in the development(dev) split.
            run_locally: If True, DataProcessor create Spark in local mode. Otherwise, Spark job is deployed to
                its cluster.
            num_partitions: The number of re-partitions. After processing, the number of parquet files will be equal to this.
                This is applied to training datasets only. Increasing num_partitions improves the randomness in shuffling and the
                stability of the model. However, if it is too high, many small-sized files will be output.
        """
        self._input_files = _ensure_list(input_files)
        self._output_path = _ensure_output_path_exists(output_path)
        self._meta_filename = meta_filename
        self._dev_split_size = dev_split_size
        self._run_locally = run_locally
        self._num_partitions = num_partitions

    @property
    def output_path(self):
        return self._output_path

    @abc.abstractmethod
    def _read_data(self, spark: sql.SparkSession) -> sql.DataFrame:
        """Reads raw data and returns Spark DataFrame."""

    @abc.abstractmethod
    def _generate_train_dev_data(self, dframe: sql.DataFrame) -> TrainDevData:
        """Reads raw data files. Returns intermediate data and item metadata."""

    @abc.abstractmethod
    def _generate_test_data(self, dframe: sql.DataFrame) -> DataSplits:
        """Reads raw data files and returns test data."""

    def write_data(self, is_test_split=False):
        """Writes processed data in parquet files.

        Args:
            is_test_split: If True, generate data for test.
        """
        logger.info(f"Writing data to {self._output_path}.")
        spark = _create_spark(
            self.__class__.__name__,
            self._run_locally,
            file_utils.get_scheme(self._input_files[0]),
        )
        dframe = self._read_data(spark)

        if not is_test_split:
            train_dev_data = self._generate_train_dev_data(dframe)

            (
                train_dev_data.data_splits.train.repartition(self._num_partitions)
                .write.mode("overwrite")
                .parquet(os.path.join(self._output_path, _TRAIN))
            )

            if train_dev_data.preprocessor:
                train_dev_data.preprocessor.write().overwrite().save(os.path.join(self._output_path, _PREPROCESSOR))
                self._generate_meta(train_dev_data.preprocessor)

            if train_dev_data.data_splits.dev:
                train_dev_data.data_splits.dev.write.mode("overwrite").parquet(os.path.join(self._output_path, _DEV))
        else:
            data_splits = self._generate_test_data(dframe)
            data_splits.test.write.mode("overwrite").parquet(os.path.join(self._output_path, _TEST))

    def _generate_meta(self, preprocessor: ml.PipelineModel):
        logger.info("Generating metadata...")
        first_stage = preprocessor.stages[0]
        if isinstance(first_stage, feature.StringIndexerModel):
            meta = metadata.Meta()
            for i, col in enumerate(first_stage.getInputCols()):
                meta.add_meta(col, list(first_stage.labelsArray[i]))
            meta.save(self._output_path, self._meta_filename)
        else:
            raise RuntimeError("First stage of pipeline must be StringIndexerModel.")

    def get_meta(self) -> metadata.Meta:
        return metadata.Meta.load(self._output_path, self._meta_filename)

    def load_preprocessor(self) -> ml.PipelineModel:
        return ml.PipelineModel.load(os.path.join(self._output_path, _PREPROCESSOR))


class InteractionDataProcessor(DataProcessor):
    _user_id_col = common.USER_ID_COL
    _item_id_col = common.ITEM_ID_COL
    _target_interaction_col = common.TARGET_INTERACTION_COL

    def __init__(
        self,
        min_item_interaction=10,
        max_item_interaction=math.inf,
        min_user_interaction=10,
        max_user_interaction=math.inf,
        target_interaction: str = "purchase",
        input_item_id_col: str = "item_id",
        input_target_interaction_col: str = "event_name",
        input_user_id_col: str = "user_id",
        random_seed=None,
        **kwargs,
    ):
        """Constructs `InteractionDataProcessor`.

        Args:
            min_item_interaction:
                The minimum number of item interactions required for a user.
                If a user has item interactions less than `min_item_interaction` will be ignored.
            max_item_interaction:
                The maximum number of item interactions required for a user.
                If a user has item interactions greater than `max_item_interaction` will be ignored.
            min_user_interaction:
                The minimum number of user interactions required for an item.
                If an item has user interactions less than `min_user_interaction` will be ignored.
            max_user_interaction:
                The maximum number of user interactions required for an item.
                If an item has user interactions greater than `max_user_interaction` will be ignored.
        """
        super().__init__(**kwargs)

        self._min_item_interaction = min_item_interaction
        self._max_item_interaction = max_item_interaction
        self._min_user_interaction = min_user_interaction
        self._max_user_interaction = max_user_interaction
        self._random_seed = random_seed
        self._target_interaction = target_interaction
        self._input_target_interaction_col = input_target_interaction_col
        self._input_user_id_col = input_user_id_col
        self._input_item_id_col = input_item_id_col

    def _read_data(self, spark):
        logger.info(f"Read files from {self._input_files}")
        dframe = spark.read.parquet(*self._input_files)
        dframe.printSchema()
        return self._common_preprocess(dframe)

    @final
    def _common_preprocess(self, dframe: sql.DataFrame) -> sql.DataFrame:
        """Preprocess data common to train and test data."""

        logger.info("Preprocessing data...")

        # rename columns to unify the names
        dframe = self._rename_input_cols(dframe)
        dframe = self._filter_data(dframe)

        dframe.printSchema()
        return dframe.distinct().persist()

    @final
    def _rename_input_cols(self, dframe: sql.DataFrame) -> sql.DataFrame:
        """Rename input columns to the standard names."""

        logger.info("Renaming columns...")
        # rename columns to unify the names
        dframe = self._rename_item_id_col(dframe)
        return dframe.withColumnRenamed(self._input_user_id_col, self._user_id_col).withColumnRenamed(
            self._input_target_interaction_col, self._target_interaction_col
        )

    def _rename_item_id_col(self, dframe: sql.DataFrame) -> sql.DataFrame:
        """Rename item id column to the standard name."""

        logger.info("Adding item column...")
        return dframe.withColumn(self._item_id_col, F.col(self._input_item_id_col))

    def _filter_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        """Filter data by the given conditions."""

        logger.info("Filtering data...")
        dframe = dframe.filter(F.lower(F.col(self._target_interaction_col)) == self._target_interaction.lower())

        window_userid_count = sql.Window.partitionBy(self._user_id_col)
        window_itemid_count = sql.Window.partitionBy(self._item_id_col)

        dframe = (
            dframe.withColumn("_user_count", F.count(self._user_id_col).over(window_userid_count))
            .withColumn("_item_count", F.count(self._item_id_col).over(window_itemid_count))
            .filter(F.col("_user_count").between(self._min_item_interaction, self._max_item_interaction))
            .filter(F.col("_item_count").between(self._min_user_interaction, self._max_user_interaction))
        )

        return dframe
