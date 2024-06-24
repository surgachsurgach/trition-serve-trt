""" Data Processor V2 module."""
import abc
import dataclasses
import functools
import os
from typing import Any, final

from loguru import logger
from pyspark import ml
from pyspark import sql
from pyspark.ml import feature
from pyspark.sql import functions as F

from data.pylib.constant import recsys as common
from data.ml.data_runner.data_processors import configs
from data.ml.data_runner.data_processors import spark_utils
from data.ml.utils import file_utils
from data.ml.utils import metadata

_USER_ID_COL = common.USER_ID_COL
_ITEM_ID_COL = common.ITEM_ID_COL
_WEIGHT_COL = common.WEIGHT_COL
_TARGET_INTERACTION_COL = common.TARGET_INTERACTION_COL
_TIMESTAMP_COL = common.TIMESTAMP_COL

_TRAIN_DATASET = common.TRAIN_DATASET_NAME
_DEV_DATASET = common.DEV_DATASET_NAME
_TEST_DATASET = common.TEST_DATASET_NAME

_PREPROCESSOR = common.PREPROCESSOR_NAME

_TRAIN_META_FILENAME = common.TRAIN_META_FILENAME
_TEST_META_FILENAME = common.TEST_META_FILENAME


@dataclasses.dataclass
class DataSplits:
    train: sql.DataFrame | None = dataclasses.field(default=None)
    dev: sql.DataFrame | None = dataclasses.field(default=None)
    test: sql.DataFrame | None = dataclasses.field(default=None)


def _ensure_list(value: Any) -> list:
    if not isinstance(value, list):
        return [value]
    return value


def _mkdir_force(dirname: str) -> str:
    fs = file_utils.get_filesystem(dirname)
    fs.makedirs(dirname, exist_ok=True)
    return dirname


class DataProcessor(abc.ABC):
    """Abstract class for data processors."""

    _train_meta_filename = _TRAIN_META_FILENAME
    _test_meta_filename = _TEST_META_FILENAME

    def __init__(self, config: configs.DataProcessConfig):
        """Constructs DataProcessor instance.

        DataProcessor reads data files from local or S3.
        Subclasses of DataProcessor should implement `_read_data` and `_generate_data` methods.
        `_read_data` is responsible for reading raw data files from local or S3 in its proper data format such as
        CSV and parquets. `_read_data` should return the raw data as Spark DataFrame.
        `_generate_data` converts a plain DataFrame into processed DataFrame. For example,
        it can filter out unnecessary rows.

        Args:
            config: DataProcessConfig.
        """
        self._config = config
        self._meta = metadata.Meta()
        self._is_test_split = None

    @property
    def input_files(self):
        return _ensure_list(self._config.input_files)

    @property
    def output_path(self):
        return _mkdir_force(self._config.output_path)

    @functools.cached_property
    def _spark_session(self):
        return spark_utils.create_spark(
            self.__class__.__name__,
            self._config.run_locally,
            file_utils.get_scheme(self.input_files[0]),
        )

    @functools.cached_property
    def _num_executors(self):
        return int(self._spark_session.conf.get("spark.executor.instances", "1"))

    @abc.abstractmethod
    def _read_data(self, spark: sql.SparkSession) -> sql.DataFrame:
        """Reads raw data and returns Spark DataFrame."""

    @abc.abstractmethod
    def _filter_train_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        """Filters train data from the raw data."""

    @abc.abstractmethod
    def _filter_test_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        """Filters test data from the raw data."""

    @abc.abstractmethod
    def _generate_train_dev_data(self, dframe: sql.DataFrame) -> DataSplits:
        """Reads raw data files. Returns train_dev data and item metadata."""

    @abc.abstractmethod
    def _generate_test_data(self, dframe: sql.DataFrame) -> DataSplits:
        """Reads raw data files and returns test data."""

    def write_data(self, is_test_split=False):
        """Writes processed data in parquet files.

        Args:
            is_test_split: If True, generate data for test.
        """
        self._is_test_split = is_test_split
        self._meta.reset()
        logger.info(f"Writing data to {self.output_path}.")
        dframe = self._read_data(self._spark_session)

        if not self._is_test_split:
            dframe = self._filter_train_data(dframe)
            data_splits = self._generate_train_dev_data(dframe)
            self._save_dataset(data_splits.train, _TRAIN_DATASET, self._config.train_num_partitions)

            if data_splits.dev:
                self._save_dataset(data_splits.dev, _DEV_DATASET, self._config.dev_num_partitions)
            self._meta.save(self.output_path, self._train_meta_filename)
        else:
            dframe = self._filter_test_data(dframe)
            data_splits = self._generate_test_data(dframe)
            self._save_dataset(data_splits.test, _TEST_DATASET, self._config.test_num_partitions)
            self._meta.save(self.output_path, self._test_meta_filename)

    def _add_meta_from_preprocessor(self, preprocessor: ml.PipelineModel):
        """Add metadata from preprocessor.

        `id <-> idx` mapping is stored in the first stage of the pipeline.
        It is necessary to save mapping information to convert idx to id in the inference phase.
        """
        logger.info("Generating metadata...")
        first_stage = preprocessor.stages[0]
        if isinstance(first_stage, feature.StringIndexerModel):
            for i, col in enumerate(first_stage.getInputCols()):

                if col == _USER_ID_COL:
                    self._meta.add_meta(col, list(map(int, first_stage.labelsArray[i])))
                else:
                    self._meta.add_meta(col, list(first_stage.labelsArray[i]))
        else:
            raise RuntimeError("First stage of pipeline must be StringIndexerModel.")

    def _save_preprocessor(self, preprocessor: ml.PipelineModel, name: str = _PREPROCESSOR):
        self._add_meta_from_preprocessor(preprocessor)
        preprocessor.write().overwrite().save(os.path.join(self.output_path, name))

    def _load_preprocessor(self, name: str = _PREPROCESSOR) -> ml.PipelineModel:
        return ml.PipelineModel.load(os.path.join(self.output_path, name))

    def _save_dataset(self, data: sql.DataFrame, name: str, num_partitions: int):
        data = data.repartition(num_partitions)
        data.write.mode("overwrite").parquet(os.path.join(self.output_path, name))

    def _load_meta(self, name: str) -> metadata.Meta:
        return metadata.Meta.load(self.output_path, name)


class InteractionDataProcessor(DataProcessor, abc.ABC):
    _user_id_col = _USER_ID_COL
    _item_id_col = _ITEM_ID_COL
    _target_interaction_col = _TARGET_INTERACTION_COL
    _timestamp_col = _TIMESTAMP_COL
    _weight_col = _WEIGHT_COL

    def __init__(
        self,
        schema_config: configs.InteractionDataSchemaConfig,
        train_process_config: configs.InteractionDataProcessConfig,
        test_process_config: configs.InteractionDataProcessConfig,
        random_seed: int | None = None,
        **kwargs,
    ):
        """Constructs `InteractionDataProcessor`.

        Args:
            schema_config: Interaction data schema config.
            train_process_config: Train interaction data process config.
            test_process_config: Test interaction data process config.
            random_seed: Random seed.
            **kwargs: Other keyword arguments.
        """
        super().__init__(**kwargs)
        self._schema_config = schema_config
        self._train_config = train_process_config
        self._test_config = test_process_config

        self._random_seed = random_seed

    def _read_data(self, spark: sql.SparkSession) -> sql.DataFrame:
        logger.info(f"Read files from {self.input_files}")
        dframe = spark.read.parquet(*self.input_files)
        dframe = self._common_preprocess(dframe)
        dframe.printSchema()
        return dframe

    @final
    def _common_preprocess(self, dframe: sql.DataFrame) -> sql.DataFrame:
        """Preprocess data common to train and test data."""

        logger.info("Preprocessing data...")

        dframe = self._rename_input_cols(dframe)
        dframe.printSchema()
        return dframe.distinct().persist()

    @final
    def _rename_input_cols(self, dframe: sql.DataFrame) -> sql.DataFrame:
        """Rename input columns to the unified names."""
        logger.info("Renaming columns...")
        dframe = (
            dframe.withColumnRenamed(self._schema_config.input_item_id_col, self._item_id_col)
            .withColumnRenamed(self._schema_config.input_user_id_col, self._user_id_col)
            .withColumnRenamed(self._schema_config.input_target_interaction_col, self._target_interaction_col)
        )
        if self._schema_config.input_weight_col:
            dframe = dframe.withColumnRenamed(self._schema_config.input_weight_col, self._weight_col)
        else:
            dframe = dframe.withColumn(self._weight_col, F.lit(1.0))
        return dframe

    def _filter_by_interaction_types(self, dframe: sql.DataFrame, target_interactions: list | None) -> sql.DataFrame:
        """Filters out rows whose interaction type is not in the target interaction types."""
        logger.info(f"Target interactions: {target_interactions}")
        logger.info("Filtering by interaction types...")
        if target_interactions:
            dframe = dframe.where(F.col(self._target_interaction_col).isin(target_interactions))
        return dframe

    def _filter_by_interaction_range(
        self,
        dframe: sql.DataFrame,
        config: configs.InteractionDataProcessConfig,
    ) -> sql.DataFrame:
        """Filters out rows whose number of interactions is not in the interaction range."""
        inputs = [
            (self._user_id_col, self._item_id_col, config.user_interaction_range),
            (self._item_id_col, self._user_id_col, config.item_interaction_range),
        ]
        if config.item_filter_first:
            inputs.reverse()

        for subject_col, object_col, interaction_range in inputs:
            window = sql.Window.partitionBy(subject_col)
            dframe = (
                dframe.withColumn("_cnt", F.count(object_col).over(window))
                .filter(F.col("_cnt").between(interaction_range.min, interaction_range.max))
                .drop("_cnt")
            )
        return dframe

    def _sync_test_to_train(self, dframe: sql.DataFrame) -> sql.DataFrame:
        """Filters out rows whose user or item is not in the training data.

        If there is a column in the key values of the meta,
        it means that the id of the column was indexed through preprocessing when creating the training data,
        so it is necessary to synchronize it for proper inference.
        """
        train_meta = self._load_meta(self._train_meta_filename)
        if self._user_id_col in train_meta.metadata:
            users = train_meta.get_meta(self._user_id_col, int)
            dframe = dframe.where(F.col(self._user_id_col).isin(users))
        if self._item_id_col in train_meta.metadata:
            items = train_meta.get_meta(self._item_id_col, str)
            dframe = dframe.where(F.col(self._item_id_col).isin(items))
        return dframe

    def _filter_train_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        dframe = self._filter_by_interaction_types(dframe, self._train_config.target_interactions)
        dframe = self._filter_by_interaction_range(dframe, self._train_config)
        self._add_interaction_metadata(dframe)
        return dframe

    def _filter_test_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        dframe = self._filter_by_interaction_types(dframe, self._test_config.target_interactions)
        dframe = self._sync_test_to_train(dframe)
        dframe = self._filter_by_interaction_range(dframe, self._test_config)
        self._add_interaction_metadata(dframe)
        return dframe

    def _add_interaction_metadata(self, dframe: sql.DataFrame):
        """Add interaction metadata to monitor the coverage of recommendation model.

        For hyperparameter tuning, it is important to monitor the coverage of recommendation model.

        Metadata:
            - num_users: The number of users in the dataset. This means the user coverage of the recommendation model.
            - num_items: The number of items in the dataset. This means the item coverage of the recommendation model.
            - num_interactions: The number of interactions in the dataset.
        """

        interaction_metadata = {
            "num_users": dframe.select(self._user_id_col).distinct().count(),
            "num_items": dframe.select(self._item_id_col).distinct().count(),
            "num_interactions": dframe.count(),
        }

        for key, value in interaction_metadata.items():
            self._meta.add_meta(key, value)

    @staticmethod
    def _convert_key_to_idx(
        dframe: sql.DataFrame,
        key_col: str,
        idx_col: str,
        preprocessor: ml.PipelineModel | None = None,
    ) -> tuple[sql.DataFrame, ml.PipelineModel]:
        """Converts key to index.

        For ML/DL models, it is necessary to use index instead.
        """
        if preprocessor is None:
            # If preprocessor is not given, generate it.
            item_indexer = feature.StringIndexer(inputCols=[key_col], outputCols=[idx_col], handleInvalid="skip")
            indexer = ml.Pipeline(stages=[item_indexer])

            preprocessor = indexer.fit(dframe)

        dframe = preprocessor.transform(dframe)
        dframe = dframe.withColumn(idx_col, F.col(idx_col).cast("int"))
        return dframe, preprocessor
