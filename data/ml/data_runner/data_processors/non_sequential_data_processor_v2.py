import gin
from loguru import logger
from pyspark import ml
from pyspark import sql
from pyspark.sql import functions as F

from data.ml.data_runner.data_processors import configs
from data.ml.data_runner.data_processors import data_processor_v2 as data_processor
from data.ml.data_runner.data_processors import spark_utils
from data.pylib.constant import recsys as common

_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_WEIGHT_OF_INPUTS_COL = common.INPUT_WEIGHTS_COL
_WEIGHT_OF_TARGETS_COL = common.TARGET_WEIGHTS_COL

_ITEM_IDX_COL = common.ITEM_IDX_COL

_USER_INPUTS_COL = common.USER_INPUTS_COL
_ITEM_INPUTS_COL = common.ITEM_INPUTS_COL
_POSITIVE_CONTEXTS_COL = common.POSITIVE_CONTEXTS_COL
_NEGATIVE_CONTEXTS_COL = common.NEGATIVE_CONTEXTS_COL

_USER_PREPROCESSOR_NAME = common.USER_PREPROCESSOR_NAME
_ITEM_PREPROCESSOR_NAME = common.ITEM_PREPROCESSOR_NAME

_GENDER_COL = common.GENDER_COL
_GENDER_IDX_COL = common.GENDER_IDX_COL
_GENDER_PREPROCESSOR_NAME = common.GENDER_PREPROCESSOR_NAME
_GENERATION_COL = common.GENERATION_COL
_GENERATION_IDX_COL = common.GENERATION_IDX_COL
_GENERATION_PREPROCESSOR_NAME = common.GENERATION_PREPROCESSOR_NAME

_ITEM_FREQUENCY_META_KEY = common.ITEM_FREQUENCY_META_KEY


def _ensure_list(value) -> list:
    if not isinstance(value, list):
        return [value]
    return value


class AggregateInteractionMixin:
    """Aggregate interaction data.

    Have utility functions for handling interaction data.
    """

    _tmp_indices_col = "_tmp_indices"
    _tmp_weights_col = "_tmp_weights"

    def _agg_idx_to_array(self, dframe: sql.DataFrame, group_by_col: str | sql.Column, idx_col: str | sql.Column) -> sql.DataFrame:
        dframe = dframe.groupBy(group_by_col).agg(F.collect_list(idx_col).alias(self._tmp_indices_col))
        return dframe

    def _agg_idx_and_weight_to_array(
        self, dframe: sql.DataFrame, group_by_col: str | sql.Column, idx_col: str | sql.Column, weight_col: str | sql.Column
    ) -> sql.DataFrame:
        dframe = dframe.groupBy(group_by_col).agg(
            F.collect_list(idx_col).alias(self._tmp_indices_col),
            F.collect_list(weight_col).alias(self._tmp_weights_col),
        )
        return dframe


@gin.configurable
class UserAutoEncoderDataProcessor(AggregateInteractionMixin, data_processor.InteractionDataProcessor):
    """Data processor for user autoencoder.

    To prevent the complexity of the implementation and user confusion,
    the use was limited to Auto Encoder for User Vector.

    The expected data format is as follows (including headers):
    +----------------------+--------+------------+
    |  user_id |  item_id  | weight | event_name |
    +----------------------+--------+------------+
    |    1234  |  '45423'  |  1.0   |   read     |
    |    1234  |  '3422'   |  1.2   |   purchase |
    |    1234  |  '83671'  |  0.7   |   open     |
    +----------------------+--------+------------+

    The resulting data format is as follows:
    Each row corresponds to a user's item-interaction history.
    ( item_id -> item_idx : (45423 -> 1), (3422 -> 2), (83671 -> 3) )
    +-------------+-------------+-------------+--------------------+--------------------+
    |  user_id    |  inputs     |  targets    |  weight_of_inputs  |  weight_of_targets |
    +-------------+-------------+-------------+--------------------+--------------------+
    |  1234       |  [1, 2, 3]  |  [1, 2, 3]  |  [1.0, 1.2, 0.7]   |  [1.0, 1.2, 0.7]   |
    +-------------+-------------+-------------+--------------------+--------------------+

    """

    _inputs_col = _INPUTS_COL
    _targets_col = _TARGETS_COL
    _weight_of_inputs_col = _WEIGHT_OF_INPUTS_COL
    _weight_of_targets_col = _WEIGHT_OF_TARGETS_COL
    _item_idx_col = _ITEM_IDX_COL

    _item_preprocessor_name = _ITEM_PREPROCESSOR_NAME

    def __init__(self, holdout_proportion: float = 0.2, **kwargs):
        """Constructs `AutoEncoderDataProcessor`.

        Args:
            holdout_proportion:
                The proportion of test targets. This is for the development datasets.
                Among the input data, (1-holdout_proportion) data are used as inputs for validation and
                holdout_proportion data are used as targets.
        """
        super().__init__(**kwargs)

        self._holdout_proportion = holdout_proportion

    def _split_by_holdout_proportion(self, split_col: str) -> tuple[sql.Column, sql.Column]:
        turning_point = F.floor(F.size(split_col) * F.lit(1 - self._holdout_proportion))
        front = F.slice(split_col, F.lit(1), turning_point)
        back = F.slice(split_col, turning_point + 1, F.size(split_col) - turning_point)
        return front, back

    def _get_user_item_interaction(self, dframe: sql.DataFrame, preprocessor: ml.PipelineModel | None = None) -> sql.DataFrame:
        if preprocessor is None:
            dframe, preprocessor = self._convert_key_to_idx(dframe, self._item_id_col, self._item_idx_col)
            self._save_preprocessor(preprocessor, self._item_preprocessor_name)
        else:
            dframe, _ = self._convert_key_to_idx(dframe, self._item_id_col, self._item_idx_col, preprocessor)
        dframe = self._agg_idx_and_weight_to_array(dframe, self._user_id_col, self._item_idx_col, self._weight_col)

        dframe = dframe.select(
            self._user_id_col,
            F.col(self._tmp_indices_col).alias(self._inputs_col),
            F.col(self._tmp_weights_col).alias(self._weight_of_inputs_col),
        )
        return dframe

    def _get_train_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        # For train dataset, inputs and targets are the same.
        dframe = dframe.withColumn(self._targets_col, F.col(self._inputs_col))
        dframe = dframe.withColumn(self._weight_of_targets_col, F.col(self._weight_of_inputs_col))
        return dframe

    def _get_dev_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        # For dev dataset, inputs are the first (1-holdout_proportion) data
        # and targets are the last holdout_proportion data.
        dframe = dframe.withColumnRenamed(self._inputs_col, f"{self._inputs_col}_all")
        dframe = dframe.withColumnRenamed(self._weight_of_inputs_col, f"{self._weight_of_inputs_col}_all")

        dev_inputs, dev_targets = self._split_by_holdout_proportion(f"{self._inputs_col}_all")
        dev_weights_of_inputs, dev_weights_of_targets = self._split_by_holdout_proportion(f"{self._weight_of_inputs_col}_all")
        dframe = (
            dframe.withColumn(self._inputs_col, dev_inputs)
            .withColumn(self._targets_col, dev_targets)
            .withColumn(self._weight_of_inputs_col, dev_weights_of_inputs)
            .withColumn(self._weight_of_targets_col, dev_weights_of_targets)
            .filter(F.size(self._inputs_col) > 0)
            .filter(F.size(self._targets_col) > 0)
        )
        return dframe

    def _generate_train_dev_data(self, dframe: sql.DataFrame) -> data_processor.DataSplits:
        dframe = self._get_user_item_interaction(dframe)

        # Split train/dev datasets.
        dframe_train, dframe_dev = spark_utils.split_randomly(
            dframe,
            self._inputs_col,
            1 - self._config.dev_split_size,
            self._random_seed,
        )

        dframe_train = self._get_train_data(dframe_train)
        dframe_dev = self._get_dev_data(dframe_dev)

        output_cols = [self._inputs_col, self._targets_col, self._weight_of_inputs_col, self._weight_of_targets_col]

        dframe_train = dframe_train.select(*output_cols)
        dframe_dev = dframe_dev.select(*output_cols)

        return data_processor.DataSplits(train=dframe_train, dev=dframe_dev)

    def _generate_test_data(self, dframe: sql.DataFrame) -> data_processor.DataSplits:
        preprocessor = self._load_preprocessor(self._item_preprocessor_name)
        dframe = self._get_user_item_interaction(dframe, preprocessor)
        return data_processor.DataSplits(test=dframe)


@gin.configurable
class UserConditionalAutoEncoderDataProcessor(UserAutoEncoderDataProcessor):
    _gender_col = _GENDER_COL
    _gender_idx_col = _GENDER_IDX_COL
    _gender_preprocessor_name = _GENDER_PREPROCESSOR_NAME
    _generation_col = _GENERATION_COL
    _generation_idx_col = _GENERATION_IDX_COL
    _generation_preprocessor_name = _GENERATION_PREPROCESSOR_NAME

    def __init__(self, user_data_config: configs.UserConditionalDataConfig, **kwargs):
        super().__init__(**kwargs)

        self._user_data_config = user_data_config

    def _read_user_data(self):
        input_files = _ensure_list(self._user_data_config.input_files)
        logger.info(f"Read User Data files from {input_files}")
        dframe = self._spark_session.read.parquet(*input_files)
        dframe = self._preprocess_user_data(dframe)
        dframe.printSchema()
        return dframe

    def _preprocess_user_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        logger.info("Renaming columns...")
        dframe = (
            dframe.withColumnRenamed(self._user_data_config.input_user_id_col, self._user_id_col)
            .withColumnRenamed(self._user_data_config.input_gender_col, self._gender_col)
            .withColumnRenamed(self._user_data_config.input_generation_col, self._generation_col)
        )
        logger.info("Processing Null values...")
        dframe = dframe.fillna({self._gender_col: "unknown", self._generation_col: "unknown"})
        return dframe

    def _process_user_data(
        self,
        dframe: sql.DataFrame,
        gender_preprocessor: ml.PipelineModel | None = None,
        generation_preprocessor: ml.PipelineModel | None = None,
    ) -> sql.DataFrame:
        if gender_preprocessor is None and generation_preprocessor is None:
            dframe, gender_preprocessor = self._convert_key_to_idx(dframe, self._gender_col, self._gender_idx_col)
            self._save_preprocessor(gender_preprocessor, self._gender_preprocessor_name)

            dframe, generation_preprocessor = self._convert_key_to_idx(dframe, self._generation_col, self._generation_idx_col)
            self._save_preprocessor(generation_preprocessor, self._generation_preprocessor_name)
        elif gender_preprocessor and generation_preprocessor:
            dframe, _ = self._convert_key_to_idx(dframe, self._gender_col, self._gender_idx_col, gender_preprocessor)
            dframe, _ = self._convert_key_to_idx(dframe, self._generation_col, self._generation_idx_col, generation_preprocessor)
        else:
            raise ValueError("Gender and Generation preprocessor should be both None or not None.")

        dframe = dframe.select(
            self._user_id_col,
            self._gender_idx_col,
            self._generation_idx_col,
        )
        return dframe

    def _add_user_data(self, dframe: sql.DataFrame, user_data: sql.DataFrame) -> sql.DataFrame:
        return dframe.join(user_data, self._user_id_col, "inner")

    def _generate_train_dev_data(self, dframe: sql.DataFrame) -> data_processor.DataSplits:
        dframe = self._get_user_item_interaction(dframe)

        # Split train/dev datasets.
        dframe_train, dframe_dev = spark_utils.split_randomly(
            dframe,
            self._inputs_col,
            1 - self._config.dev_split_size,
            self._random_seed,
        )

        user_data = self._read_user_data()
        user_data = self._process_user_data(user_data)
        user_data.persist()

        dframe_train = self._get_train_data(dframe_train)
        dframe_dev = self._get_dev_data(dframe_dev)

        dframe_train = self._add_user_data(dframe_train, user_data)
        dframe_dev = self._add_user_data(dframe_dev, user_data)

        user_data.unpersist()

        output_cols = [
            self._inputs_col,
            self._targets_col,
            self._weight_of_inputs_col,
            self._weight_of_targets_col,
            self._gender_idx_col,
            self._generation_idx_col,
        ]

        dframe_train = dframe_train.select(*output_cols)
        dframe_dev = dframe_dev.select(*output_cols)

        return data_processor.DataSplits(train=dframe_train, dev=dframe_dev)

    def _generate_test_data(self, dframe: sql.DataFrame) -> data_processor.DataSplits:

        item_preprocessor = self._load_preprocessor(self._item_preprocessor_name)
        gender_preprocessor = self._load_preprocessor(self._gender_preprocessor_name)
        generation_preprocessor = self._load_preprocessor(self._generation_preprocessor_name)

        dframe = self._get_user_item_interaction(dframe, item_preprocessor)

        user_data = self._read_user_data()
        user_data = self._process_user_data(user_data, gender_preprocessor, generation_preprocessor)
        user_data.persist()

        test_data = self._add_user_data(dframe, user_data)
        return data_processor.DataSplits(test=test_data)


@gin.configurable
class ItemEmbeddingDataProcessor(AggregateInteractionMixin, data_processor.InteractionDataProcessor):
    """Data processor for item embedding.

    To prevent the complexity of the implementation and user confusion,
    the use was limited to Auto Encoder for User Vector.

    The expected data format is as follows (including headers):
    +----------------------+--------+------------+
    |  user_id |  item_id  | weight | event_name |
    +----------------------+--------+------------+
    |    1234  |  '45423'  |  1.0   |   read     |
    |    1234  |  '3422'   |  1.2   |   purchase |
    |    1235  |  '3422'   |  0.7   |   open     |
    +----------------------+--------+------------+

    The resulting data format is as follows:
    Each row corresponds to a item sequence of a user.
    ( item_id -> item_idx : (45423 -> 1), (3422 -> 2) )
    +----------+
    |  inputs  |
    +----------+
    |  [1, 2]  |
    +----------+
    |  [2]     |
    +----------+

    """

    _inputs_col = _INPUTS_COL
    _weight_of_inputs_col = _WEIGHT_OF_INPUTS_COL
    _item_idx_col = _ITEM_IDX_COL

    _item_preprocessor_name = _ITEM_PREPROCESSOR_NAME
    _item_frequency_meta_key = _ITEM_FREQUENCY_META_KEY

    def _get_item_frequency(self, dframe: sql.DataFrame) -> list[int]:
        """Get frequency of item."""
        dframe = dframe.groupBy(self._item_idx_col).agg(F.count(self._user_id_col).alias("count"))
        dframe = dframe.orderBy(self._item_idx_col).select(F.col("count").alias("item_frequency"))
        return [row.item_frequency for row in dframe.collect()]

    def _generate_train_dev_data(self, dframe: sql.DataFrame) -> data_processor.DataSplits:
        dframe, item_preprocessor = self._convert_key_to_idx(dframe, self._item_id_col, self._item_idx_col)
        self._save_preprocessor(item_preprocessor, self._item_preprocessor_name)

        item_frequency = self._get_item_frequency(dframe)
        self._meta.add_meta(self._item_frequency_meta_key, item_frequency)

        dframe = self._agg_idx_to_array(dframe, self._user_id_col, self._item_idx_col)
        dframe = dframe.select(F.col(self._tmp_indices_col).alias(self._inputs_col))

        if self._config.dev_split_size > 0:
            dframe_train, dframe_dev = spark_utils.split_randomly(
                dframe,
                self._inputs_col,
                1 - self._config.dev_split_size,
                self._random_seed,
            )
        else:
            dframe_train = dframe
            dframe_dev = None
        return data_processor.DataSplits(train=dframe_train, dev=dframe_dev)

    def _generate_test_data(self, dframe: sql.DataFrame) -> data_processor.DataSplits:
        item_preprocessor = self._load_preprocessor(self._item_preprocessor_name)

        dframe, _ = self._convert_key_to_idx(dframe, self._item_id_col, self._item_idx_col, item_preprocessor)

        dframe = dframe.select(self._item_idx_col).distinct()
        return data_processor.DataSplits(test=dframe)
