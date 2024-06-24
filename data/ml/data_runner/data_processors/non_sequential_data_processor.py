import gin
from pyspark import ml
from pyspark import sql
from pyspark.ml import feature
from pyspark.sql import functions as F

from data.pylib.constant import recsys as common
from data.ml.data_runner.data_processors import data_processor
from data.ml.data_runner.data_processors import spark_utils

_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_USER_INPUTS_COL = common.USER_INPUTS_COL
_ITEM_INPUTS_COL = common.ITEM_INPUTS_COL


@gin.configurable
class AutoEncoderDataProcessor(data_processor.InteractionDataProcessor):
    """data processor for Auto-Encoder.

    The expected data format is as follows (including headers):
    +--------------------+------------+
    |  user_id | item_id | event_name |
    +--------------------+------------+
    |    1234  |  45423  |   read     |
    |    1234  |  3422   |   purchase |
    |    1234  |  83671  |   open     |
    +--------------------+------------+

    The resulting data format is as follows:
    Each row corresponds to a user's item-interaction history.
    +-----------------------------------------------------------+
    |  inputs                  |  targets                       |
    +-----------------------------------------------------------+
    |  [45423, 3422, 83671]    |  [45423, 3422, 83671]          |
    +-----------------------------------------------------------+
    """

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
        self._item_id_idx_col = f"{self._item_id_col}_idx"

    def _generate_train_dev_data(self, dframe: sql.DataFrame) -> data_processor.TrainDevData:
        item_indexer = feature.StringIndexer(inputCols=[self._item_id_col], outputCols=[self._item_id_idx_col], handleInvalid="skip")
        indexer = ml.Pipeline(stages=[item_indexer])

        preprocessor = indexer.fit(dframe)

        dframe = (
            preprocessor.transform(dframe)
            .dropna()
            .withColumn(self._item_id_idx_col, F.col(self._item_id_idx_col).cast("int"))
            .groupby(self._user_id_col)
            .agg(F.collect_list(self._item_id_idx_col).alias("item_ids"))
            .withColumnRenamed("item_ids", _INPUTS_COL)
            .select(_INPUTS_COL)
        )

        # Split train/dev datasets.
        dframe_train, dframe_dev = spark_utils.split_randomly(dframe, _INPUTS_COL, 1 - self._dev_split_size, self._random_seed)
        dframe_train = dframe_train.withColumn(_TARGETS_COL, F.col(_INPUTS_COL))

        # For dev datasets, extract holdout targets.
        dframe_dev = dframe_dev.select(
            F.slice(
                _INPUTS_COL,
                F.lit(1),
                F.floor(F.size(_INPUTS_COL) * (1 - self._holdout_proportion)),
            ).alias(_INPUTS_COL),
            F.slice(
                _INPUTS_COL,
                F.floor(F.size(_INPUTS_COL) * (1 - self._holdout_proportion) + 1),
                F.ceil(F.size(_INPUTS_COL) * self._holdout_proportion),
            ).alias(_TARGETS_COL),
        )

        return data_processor.TrainDevData(preprocessor, data_processor.DataSplits(train=dframe_train, dev=dframe_dev))

    def _generate_test_data(self, dframe: sql.DataFrame) -> data_processor.DataSplits:
        preprocessor = self.load_preprocessor()

        dframe = (
            preprocessor.transform(dframe)
            .dropna()
            .withColumn(self._item_id_idx_col, F.col(self._item_id_idx_col).cast("int"))
            .groupby(self._user_id_col)
            .agg(F.collect_list(self._item_id_idx_col).alias("item_ids"))
            .withColumnRenamed("item_ids", _INPUTS_COL)
            .select(self._user_id_col, _INPUTS_COL)
        )

        return data_processor.DataSplits(test=dframe)


@gin.configurable
class KeywordAutoEncoderDataProcessor(AutoEncoderDataProcessor):
    """Keyword data processor for Auto-Encoder.

    The expected data format is as follows (including headers):
    +-------+----------+----------+------------------------+
    |user_id| timestamp|event_name|         item           |
    +-------+----------+----------+------------------------+
    |1029394|1671355496|  purchase| {945058569, 블랙 앤 ... |
    |1029394|1671354875|  read    | {2971000001, 연애수 ... |
    |1029394|1671364995|  read    | {3049005676, 무향의 ... |
    |1029394|1671364771|  open    | {3200000003, 퍼펙트 ... |
    |1029535|1671367903|  saved   | {3302003565, 산해진 ... |
    |1032680|1671363702|  purchase| {2107097460, 주인님 ... |
    +-------+----------+----------+------------------------

    Item field contains following values.
    |-- item: struct (nullable = true)
    |    |-- id: string (nullable = true)
    |    |-- tags: array<long> (nullable = true)
    |    |-- ...

    The resulting data format is as follows:
    Each row corresponds to a user's interaction history.
    +-----------------------------------------------------------+
    |  inputs                  |  targets                       |
    +-----------------------------------------------------------+
    |  [45423, 3422, 83671]    |  [45423, 3422, 83671]          |
    +-----------------------------------------------------------+
    """

    _IGNORE_TAG_IDS = [
        # 리뷰 n개 이상
        2700,
        2792,
        3673,
        3675,
        3676,
        3781,
        3782,
        # 별점 n개 이상
        3426,
        3427,
        3428,
        3429,
        3430,
        3431,
        3674,
        3677,
        3678,
        3783,
        3784,
        # 평점 n 이상
        2699,
        2791,
        3780,
        3770,  # 리다무
        3773,  # 웹소설
        3772,  # e북
        3771,  # 대여
        # 연재중/완결
        1373,
        1392,
        1417,
        3051,
        3052,
        3053,
        3054,
        3113,
        3114,
        3623,
        3624,
        3774,
        3775,
        3769,  # RIDI-ONLY
        # n권 이상,
        2795,
        3785,
        3786,
    ]

    def _rename_item_id_col(self, dframe: sql.DataFrame) -> sql.DataFrame:
        # TODO: explode tags before filtering in etl job and not override this method.
        return dframe.withColumn(self._item_id_col, F.explode(self._input_item_id_col))

    def _filter_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        dframe = dframe.selectExpr(self._user_id_col, self._target_interaction_col, self._item_id_col).distinct()
        dframe = dframe.filter(~F.col(self._item_id_col).isin(self._IGNORE_TAG_IDS))
        return super()._filter_data(dframe)


@gin.configurable
class ClassificationDataProcessor(data_processor.InteractionDataProcessor):
    """data processor for Classification.

    This data processor includes negative samples and the number of
    negative samples per positive can be controlled with `negative_sample_ratio`.

    The expected data format is as follows (including headers):
    +---------------------------------------------+
    |  user_id | item_id | timestamp | event_name |
    +---------------------------------------------+
    |    1234  |  45423  |  1235411  |  purchase  |
    |    1234  |  3422   |  1235411  |  read      |
    |    1234  |  83671  |  1235411  |  open      |
    +---------------------------------------------+

    The resulting data format is as follows:
    Each row corresponds to a user's item-interaction history.
    +-----------------------------------------------------------+
    |  user_inputs       |  item_inputs   |    targets          |
    +-----------------------------------------------------------+
    |  45423             |  3422          |        1            |
    |  45423             |  87631         |        0            |
    |  542               |  1221          |        1            |
    +-----------------------------------------------------------+
    """

    def __init__(
        self,
        negative_sample_ratio: float = 1.0,
        shuffle_items: bool = True,
        **kwargs,
    ):
        """Constructs `ClassificationDataProcessor`.

        Args:
            negative_sample_ratio:
            shuffle_items: If True, shuffle negative candidate samples for a user before sampling.

        """

        super().__init__(**kwargs)
        self._negative_sample_ratio = negative_sample_ratio
        self._shuffle_items = shuffle_items
        self._item_id_idx_col = f"{self._item_id_col}_idx"
        self._user_id_idx_col = f"{self._user_id_col}_idx"

    def _generate_train_dev_data(self, dframe):
        item_user_indexer = feature.StringIndexer(
            inputCols=[self._item_id_col, self._user_id_col],
            outputCols=[self._item_id_idx_col, self._user_id_idx_col],
            handleInvalid="skip",
        )
        indexer = ml.Pipeline(stages=[item_user_indexer])
        preprocessor = indexer.fit(dframe)

        # Split train/dev datasets.
        dframe_train, dframe_dev = spark_utils.split_chronologically(
            dframe, self._user_id_col, "timestamp", 1 - self._dev_split_size, self._random_seed
        )

        # NOTE: meta.item_ids can be replaced with [0 ... len(item_ids)],
        #       if preprocessor is applied before.
        negative_sampling = spark_utils.negative_sampling(
            self._user_id_col,
            self._item_id_col,
            _TARGETS_COL,
            list(map(int, preprocessor.stages[-1].labelsArray[0])),
            self._negative_sample_ratio,
            self._shuffle_items,
            self._random_seed,
        )

        cols = [
            F.col(self._user_id_idx_col).cast("int").alias(_USER_INPUTS_COL),
            F.col(self._item_id_idx_col).cast("int").alias(_ITEM_INPUTS_COL),
            _TARGETS_COL,
        ]

        dframe_train = dframe_train.select(self._user_id_col, self._item_id_col).transform(negative_sampling)
        dframe_train = preprocessor.transform(dframe_train).select(*cols)

        dframe_dev = dframe_dev.select(self._user_id_col, self._item_id_col).transform(negative_sampling)
        dframe_dev = preprocessor.transform(dframe_dev).select(*cols)

        return data_processor.TrainDevData(preprocessor, data_processor.DataSplits(train=dframe_train, dev=dframe_dev))

    def _generate_test_data(self, dframe):
        preprocessor = self.load_preprocessor()

        dframe = (
            preprocessor.transform(dframe)
            .dropna()
            .select(
                F.col(self._user_id_idx_col).cast("int").alias(_USER_INPUTS_COL),
                F.col(self._item_id_idx_col).cast("int").alias(_ITEM_INPUTS_COL),
            )
        )

        return data_processor.DataSplits(test=dframe)
