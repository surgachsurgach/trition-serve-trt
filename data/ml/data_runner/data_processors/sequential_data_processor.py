from typing import List

import gin
from pyspark import ml
from pyspark import sql
from pyspark.ml import feature
from pyspark.sql import functions as F

from data.ml.data_runner.data_processors import data_processor
from data.ml.data_runner.data_processors import spark_utils
from data.ml.utils import metadata


@gin.configurable
class SequentialDataProcessor(data_processor.InteractionDataProcessor):
    """Sequence data processor.

    The expected data format is as follows (including headers):
    +-------+----------+----------+------------------------+----------+
    |  u_idx| timestamp|event_name|                    item|paid_price|
    +-------+----------+----------+------------------------+----------+
    |1029394|1671355496|  purchase| {945058569, 블랙 앤 ... |         0|
    |1029394|1671354875|  purchase| {2971000001, 연애수 ... |         0|
    |1029394|1671364995|  purchase| {3049005676, 무향의 ... |         0|
    |1029394|1671364771|  purchase| {3200000003, 퍼펙트 ... |         0|
    |1029535|1671367903|  purchase| {3302003565, 산해진 ... |         0|
    |1032680|1671363702|  purchase| {2107097460, 주인님 ... |         0|
    +-------+----------+----------+------------------------+----------+

    Item field contains following values.
    |-- item: struct (nullable = true)
    |    |-- id: string (nullable = true)
    |    |-- title: string (nullable = true)
    |    |-- pubdate: long (nullable = true)
    |    |-- publisher_name: string (nullable = true)
    |    |-- author: string (nullable = true)
    |    |-- genre: string (nullable = true)
    |    |-- category: integer (nullable = true)
    |    |-- price: integer (nullable = true)
    |    |-- age_limit: integer (nullable = true)
    |    |-- is_setbook: boolean (nullable = true)

    The resulting data format is as follows:
    Each row corresponds to a user's item-interaction sequential history.
    +-------+--------------------+--------------------+------------------+--------------------+
    |  u_idx|           timestamp|            item_id |    item_category |        item_author |
    +-------+--------------------+--------------------+------------------+--------------------+
    |1234556|[1671347739, 1671...|             [6, 56]|            [4, 9]|            [16, 54]|
    |2345612|[1671401930, 1671...|  [31, 41, 52, 5, 9]|   [0, 0, 0, 2, 0]| [52, 25, 29, 12, 7]|
    |12351sd|[1671393917, 1671...|  [1, 4, 17, 33, 59]|   [0, 0, 2, 0, 0]|  [1, 3, 17, 33, 48]|
    |123azx1|[1671357791, 1671...|          [0, 1, 19]|         [0, 0, 0]|          [0, 1, 35]|
    +-------+--------------------+--------------------+------------------+--------------------+

    """

    def __init__(self, side_features: List[str], **kwargs):
        super().__init__(**kwargs)
        self._side_feature_elements = side_features
        self._side_feature_cols = [f"item_{col}" for col in self._side_feature_elements]
        self._side_feature_idx_cols = [f"{col}_idx" for col in self._side_feature_cols]
        self._item_feature_cols = [self._item_id_col] + self._side_feature_cols
        self._item_feature_idx_cols = [f"{col}_idx" for col in self._item_feature_cols]

    def _rename_item_id_col(self, dframe: sql.DataFrame) -> sql.DataFrame:
        dframe = super()._rename_item_id_col(dframe)
        return dframe.withColumn(
            self._item_id_col, F.concat_ws(metadata.ITEM_ID_PREFIX_DELIM, F.col("item.genre"), F.col(self._item_id_col))
        )

    def _filter_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        dframe = dframe.filter(dframe[self._user_id_col].isNotNull()).filter(dframe[self._user_id_col] != 0)
        return super()._filter_data(dframe)

    def _unnest_side_features(self, dframe: sql.DataFrame) -> sql.DataFrame:
        for element, col in zip(self._side_feature_elements, self._side_feature_cols):
            dframe = dframe.withColumn(col, F.col(f"item.{element}"))
        return dframe

    def _build_data(self, dframe: sql.DataFrame, preprocessor: ml.PipelineModel) -> sql.DataFrame:
        dframe = preprocessor.transform(dframe)

        cols = [F.col("timestamp")] + [F.col(col).cast("int").alias(col) for col in self._item_feature_idx_cols]

        # Aggregate and sort in the ascending order of timestamp.
        dframe = (
            dframe.withColumn("indexed_item", F.struct(*cols))
            .groupBy(self._user_id_col)
            .agg(F.array_sort(F.collect_list("indexed_item")).alias("items"))
        )

        for col in ["timestamp"] + self._item_feature_idx_cols:
            dframe = dframe.withColumn(col, F.transform("items", lambda x: x[col]))  # pylint: disable=cell-var-from-loop

        for col, rename_col in zip(self._item_feature_idx_cols, self._item_feature_cols):
            dframe = dframe.withColumnRenamed(col, rename_col)

        return dframe.drop("items")

    def _generate_train_dev_data(self, dframe: sql.DataFrame) -> data_processor.TrainDevData:
        dframe = self._unnest_side_features(dframe)

        # NOTE: Metadata is created based on `input_cols` and dataset is on `output_cols`.
        # In order to sync column name of metadata with dataset, `output_cols` will later be renamed to `input_cols`.
        item_indexer = feature.StringIndexer(
            inputCols=self._item_feature_cols,
            outputCols=self._item_feature_idx_cols,
            handleInvalid="skip",
            stringOrderType="alphabetAsc",
        )

        indexer = ml.Pipeline(stages=[item_indexer])
        preprocessor = indexer.fit(dframe)

        dframe = self._build_data(dframe, preprocessor)
        dframe_train, dframe_dev = spark_utils.split_randomly(dframe, self._user_id_col, 1 - self._dev_split_size, self._random_seed)

        return data_processor.TrainDevData(preprocessor, data_processor.DataSplits(train=dframe_train, dev=dframe_dev))

    def _generate_test_data(self, dframe: sql.DataFrame) -> data_processor.DataSplits:
        dframe = self._unnest_side_features(dframe)
        preprocessor = self.load_preprocessor()
        dframe = self._build_data(dframe, preprocessor)

        return data_processor.DataSplits(test=dframe)
