import gin
from pyspark import sql
from pyspark.sql import functions as F

from data.ml.data_runner.data_processors import data_processor_v2 as data_processor
from data.ml.data_runner.data_processors import spark_utils
from data.pylib.constant import recsys as common

_ITEM_IDX_COL = common.ITEM_IDX_COL
_ITEM_PREPROCESSOR_NAME = common.ITEM_PREPROCESSOR_NAME

_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL


@gin.configurable
class UserSequentialDataProcessor(data_processor.InteractionDataProcessor):
    """Sequence data processor.

    The expected data format is as follows (including headers):
    +----------------------+--------+------------+--------------+
    |  user_id |  item_id  | weight | event_name | timestamp    |
    +----------------------+--------+------------+--------------+
    |    1234  |  '45423'  |  1.0   |   read     |  1671345496  |
    |    1234  |  '3422'   |  1.2   |   purchase |  1671354875  |
    |    1234  |  '83671'  |  0.7   |   open     |  1671364995  |
    +----------------------+--------+------------+--------------+

    The resulting data format is as follows:
    Each row corresponds to a user's item-interaction history.
    ( item_id -> item_idx : (45423 -> 1), (3422 -> 2), (83671 -> 3) )
    +-------------+-------------+
    |  user_id    |  inputs     |
    +-------------+-------------+
    |  1234       |  [1, 2, 3]  |
    +-------------+-------------+
    """

    _item_idx_col = _ITEM_IDX_COL
    _item_preprocessor_name = _ITEM_PREPROCESSOR_NAME

    _inputs_col = _INPUTS_COL
    _targets_col = _TARGETS_COL

    def __init__(self, holdout_proportion: float = 0.2, max_seq_len: int | None = None, slice_step_size: int = 1, **kwargs):
        super().__init__(**kwargs)

        self._holdout_proportion = holdout_proportion
        self._max_seq_len = max_seq_len
        self._slice_step_size = slice_step_size

    def _split_by_holdout_proportion(self, split_col: str | sql.Column) -> tuple[sql.Column, sql.Column]:
        turning_point = F.floor(F.size(split_col) * F.lit(1 - self._holdout_proportion))
        front = F.slice(split_col, F.lit(1), turning_point)
        back = F.slice(split_col, turning_point + 1, F.size(split_col) - turning_point)
        return front, back

    def _generate_train_dev_data(self, dframe: sql.DataFrame) -> data_processor.DataSplits:
        dframe, item_preprocessor = self._convert_key_to_idx(dframe, self._item_id_col, self._item_idx_col)
        self._save_preprocessor(item_preprocessor, self._item_preprocessor_name)

        user = dframe.select(self._user_id_col).distinct()
        train_user, dev_user = spark_utils.split_randomly(user, self._user_id_col, 1 - self._holdout_proportion, self._random_seed)

        dframe_train = dframe.join(train_user, self._user_id_col, "inner")
        dframe_dev = dframe.join(dev_user, self._user_id_col, "inner")

        dframe_train = self._get_train_data(dframe_train)
        dframe_dev = self._get_dev_data(dframe_dev)

        return data_processor.DataSplits(train=dframe_train, dev=dframe_dev)

    def _generate_test_data(self, dframe: sql.DataFrame) -> data_processor.DataSplits:
        preprocessor = self._load_preprocessor(self._item_preprocessor_name)
        dframe, _ = self._convert_key_to_idx(dframe, self._item_id_col, self._item_idx_col, preprocessor)
        dframe = self._get_test_data(dframe)
        return data_processor.DataSplits(test=dframe)

    def _get_user_item_sequence(self, dframe: sql.DataFrame) -> sql.DataFrame:
        dframe = (
            dframe.groupBy(self._user_id_col)
            .agg(F.collect_list(F.struct(self._timestamp_col, self._item_idx_col)).alias("items"))
            .withColumn("sequence", F.array_sort("items"))
            .select(self._user_id_col, "sequence")
        )
        return dframe

    def _extract_inputs_from_sequence(self, sequence: sql.Column) -> sql.Column:
        return F.transform(sequence, lambda x: x[self._item_idx_col])

    def _get_train_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        dframe = self._get_user_item_sequence(dframe)

        if self._max_seq_len:
            start = F.lit(1)
            end = F.size("sequence") - F.lit(self._max_seq_len - 1)
            step = F.lit(self._slice_step_size)

            start_indices = F.sequence(start, end, step)
            sliced_sequences = F.when(
                F.size("sequence") > self._max_seq_len,
                F.reverse(
                    F.transform(
                        start_indices,
                        lambda i: F.reverse(F.slice(F.reverse("sequence"), i, self._max_seq_len)),
                    )
                ),
            ).otherwise(F.array("sequence"))

            dframe = dframe.withColumn("sequences", sliced_sequences)
            dframe = dframe.withColumn("sequence", F.explode("sequences"))

        inputs = self._extract_inputs_from_sequence(F.col("sequence"))
        return dframe.select(self._user_id_col, inputs.alias(self._inputs_col))

    def _get_dev_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        dframe = self._get_user_item_sequence(dframe)
        sequence = F.col("sequence")

        if self._max_seq_len:
            max_input_target_size = F.lit(int(self._max_seq_len / (1 - self._holdout_proportion)))
            input_target_size = F.least(F.size(sequence), max_input_target_size)

            sequence = F.slice(sequence, -input_target_size, input_target_size)

        inputs = self._extract_inputs_from_sequence(sequence)
        inputs, targets = self._split_by_holdout_proportion(inputs)

        return dframe.select(
            F.col(self._user_id_col),
            inputs.alias(self._inputs_col),
            targets.alias(self._targets_col),
        )

    def _get_test_data(self, dframe: sql.DataFrame) -> sql.DataFrame:
        dframe = self._get_user_item_sequence(dframe)
        sequence = F.col("sequence")

        if self._max_seq_len:
            max_input_size = F.lit(self._max_seq_len)
            input_size = F.least(F.size(sequence), max_input_size)
            sequence = F.slice(sequence, -input_size, input_size)

        inputs = self._extract_inputs_from_sequence(sequence)
        return dframe.select(self._user_id_col, inputs.alias(self._inputs_col))
