import itertools
import math
from typing import Callable, Dict, List, Optional, Union

from loguru import logger
import numpy as np
from pyspark import conf
from pyspark import sql
from pyspark.sql import functions as F


def create_spark(app_name: str, run_locally: bool, scheme: str):
    spark_builder = sql.SparkSession.builder
    spark_conf = conf.SparkConf()
    if run_locally:
        logger.info("Run PySpark locally.")
        spark_builder = spark_builder.appName(app_name)
        spark_conf.set("spark.master", "local[*]")
        spark_conf.set("spark.driver.host", "localhost")

    spark_builder = spark_builder.config(conf=spark_conf)
    return spark_builder.getOrCreate()


def _get_ratio(ratio: Union[float, List[float]]) -> Union[float, List[float]]:
    if isinstance(ratio, float):
        if ratio < 0 or ratio > 1.0:
            raise ValueError("Split ratio has to be between 0 and 1")
        ratio = [ratio, 1 - ratio]
    else:
        if math.fsum(ratio) != 1.0:
            ratio = [x / math.fsum(ratio) for x in ratio]
    return ratio


def _stratification(
    dframe: sql.DataFrame,
    split_by: str,
    order_by: str = None,
    ratio: Union[float, List[float]] = 0.75,
    is_partitioned: bool = True,
    order_randomly: bool = True,
    seed=None,
):
    partition_by = split_by if is_partitioned else []

    window_count = sql.Window.partitionBy(partition_by)
    if order_randomly:
        window_spec = sql.Window.partitionBy(partition_by).orderBy(F.rand(seed))
    else:
        window_spec = sql.Window.partitionBy(partition_by).orderBy(order_by)

    dframe = (
        dframe.withColumn("_count", F.count(split_by).over(window_count))
        .withColumn("_rank", F.row_number().over(window_spec) / F.col("_count"))
        .drop("_count")
    )

    # Persist to avoid duplicate rows in splits caused by lazy evaluation
    dframe.persist()

    ratio = _get_ratio(ratio)

    splits = []
    prev_split = -np.inf
    for split in np.cumsum(ratio):
        condition = F.col("_rank") <= split
        condition &= F.col("_rank") > prev_split
        splits.append(dframe.filter(condition).drop("_rank"))
        prev_split = split

    return splits


def split_randomly(
    dframe: sql.DataFrame, split_by: str, ratio: Union[float, List[float]], seed: Optional[int] = None
) -> List[sql.DataFrame]:
    return _stratification(dframe, split_by, ratio=ratio, is_partitioned=False, order_randomly=True, seed=seed)


def split_chronologically(
    dframe: sql.DataFrame, split_by: str, order_by: str, ratio: Union[float, List[float]], seed: Optional[int] = None
) -> List[sql.DataFrame]:
    return _stratification(dframe, split_by, order_by=order_by, ratio=ratio, is_partitioned=True, order_randomly=False, seed=seed)


def negative_sampling(
    split_by: str,
    positive_col: str,
    target_col: str,
    positive_item_ids: List[int],
    negative_sample_ratio: float,
    shuffle_items: bool,
    seed: Optional[int] = None,
) -> Callable:
    negative_col = f"negative_{positive_col}"

    def wrapper(dframe: sql.DataFrame) -> sql.DataFrame:
        # fmt: off
        dframe = (
            dframe
            .groupby(split_by).agg(F.collect_list(positive_col).alias(positive_col))
            .withColumn(negative_col, F.array(*[F.lit(x) for x in positive_item_ids]))
            .withColumn(negative_col, F.array_except(negative_col, positive_col))
        )

        if shuffle_items:
            dframe = dframe.withColumn(negative_col, F.shuffle(negative_col))

        dframe = dframe.withColumn(
            negative_col,
            F.slice(
                negative_col,
                F.lit(1),
                F.ceil(F.size(F.col(positive_col)) * negative_sample_ratio),
            ),
        )
        dframe_positive = dframe.select(
            split_by,
            F.explode(positive_col).alias(positive_col),
            F.lit(1).alias(target_col),
        )
        dframe_negative = dframe.select(
            split_by,
            F.explode(negative_col).alias(positive_col),
            F.lit(0).alias(target_col),
        )
        return dframe_positive.union(dframe_negative).orderBy(F.rand(seed))
        # fmt: on

    return wrapper


def get_id2index_map(id2idx: Dict[int, int]) -> sql.Column:
    return F.create_map(*[F.lit(x) for x in itertools.chain(*id2idx.items())])
