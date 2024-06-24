import dataclasses
import math

import gin


@gin.configurable
@dataclasses.dataclass
class DataProcessConfig:
    """Configuration for data processing.

    Attributes:
        input_files: string or a list of strings. A list of input files. Input files can be local files or
            S3 objects. For example, ['data/file1.csv', 'data/file2.csv'] or ['s3://bucket/file1.csv',
            's3://bucket/file2.csv']. All input files must have the same scheme.
        output_path: Output directory path. It can be local path or S3.
        dev_split_size: The proportion of the dataset to include in the development(dev) split.
        run_locally: If True, DataProcessor create Spark in local mode. Otherwise, Spark job is deployed to
            its cluster.
        *_num_partitions: The number of re-partitions. After processing, the number of parquet files will be equal to this.
            This is applied to training datasets only. Increasing num_partitions improves the randomness in shuffling and the
            stability of the model. However, if it is too high, many small-sized files will be output.
    """

    input_files: str | list[str]
    output_path: str
    dev_split_size: float = dataclasses.field(default=0.1)
    run_locally: bool = dataclasses.field(default=False)
    train_num_partitions: int = dataclasses.field(default=10)
    test_num_partitions: int = dataclasses.field(default=1)
    dev_num_partitions: int | None = dataclasses.field(default=1)


@gin.configurable
@dataclasses.dataclass
class InteractionRange:
    min: int = dataclasses.field(default=0)
    max: int = dataclasses.field(default=math.inf)


@gin.configurable
@dataclasses.dataclass
class InteractionDataProcessConfig:
    """Configuration for interaction data processing.

    Attributes:
        target_interactions: The list of target interactions. If it is None, all interactions are considered as target.
        user_interaction_range: The range of user interactions to be considered.
        item_interaction_range: The range of item interactions to be considered.
        item_filter_first: If True, items are filtered before users. Otherwise, users are filtered first.
    """

    target_interactions: list[str] | None = dataclasses.field(default=None)
    user_interaction_range: InteractionRange = dataclasses.field(default_factory=InteractionRange)
    item_interaction_range: InteractionRange = dataclasses.field(default_factory=InteractionRange)
    item_filter_first: bool = dataclasses.field(default=True)


@gin.configurable
@dataclasses.dataclass
class InteractionDataSchemaConfig:
    """Configuration for input interaction data schema.

    For various interaction data schema, below configurations can be used to specify the column names.
    All column names will be renamed to unified names internally.

    Attributes:
        input_user_id_col: The name of the user id column in the input data.
        input_item_id_col: The name of the item id column in the input data.
        input_weight_col: The name of the weight column in the input data.
        input_target_interaction_col: The name of the target interaction column in the input data.
    """

    input_user_id_col: str = dataclasses.field(default="user_id")
    input_item_id_col: str = dataclasses.field(default="item_id")
    input_weight_col: str | None = dataclasses.field(default=None)
    input_target_interaction_col: str | None = dataclasses.field(default=None)


@gin.configurable
@dataclasses.dataclass
class UserConditionalDataConfig:
    """Configuration for user conditional data processing."""

    input_files: str | list[str]
    input_user_id_col: str = dataclasses.field(default="user_id")
    input_gender_col: str = dataclasses.field(default="gender")
    input_generation_col: str = dataclasses.field(default="generation")
