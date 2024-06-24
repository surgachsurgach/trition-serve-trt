import enum

import gin


@gin.constants_from_enum
class DatasetType(enum.Enum):
    TRAIN_DATASET = "train"
    DEV_DATASET = "dev"
    PREDICT_DATASET = "predict"
