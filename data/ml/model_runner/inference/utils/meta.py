import os

from loguru import logger

from data.ml.utils import metadata
from data.pylib.constant import recsys as common

_DEFAULT_META_PATH = "s3://sagemaker-us-east-1-368316345532"
_META_PATH_ENV = "META_PATH"


class MetaLazyFactory:
    _instance = {}

    @classmethod
    def get_instance(cls, model_name: str, meta_path: str | None = None):
        resolved_path = meta_path or os.getenv(_META_PATH_ENV) or _DEFAULT_META_PATH
        logger.info(f"Resolved meta path: {resolved_path}")

        if cls._instance.get(model_name) is None:
            cls._instance[model_name] = metadata.Meta(filepath=os.path.join(resolved_path, "meta.json"))
        return cls._instance[model_name]

    @classmethod
    def clear(cls):
        cls._instance.clear()


def get_item_size(model_name: str):
    meta = MetaLazyFactory.get_instance(model_name)
    return meta.get_meta_size(common.ITEM_ID_COL)


def convert_id_to_idx(model_name: str, ids: list[str]):
    meta = MetaLazyFactory.get_instance(model_name)
    id_to_idx = meta.get_id_to_idx(common.ITEM_ID_COL)
    return list(map(lambda x: id_to_idx[x], ids))


def convert_idx_to_id(model_name, idx: list[int]):
    meta = MetaLazyFactory.get_instance(model_name)
    idx_to_id = meta.get_idx_to_id(common.ITEM_ID_COL)
    return list(map(lambda x: idx_to_id[x], idx))
