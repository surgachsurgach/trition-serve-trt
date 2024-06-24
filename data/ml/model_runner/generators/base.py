import abc
import functools

from loguru import logger
import pandas as pd
import torch

from data.ml.utils import metadata
from data.pylib.constant import recsys as common

_ITEM_ID_COL = common.ITEM_ID_COL
_ITEM_IDX_COL = common.ITEM_IDX_COL


class Generator(abc.ABC):
    def __init__(self, meta: metadata.Meta, revert_idx_to_id: bool = True):
        self._meta = meta
        self._revert_idx_to_id = revert_idx_to_id

        logger.info(f"Reverse indexing: {self._revert_idx_to_id}")

    @property
    def _item_size(self) -> int:
        return self._meta.get_meta_size(_ITEM_ID_COL)

    @functools.cached_property
    def _item_idx_to_id(self):
        return self._meta.get_idx_to_id(_ITEM_ID_COL)

    @staticmethod
    def _convert_to_list(tensor: torch.Tensor):
        return tensor.detach().cpu().numpy().tolist()

    @abc.abstractmethod
    def _postprocess(self, dframe: pd.DataFrame) -> pd.DataFrame:
        """Postprocess Generated Data

        Postprocess like aggregate, explode, idx2id, etc.
        """
        pass

    @abc.abstractmethod
    def _generate(self, *args, **kwargs) -> pd.DataFrame:
        """Generate Target Data"""
        pass

    def generate(self, *args, **kwargs) -> pd.DataFrame:
        return self._postprocess(self._generate(*args, **kwargs))

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
