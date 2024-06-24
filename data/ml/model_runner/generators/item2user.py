import abc
import functools

import gin
from loguru import logger
import numpy as np
import pandas as pd
import torch

from data.ml.model_runner.generators import base
from data.pylib.constant import recsys as common

_INPUTS_COL = common.INPUTS_COL
_USER_ID_COL = common.USER_ID_COL
_ITEM_ID_COL = common.ITEM_ID_COL
_ITEM_IDX_COL = common.ITEM_IDX_COL
_SCORE_COL = common.SCORE_COL


class Item2UserGenerator(base.Generator, abc.ABC):
    _user_id_col = _USER_ID_COL
    _item_id_col = _ITEM_ID_COL
    _item_idx_col = _ITEM_IDX_COL
    _score_col = _SCORE_COL

    def _to_pdf(
        self,
        user_id: torch.Tensor,
        item_idx: torch.Tensor,
        score: torch.Tensor,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self._user_id_col: self._convert_to_list(user_id),
                self._item_idx_col: self._convert_to_list(item_idx),
                self._score_col: self._convert_to_list(score),
            }
        )

    @functools.cached_property
    def _item_idx_to_id_pdf(self) -> pd.DataFrame:
        item_idx_to_id = [
            {
                self._item_idx_col: idx,
                self._item_id_col: item_id,
            }
            for idx, item_id in self._item_idx_to_id.items()
        ]
        return pd.DataFrame(item_idx_to_id)

    def _postprocess(self, dframe: pd.DataFrame) -> pd.DataFrame:
        dframe = (
            dframe.set_index(self._user_id_col)
            .apply(pd.Series.explode)
            .reset_index()
            .astype({self._item_idx_col: "int", self._score_col: "float32"})
        )
        if self._revert_idx_to_id:
            dframe = dframe.merge(self._item_idx_to_id_pdf, on=self._item_idx_col)
            dframe = dframe.drop(columns=[self._item_idx_col])
            dframe = dframe.reindex(columns=[self._user_id_col, self._item_id_col, self._score_col])
        return dframe


@gin.configurable
class TopKGenerator(Item2UserGenerator):
    def __init__(self, top_k: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._top_k = min(self._item_size, top_k) if top_k else self._item_size

        logger.info(f"Top K: {self._top_k}")

    def _generate(self, logits: torch.Tensor, batch_data: dict[str, torch.Tensor]) -> pd.DataFrame:
        topk_score, topk_idx = torch.topk(logits, k=self._top_k)
        return self._to_pdf(batch_data[self._user_id_col], topk_idx, topk_score)


@gin.configurable
class ExclusionGenerator(TopKGenerator):
    def __init__(self, exclusion_col: str = _INPUTS_COL, threshold: float = 0, **kwargs):
        super().__init__(**kwargs)
        self._exclusion_col = exclusion_col
        self._threshold = threshold

        logger.info(f"Exclusion column: {self._exclusion_col}, threshold: {self._threshold}")

    def _generate(self, logits: torch.Tensor, batch_data: dict[str, torch.Tensor]) -> pd.DataFrame:
        logits[batch_data[self._exclusion_col] > self._threshold] = -np.inf
        return super()._generate(logits, batch_data)
