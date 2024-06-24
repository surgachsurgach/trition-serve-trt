import abc
import functools
from typing import Callable

import gin
import pandas as pd
import torch

from data.ml.model_runner.generators import base
from data.pylib.constant import recsys as common

_ITEM_ID_COL = common.ITEM_ID_COL
_ITEM_IDX_COL = common.ITEM_IDX_COL
_OTHER_ITEM_ID_COL = common.OTHER_ITEM_ID_COL
_OTHER_ITEM_IDX_COL = common.OTHER_ITEM_IDX_COL
_EMBEDDING_VECTOR_COL = common.EMBEDDING_VECTOR_COL
_SIMILARITY_COL = common.SIMILARITY_COL


class Item2ItemGenerator(base.Generator, abc.ABC):
    _item_id_col = _ITEM_ID_COL
    _item_idx_col = _ITEM_IDX_COL

    @functools.cached_property
    def _item_idx_to_id_pdf(self) -> pd.DataFrame:
        """Reverse Indexing for Item Idx to Id"""
        item_idx_to_id = [
            {
                self._item_idx_col: idx,
                self._item_id_col: item_id,
            }
            for idx, item_id in self._item_idx_to_id.items()
        ]
        return pd.DataFrame(item_idx_to_id)


@gin.configurable
class EmbeddingVectorGenerator(Item2ItemGenerator):
    _embedding_vector_col = _EMBEDDING_VECTOR_COL

    def _to_pdf(
        self,
        item_idx: torch.Tensor,
        batch_embedding_vectors: torch.Tensor,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self._item_idx_col: item_idx,
                self._embedding_vector_col: batch_embedding_vectors,
            }
        )

    def _postprocess(self, dframe: pd.DataFrame) -> pd.DataFrame:
        dframe = dframe.astype({self._item_idx_col: "int64", self._embedding_vector_col: "object"})
        if self._revert_idx_to_id:
            dframe = dframe.merge(self._item_idx_to_id_pdf, on=self._item_idx_col)
            dframe = dframe.drop(columns=[self._item_idx_col])
            dframe = dframe.reindex(columns=[self._item_id_col, self._embedding_vector_col])
        return dframe

    def _generate(self, batch_data: dict[str, torch.Tensor], embedding_fn: Callable, **kwargs) -> pd.DataFrame:
        batch_targets = batch_data[self._item_idx_col]

        item_idx = self._convert_to_list(batch_targets)
        batch_embedding_vectors = embedding_fn(batch_targets)
        batch_embedding_vectors = self._convert_to_list(batch_embedding_vectors)
        return self._to_pdf(item_idx, batch_embedding_vectors)


@gin.configurable
class SimilarityGenerator(Item2ItemGenerator):
    _other_item_id_col = _OTHER_ITEM_ID_COL
    _other_item_idx_col = _OTHER_ITEM_IDX_COL
    _similarity_col = _SIMILARITY_COL

    def __init__(self, top_k: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._top_k = min(self._item_size, top_k) if top_k else self._item_size

    @functools.cached_property
    def _other_item_idx_to_id_pdf(self) -> pd.DataFrame:
        """Reverse Indexing for Other Item Idx to Id"""
        return self._item_idx_to_id_pdf.rename(
            columns={
                self._item_idx_col: self._other_item_idx_col,
                self._item_id_col: self._other_item_id_col,
            }
        )

    def _get_top_k_similarity(self, batch_similarity: torch.Tensor, batch_targets: torch.Tensor) -> pd.DataFrame:
        top_k_similarity, top_k_sim_item_idx = torch.topk(batch_similarity, k=self._top_k)
        dframe = pd.DataFrame(
            {
                self._item_idx_col: self._convert_to_list(batch_targets),
                self._other_item_idx_col: self._convert_to_list(top_k_sim_item_idx),
                self._similarity_col: self._convert_to_list(top_k_similarity),
            }
        )
        return dframe

    def _get_total_similarity(self, batch_similarity: torch.Tensor, batch_targets: torch.Tensor) -> pd.DataFrame:
        item_idx = self._convert_to_list(batch_targets)
        batch_output = {self._item_idx_col: item_idx}

        batch_similarity = batch_similarity.detach().cpu().numpy()
        for j in range(self._item_size):
            batch_output[j] = batch_similarity[:, j].tolist()

        dframe = pd.DataFrame(batch_output)

        # Reverse of pivot.
        return pd.melt(
            dframe,
            id_vars=[self._item_idx_col],
            var_name=self._other_item_idx_col,
            value_name=self._similarity_col,
        )

    def _generate(self, batch_data: dict[str, torch.Tensor], embedding_fn: Callable, **kwargs) -> pd.DataFrame:
        batch_targets = batch_data[self._item_idx_col]
        total_contexts = torch.arange(self._item_size, device=batch_targets.device)

        batch_targets_embedding = embedding_fn(batch_targets)
        total_contexts_embedding = embedding_fn(total_contexts)

        batch_similarity = torch.matmul(batch_targets_embedding, total_contexts_embedding.transpose(0, 1))
        batch_similarity = batch_similarity / (batch_targets_embedding.norm(dim=1).unsqueeze(1) * total_contexts_embedding.norm(dim=1))

        if self._top_k != self._item_size:
            dframe = self._get_top_k_similarity(batch_similarity, batch_targets)
        else:
            dframe = self._get_total_similarity(batch_similarity, batch_targets)
        return dframe

    def _postprocess(self, dframe: pd.DataFrame) -> pd.DataFrame:
        dframe = (
            dframe.set_index(self._item_idx_col)
            .apply(pd.Series.explode)
            .reset_index()
            .astype(
                {
                    self._item_idx_col: "int64",
                    self._other_item_idx_col: "int64",
                    self._similarity_col: "float32",
                }
            )
        )
        if self._revert_idx_to_id:
            dframe = dframe.merge(self._item_idx_to_id_pdf, on=self._item_idx_col)
            dframe = dframe.drop(columns=[self._item_idx_col])
            dframe = dframe.merge(self._other_item_idx_to_id_pdf, on=self._other_item_idx_col)
            dframe = dframe.drop(columns=[self._other_item_idx_col])
            dframe = dframe.reindex(columns=[self._item_id_col, self._other_item_id_col, self._similarity_col])
        return dframe
