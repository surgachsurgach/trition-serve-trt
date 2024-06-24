"""
https://arxiv.org/pdf/1603.04259.pdf
"""

import gin
import torch
from torch import nn
from torch.nn import functional as F

from data.pylib.constant import recsys as common
from data.ml.model_runner.models import model_base

_ITEM_IDX_COL = common.ITEM_IDX_COL
_OTHER_ITEM_IDX_COL = common.OTHER_ITEM_IDX_COL
_EMBEDDING_VECTOR_COL = common.EMBEDDING_VECTOR_COL
_SIMILARITY_COL = common.SIMILARITY_COL

_TARGETS_COL = common.TARGETS_COL
_POSITIVE_CONTEXTS_COL = common.POSITIVE_CONTEXTS_COL
_NEGATIVE_CONTEXTS_COL = common.NEGATIVE_CONTEXTS_COL


@gin.configurable
class Item2Vec(model_base.RecsysSageMakerModelBase):
    _item_idx_col = _ITEM_IDX_COL
    _other_item_idx_col = _OTHER_ITEM_IDX_COL
    _embedding_vector_col = _EMBEDDING_VECTOR_COL
    _similarity_col = _SIMILARITY_COL

    _targets_col = _TARGETS_COL
    _positive_contexts_col = _POSITIVE_CONTEXTS_COL
    _negative_contexts_col = _NEGATIVE_CONTEXTS_COL

    def __init__(
        self,
        embedding_dim: int = 2,
        negative_loss_weight: float = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._embedding_dim = embedding_dim
        self._embedding_layer = self._make_embedding_layer()
        self._negative_loss_weight = negative_loss_weight
        self._param = kwargs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)

    def _make_embedding_layer(self) -> nn.Embedding:
        return nn.Embedding(self._item_size + 1, self._embedding_dim, padding_idx=self._item_size)

    def embedding(self, data: torch.Tensor) -> torch.Tensor:
        return self._embedding_layer(data)

    def forward(self, targets: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        """Forward pass.
        targets: (B, K)
        contexts: (B, K, C)
        """
        targets_embedding = self.embedding(targets)  # (B, K, M)
        contexts_embedding = self.embedding(contexts)  # (B, K, C, M)

        # (B, K, 1, M) @ (B, K, C, M) ^ T -> (B, K, 1, C) -> (B, K, C)
        similarity = torch.matmul(targets_embedding.unsqueeze(2), contexts_embedding.transpose(2, 3))
        similarity = similarity.squeeze(2)
        return similarity  # similarity is in [-inf, +inf].

    # pylint: disable=arguments-differ
    def training_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int):
        """Training step.
        B   : batch size
        K   : length of sequence
        C_p : number of contexts (positive samples). In paper, C = K - 1
        C_n : number of contexts (negative samples).
        C   : C_p + C_n
        M   : embedding dimension
        """
        batch_targets = batch_data[self._targets_col]  # (B, K)
        batch_positive_contexts = batch_data[self._positive_contexts_col]  # (B, K, C_p)
        batch_negative_contexts = batch_data[self._negative_contexts_col]  # (B, K, C_n)

        batch_contexts = torch.cat([batch_positive_contexts, batch_negative_contexts], dim=2)  # (B, K, C)
        y_pred = self.forward(batch_targets, batch_contexts)  # (B, K, C)

        y_true = torch.cat(
            [
                torch.ones_like(batch_positive_contexts, dtype=torch.float32),
                torch.zeros_like(batch_negative_contexts, dtype=torch.float32),
            ],
            dim=2,
        )  # (B, K, C)
        weight = torch.cat(
            [
                torch.ones_like(batch_positive_contexts, dtype=torch.float32),
                torch.ones_like(batch_negative_contexts, dtype=torch.float32) * self._negative_loss_weight,
            ],
            dim=2,
        )  # (B, K, C)

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=weight)
        self.log("loss/train", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # pylint: disable=arguments-differ
    def validation_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        batch_targets = batch_data[self._targets_col]
        batch_positive_contexts = batch_data[self._positive_contexts_col]
        batch_negative_contexts = batch_data[self._negative_contexts_col]

        batch_contexts = torch.cat([batch_positive_contexts, batch_negative_contexts], dim=2)
        y_pred = self.forward(batch_targets, batch_contexts)

        y_true = torch.cat(
            [
                torch.ones_like(batch_positive_contexts, dtype=torch.float32),
                torch.zeros_like(batch_negative_contexts, dtype=torch.float32),
            ],
            dim=2,
        )
        weight = torch.cat(
            [
                torch.ones_like(batch_positive_contexts, dtype=torch.float32),
                torch.ones_like(batch_negative_contexts, dtype=torch.float32) * self._negative_loss_weight,
            ],
            dim=2,
        )

        loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight=weight)
        self.log("loss/dev", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # pylint: disable=arguments-renamed
    def predict_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        return self._generator(batch_data, self.embedding)
