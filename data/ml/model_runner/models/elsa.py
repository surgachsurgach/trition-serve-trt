"""Scalable Linear Shallow Autoencoder for Collaborative Filtering

paper: https://dl.acm.org/doi/abs/10.1145/3523227.3551482
code: https://github.com/recombee/ELSA
"""
from typing import Dict

import gin
import pandas as pd
import torch
from torch import nn

from data.pylib.constant import recsys as common
from data.ml.model_runner.metrics import ndcg
from data.ml.model_runner.metrics import recall
from data.ml.model_runner.models import model_base

_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_ALL_TARGETS_COL = common.ALL_TARGETS_COL
_NEXT_TARGET_COL = common.NEXT_TARGET_COL
_SEQ_LEN_COL = common.SEQ_LEN_COL
_USER_ID_COL = common.USER_ID_COL


@gin.configurable
class ELSA(model_base.RecsysModelBase):
    def __init__(self, num_dims: int, lr: float = 0.01, **kwargs):
        super().__init__(**kwargs)

        self._num_dims = num_dims
        self._lr = lr

        self._weights = nn.ParameterList(
            [nn.Parameter(nn.init.xavier_uniform_(torch.empty(self._item_size, self._num_dims, requires_grad=True)))]
        )
        self._criterion = nn.MSELoss()
        self._cos_sim = nn.CosineSimilarity(dim=1, eps=1e-08)

    def _get_weights(self):
        return torch.vstack(list(self._weights))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=invalid-name
        A = nn.functional.normalize(self._get_weights(), dim=-1)
        xA = torch.matmul(x, A)
        xAAT = torch.matmul(xA, A.T)
        return xAAT - x

    def configure_optimizers(self):
        return torch.optim.NAdam(self.parameters(), lr=self._lr)

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self._criterion(nn.functional.normalize(y_pred, dim=-1), nn.functional.normalize(y_true, dim=-1))
        return loss

    def training_step(self, batch_data: Dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        batch = batch_data[_INPUTS_COL]
        output = self(batch)
        loss = self.loss(output, batch_data[_TARGETS_COL])
        return loss

    def validation_step(self, batch_data: Dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        batch = batch_data[_INPUTS_COL]
        output = self(batch)
        loss = self.loss(output, batch_data[_TARGETS_COL])
        self.log("loss/dev", loss, on_step=True, on_epoch=True)

        ndcg_at_100 = ndcg.normalized_dcg_at_k(batch_data[_TARGETS_COL], output, k=100)
        recall_at_50 = recall.recall_at_k(batch_data[_TARGETS_COL], output, k=50)
        recall_at_20 = recall.recall_at_k(batch_data[_TARGETS_COL], output, k=20)

        self.log("metrics/cos_sim", torch.mean(self._cos_sim(output, batch_data[_TARGETS_COL]), dim=-1).item())
        self.log_dict({"metrics/ndcg@100": ndcg_at_100}, on_step=True, on_epoch=True)
        self.log_dict(
            {"metrics/recall@20": recall_at_20, "metrics/recall@50": recall_at_50},
            on_step=True,
            on_epoch=True,
        )

    def predict_step(
        self,
        batch_data: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        batch = batch_data.pop(_INPUTS_COL)
        output = self(batch)
        topk_score, topk_idx = torch.topk(output, k=self._predict_top_k)

        dframe = pd.DataFrame(
            {
                "user_id": batch_data[_USER_ID_COL].detach().cpu().numpy().tolist(),
                "item_idx": topk_idx.detach().cpu().numpy().tolist(),
                "score": topk_score.detach().cpu().numpy().tolist(),
            }
        )

        dframe = dframe.set_index("user_id").apply(pd.Series.explode).reset_index().astype({"item_idx": "int", "score": "float"})

        return dframe
