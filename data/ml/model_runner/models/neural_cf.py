"""Neural Collaborative Filtering.

Paper: https://arxiv.org/pdf/1708.05031.pdf
"""
from typing import Dict, List

import gin
import torch
from torch import nn

from data.pylib.constant import recsys as common
from data.ml.model_runner.models import model_base

# TODO(ywlee): Implements metrics such as NDCG, MAP.
# Because NCF is a sort of 0/1 classification model, we need to feed pairs of user-item
# for all users and items which is very time-consuming and impractical.
# Thus, we only implement training_step for now.


_USER_INPUTS_COL = common.USER_INPUTS_COL
_ITEM_INPUTS_COL = common.ITEM_INPUTS_COL
_TARGETS_COL = common.TARGETS_COL


@gin.configurable
class NeuralCF(model_base.RecsysModelBase):
    def __init__(
        self,
        mlp_latent_dim: int,
        mf_latent_dim: int,
        mlp_dims: List[int],
        dropout: float = 0.0,
        **kwargs,
    ):
        """Constructs Neural Collaborative Filtering model.

        Args:
            mlp_latent_dim: The size of embedding vector of Multi-Layer Perceptron(MLP).
            mf_latent_dim: The sie of embedding vector of Matrix Factorization(MF).
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self._mlp_latent_dim = mlp_latent_dim
        self._mf_latent_dim = mf_latent_dim
        self._mlp_dims = mlp_dims

        self._embed_user_mlp = nn.Embedding(self._user_size, self._mlp_latent_dim)
        self._embed_item_mlp = nn.Embedding(self._item_size, self._mlp_latent_dim)
        self._embed_user_mf = nn.Embedding(self._user_size, self._mf_latent_dim)
        self._embed_item_mf = nn.Embedding(self._item_size, self._mf_latent_dim)
        self._mlp_layers = self._make_mlp_layers()
        self._dropout = nn.Dropout(dropout)
        self._final = nn.Linear(self._mlp_dims[-1] + self._mf_latent_dim, 1)

        self._criterion = nn.BCEWithLogitsLoss(reduction="none")

        self.param = kwargs

        self._init_weights()

    def _make_mlp_layers(self) -> nn.ModuleList:
        return nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(self._mlp_dims[:-1], self._mlp_dims[1:])])

    def _init_weights(self):
        nn.init.xavier_normal(self._embed_user_mlp.weight)
        nn.init.xavier_normal(self._embed_item_mlp.weight)
        nn.init.xavier_normal(self._embed_user_mf.weight)
        nn.init.xavier_normal(self._embed_item_mf.weight)
        for layer in self._mlp_layers:
            nn.init.xavier_normal(layer.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, **kwargs) -> torch.Tensor:  # pylint: disable=arguments-differ
        # MF
        embed_user_mf = self._embed_user_mf(user_ids)
        embed_item_mf = self._embed_item_mf(item_ids)
        mf_vector = torch.mul(embed_user_mf, embed_item_mf)  # element-wise multiplication.

        # MLP
        embed_user_mlp = self._embed_user_mlp(user_ids)
        embed_item_mlp = self._embed_item_mlp(item_ids)
        mlp_vector = torch.concat([embed_user_mlp, embed_item_mlp], dim=-1)
        for mlp_layer in self._mlp_layers:
            mlp_vector = mlp_layer(mlp_vector)
            mlp_vector = torch.relu(mlp_vector)
            mlp_vector = self._dropout(mlp_vector)

        x = torch.concat([mf_vector, mlp_vector], dim=-1)
        return self._final(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = self._criterion(y_pred, y_true.view(-1, 1)).float()
        loss = torch.mean(loss.view(-1), 0)

        return loss

    def training_step(self, batch_data: Dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        batch_user = batch_data[_USER_INPUTS_COL]
        batch_item = batch_data[_ITEM_INPUTS_COL]
        logits = self(batch_user, batch_item)
        loss = self.loss(logits, batch_data[_TARGETS_COL].float())

        self.log("loss/train", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch_data: Dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        batch_user = batch_data[_USER_INPUTS_COL]
        batch_item = batch_data[_ITEM_INPUTS_COL]
        logits = self(batch_user, batch_item)
        loss = self.loss(logits, batch_data[_TARGETS_COL].float())

        self.log("loss/dev", loss, on_step=True, on_epoch=True)
