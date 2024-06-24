"""Variational Autoencoders for Collaborative Filtering

Paper: https://arxiv.org/pdf/1802.05814.pdf
This code is based on the author's implementation:
https://github.com/dawenl/vae_cf
"""
import copy
from typing import Union

import gin
import torch
from torch import nn
from torch.nn import functional as F

from data.pylib.constant import recsys as common
from data.ml.model_runner.metrics import ndcg
from data.ml.model_runner.metrics import recall
from data.ml.model_runner.models import model_base

_USER_ID_COL = common.USER_ID_COL
_ITEM_IDX_COL = common.ITEM_IDX_COL
_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_SCORE_COL = common.SCORE_COL


def _get_annealing_steps(total_anneal_steps: Union[float, int], total_training_steps: int) -> int:
    if not total_training_steps:
        return 0

    if isinstance(total_anneal_steps, float):
        return int(total_training_steps * total_anneal_steps)

    return total_anneal_steps


@gin.configurable
class VAE(model_base.RecsysSageMakerModelBase):
    _inputs_col = _INPUTS_COL
    _user_id_col = _USER_ID_COL
    _targets_col = _TARGETS_COL
    _item_idx_col = _ITEM_IDX_COL
    _score_col = _SCORE_COL

    def __init__(
        self,
        encoder_dims: list[int],
        decoder_dims: list[int],
        dropout: float = 0.5,
        anneal_cap: float = 0.2,
        total_anneal_steps: Union[float, int] = 0.2,
        **kwargs,
    ):
        """Constructs Variational Auto-Encoder.

        For the details of KL annealing, please refer to https://aclanthology.org/K16-1002.pdf.

        Args:
            encoder_dims: A list of latent dims.
            decoder_dims: A list of latent dims.
            dropout: Dropout rate.
            anneal_cap: The maximum annealing ratio.
            total_anneal_steps: The total number of gradient updates for annealing.
                If float, annealing will be done for `total_training_steps * total_anneal_steps` steps.
                If int, annealing will be done for `total_training_steps` steps.
        """
        super().__init__(**kwargs)

        encoder_dims = self._get_encoder_dims(encoder_dims)
        decoder_dims = self._get_decoder_dims(decoder_dims)

        self._assert_dims_valid(encoder_dims, decoder_dims)

        self._encoder_dims = encoder_dims
        self._decoder_dims = decoder_dims

        self._encoder = self._make_encoder_layers()
        self._decoder = self._make_decoder_layers()
        self._dropout = nn.Dropout(dropout)

        # Annealing params.
        self._global_steps = 0
        self._anneal_cap = anneal_cap
        self._total_anneal_steps = total_anneal_steps

        self._init_weights()

    def _get_encoder_dims(self, hidden_dims: list[int]) -> list[int]:
        return [self._item_size] + hidden_dims

    def _get_decoder_dims(self, hidden_dims: list[int]) -> list[int]:
        return hidden_dims + [self._item_size]

    def _assert_dims_valid(self, encoder_dims: list[int], decoder_dims: list[int]):
        assert encoder_dims[-1] == decoder_dims[0], "Latent dimension for encoder and decoder mismatches."

    def _make_encoder_layers(self) -> nn.ModuleList:
        # Last dim is for mean and variance.
        return nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(self._encoder_dims[:-2], self._encoder_dims[1:-1])]
            + [nn.Linear(self._encoder_dims[-2], self._encoder_dims[-1] * 2)]
        )

    def _make_decoder_layers(self) -> nn.ModuleList:
        return nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(self._decoder_dims[:-1], self._decoder_dims[1:])])

    def _init_weights(self):
        for layer in self._encoder:
            nn.init.xavier_normal_(layer.weight)
        for layer in self._decoder:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, data: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # pylint: disable=arguments-differ
        data = F.normalize(data)  # L2-normalization
        data = self._dropout(data)
        mean, logvar = self._encode(data)
        z = self._reparameterize(mean, logvar)
        return self._decode(z), mean, logvar

    def _encode(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = data

        for layer in self._encoder[:-1]:
            h = layer(h)
            h = torch.tanh(h)

        h = self._encoder[-1](h)
        mean = h[:, : self._encoder_dims[-1]]
        logvar = h[:, self._encoder_dims[-1] :]
        return mean, logvar

    def _decode(self, data: torch.Tensor) -> torch.Tensor:
        h = data
        for layer in self._decoder[:-1]:
            h = layer(h)
            h = torch.tanh(h)
        h = self._decoder[-1](h)
        return h

    def _reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)

        return mean

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)

    def loss(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        # pylint: disable=invalid-name
        anneal_steps = _get_annealing_steps(self._total_anneal_steps, self._total_training_steps)

        if anneal_steps > 0:
            anneal = min(self._anneal_cap, 1.0 * self._global_steps / anneal_steps)
        else:
            anneal = self._anneal_cap

        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        if self.training:
            self.log_dict(
                {
                    "anneal": anneal,
                    "anneal_cap": self._anneal_cap,
                },
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return BCE + anneal * KLD

    def training_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        batch = batch_data[self._inputs_col]
        recon_batch, mu, logvar = self(batch)
        loss = self.loss(recon_batch, batch_data[self._targets_col], mu, logvar)
        self.log("loss/train", loss, on_step=False, on_epoch=True, sync_dist=True)
        self._global_steps += 1
        self._collect_metric_on_train_step({"loss/train": loss})
        return loss

    def validation_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        batch = batch_data[self._inputs_col]
        recon_batch, mu, logvar = self(batch)
        loss = self.loss(recon_batch, batch, mu, logvar)
        self.log("loss/dev", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._collect_metric_on_validation_step({"loss/dev": loss})

        metrics_ndcg = {}
        metrics_recall = {}
        for k in [10, 20, 100]:
            ndcg_metric = f"metrics/ndcg@{k}"
            recall_metric = f"metrics/recall@{k}"

            metrics_ndcg[ndcg_metric] = ndcg.normalized_dcg_at_k(batch_data[_TARGETS_COL], recon_batch, k=k)
            metrics_recall[recall_metric] = recall.recall_at_k(batch_data[_TARGETS_COL], recon_batch, k=k)
            self._collect_metric_on_validation_step(
                {f"{ndcg_metric}": metrics_ndcg[ndcg_metric], f"{recall_metric}": metrics_recall[recall_metric]}
            )

        self.log_dict(metrics_ndcg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(metrics_recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(
        self,
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):  # pylint: disable=arguments-renamed
        batch = batch_data[self._inputs_col]
        recon_batch, _, _ = self(batch)
        return self._generator(recon_batch, batch_data)


class CVAE(VAE):
    """Conditional Variational Auto-Encoder
    Paper: https://proceedings.neurips.cc/paper/2014/file/d523773c6b194f37b938d340d5d02232-Paper.pdf
    """

    def __init__(self, label_cols: list[str], **kwargs):
        """Constructs Conditional Variational Auto-Encoder.

        Args:
            label_cols: A list of label columns.
        """
        self._label_cols = label_cols
        super().__init__(**kwargs)

    @property
    def _labels_size(self) -> int:
        size = 0
        for col in self._label_cols:
            size += self._meta.get_meta_size(col)
        return size

    def _get_encoder_dims(self, hidden_dims: list[int]) -> list[int]:
        return [self._item_size + self._labels_size] + hidden_dims

    def _get_decoder_dims(self, hidden_dims: list[int]) -> list[int]:
        hidden_dims = copy.deepcopy(hidden_dims)
        hidden_dims[0] += self._labels_size
        return hidden_dims + [self._item_size]

    def _assert_dims_valid(self, encoder_dims: list[int], decoder_dims: list[int]):
        assert encoder_dims[-1] == decoder_dims[0] - self._labels_size, "Latent dimension for encoder and decoder mismatches."

    def _concat_labels(self, batch_data: dict[str, torch.Tensor]) -> torch.Tensor:
        labels = []
        for col in self._label_cols:
            labels.append(batch_data[col])
        return torch.cat(labels, dim=1)

    # pylint: disable=arguments-differ
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = F.normalize(inputs)  # L2-normalization
        inputs = self._dropout(inputs)

        data = torch.cat([inputs, labels], dim=1)

        mean, logvar = self._encode(data)
        z = self._reparameterize(mean, logvar)
        return self._decode(torch.cat([z, labels], dim=1)), mean, logvar

    # pylint: disable=arguments-differ
    def training_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int):
        batch_inputs = batch_data[self._intputs_col]
        batch_labels = self._concat_labels(batch_data)

        recon_batch, mu, logvar = self(batch_inputs, batch_labels)
        loss = self.loss(recon_batch, batch_data[self._targets_col], mu, logvar)
        self.log("loss/train", loss, on_step=False, on_epoch=True, sync_dist=True)
        self._global_steps += 1

        return loss

    def validation_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        batch_inputs = batch_data[self._intputs_col]
        batch_labels = self._concat_labels(batch_data)

        recon_batch, mu, logvar = self(batch_inputs, batch_labels)
        loss = self.loss(recon_batch, batch_inputs, mu, logvar)
        self.log("loss/dev", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        metrics_ndcg = {}
        metrics_recall = {}
        for k in [10, 20, 100]:
            metrics_ndcg[f"metrics/ndcg@{k}"] = ndcg.normalized_dcg_at_k(batch_data[self._targets_col], recon_batch, k=k)
            metrics_recall[f"metrics/recall@{k}"] = recall.recall_at_k(batch_data[self._targets_col], recon_batch, k=k)

        self.log_dict(metrics_ndcg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log_dict(metrics_recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(
        self,
        batch_data: dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):  # pylint: disable=arguments-renamed
        batch_inputs = batch_data[self._inputs_col]
        batch_labels = self._concat_labels(batch_data)

        recon_batch, _, _ = self(batch_inputs, batch_labels)
        return self._generator(recon_batch, batch_data)
