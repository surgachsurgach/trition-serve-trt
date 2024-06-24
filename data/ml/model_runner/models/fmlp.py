"""Filter-enhanced MLP is All You Need for Sequential Recommendation, WWW 2022
https://arxiv.org/abs/2202.13556
"""

import copy

import gin
from lightning import pytorch as pl
import torch
from torch import nn

from data.pylib.constant import recsys as common
from data.ml.model_runner.metrics import ndcg
from data.ml.model_runner.metrics import recall
from data.ml.model_runner.models import model_base
from data.ml.model_runner.modules.layers import mlp

_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_ALL_TARGETS_COL = common.ALL_TARGETS_COL
_NEXT_TARGET_COL = common.NEXT_TARGET_COL
_SEQ_LEN_COL = common.SEQ_LEN_COL
_USER_ID_COL = common.USER_ID_COL


class FilterLayer(pl.LightningModule):
    """
    Perform Fast Fourier Transform (FFT) to convert the input representations into the
    frequency domain and an inverse FFT procedure recovers the denoised representations.
    The filter component plays a key role in reducing the influence of the noise from item representations.

    Args:
    - hidden_size(num): output hidden size.
    - max_seq_len: maximum length of sequence to use.
    - dropout(float): probability of an element to be zeroed. Default: 0.0
    """

    def __init__(self, hidden_size: int, max_seq_len: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self._complex_weight = nn.Parameter(torch.randn(1, max_seq_len // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        self._dropout = nn.Dropout(dropout)
        self._norm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor: torch.Tensor):
        # [batch, seq_len, hidden]
        _, seq_len, _ = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        weight = torch.view_as_complex(self._complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")

        hidden_states = self._dropout(sequence_emb_fft)
        hidden_states = self._norm(hidden_states + input_tensor)
        return hidden_states


class EncoderLayer(pl.LightningModule):
    def __init__(self, hidden_size: int, max_seq_len: int, dropout: float = 0.3, mlp_dropout: float = 0.3, **kwargs):
        super().__init__(**kwargs)

        self._filter_layer = FilterLayer(hidden_size, max_seq_len, dropout)
        self._mlp = mlp.MLP([hidden_size, hidden_size * 4, hidden_size], dropout=mlp_dropout)

    def forward(self, x: torch.Tensor):
        x = self._filter_layer(x)
        x = self._mlp(x)
        return x


class Encoder(pl.LightningModule):
    def __init__(
        self,
        num_encoder_layers: int,
        hidden_size: int,
        max_seq_len: int,
        dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        layer = EncoderLayer(hidden_size, max_seq_len, dropout, mlp_dropout, **kwargs)
        self._layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_encoder_layers)])

    def forward(self, x: torch.Tensor, output_all_encoded_layers: bool = True):
        all_encoder_layers = []
        for layer in self._layers:
            x = layer(x)
            if output_all_encoded_layers:
                all_encoder_layers.append(x)

        if not output_all_encoded_layers:
            return [x]
        return all_encoder_layers


_ITEM_IDX = common.ITEM_ID_COL


@gin.configurable
class FMLP(model_base.RecsysModelBase):
    """FMLP-Rec stacks multiple Filter-enhanced Blocks to produce the representation of
    sequential user preference for recommendation.
    The key difference between our approach and SASRec is to replace
    the multi-head self-attention structure in Transformer with a novel filter structure.

    Args:
    - d_model(num): output hidden size.
    - max_seq_len: maximum length of sequence to use.
    - dropout(float): probability of an element to be zeroed. Default: 0.3
    - mlp_dropout(float): dropout prob for mlp layers. Default: 0.3
    - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        d_model: int,
        num_encoder_layers: int,
        max_seq_len: int,
        dropout: float = 0.3,
        mlp_dropout: float = 0.3,
        layer_norm_eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._item_emb = nn.Embedding(self._item_size, d_model)
        self._pos_emb = nn.Embedding(max_seq_len, d_model)
        self._norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)
        self._item_encoder = Encoder(num_encoder_layers, d_model, max_seq_len, dropout, mlp_dropout)

        self._criterion = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def _gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, data, steps):
        item_emb = self._item_emb(data[_ITEM_IDX].long())

        # position embedding
        pos_ids = torch.arange(data[_ITEM_IDX].size(1), dtype=torch.long, device=self.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(data[_ITEM_IDX])
        pos_emb = self._pos_emb(pos_ids)

        x = self._norm(item_emb + pos_emb)
        x = self._dropout(x)
        x = self._item_encoder(x, output_all_encoded_layers=True)
        x = self._gather_indexes(x[-1], steps - 1)
        return x

    def loss(self, y_pred, y_true):
        test_item_emb = self._item_emb.weight
        logits = torch.matmul(y_pred, test_item_emb.transpose(0, 1))
        loss = self._criterion(logits, y_true[_ITEM_IDX].long())

        return loss

    def training_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # pylint: disable=arguments-differ
        logits = self(batch_data[_INPUTS_COL], batch_data[_SEQ_LEN_COL])
        loss = self.loss(logits, batch_data[_TARGETS_COL])

        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        logits = self(batch_data[_INPUTS_COL], batch_data[_SEQ_LEN_COL])
        loss = self.loss(logits, batch_data[_TARGETS_COL])

        self.log("loss/dev", loss, on_step=False, on_epoch=True, prog_bar=True)

        test_item_emb = self._item_emb.weight

        logits = torch.matmul(logits, test_item_emb.transpose(0, 1))

        target_types = []
        if _ALL_TARGETS_COL in batch_data:
            target_types.append(_ALL_TARGETS_COL)
        if _NEXT_TARGET_COL in batch_data:
            target_types.append(_NEXT_TARGET_COL)

        for target in target_types:
            metric_ndcg = {}
            metric_recall = {}
            for top_k in [100, 50, 20, 10]:
                metric_ndcg[f"metrics_{target}/ndcg@{top_k}"] = ndcg.normalized_dcg_at_k(batch_data[target], logits, k=top_k)
                metric_recall[f"metrics_{target}/recall@{top_k}"] = recall.recall_at_k(batch_data[target], logits, k=top_k)

            self.log_dict(metric_ndcg, on_step=False, on_epoch=True)
            self.log_dict(metric_recall, on_step=False, on_epoch=True)
