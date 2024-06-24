"""Decouple Side Information Fusion for Sequential Recommendation
https://arxiv.org/abs/2204.11046

"""
import copy
import functools
import math
from typing import Dict, List, Optional

import gin
from lightning import pytorch as pl
import pandas as pd
import torch
from torch import nn

from data.ml.model_runner.metrics import ndcg
from data.ml.model_runner.metrics import recall
from data.ml.model_runner.models import model_base
from data.ml.model_runner.models import transformer
from data.ml.model_runner.modules.layers import attention
from data.ml.model_runner.modules.layers import feed_forward
from data.pylib.constant import recsys as common

_USER_ID_COL = common.USER_ID_COL
_ITEM_IDX_COL = common.ITEM_IDX_COL
_ALL_TARGETS_COL = common.ALL_TARGETS_COL
_NEXT_TARGET_COL = common.NEXT_TARGET_COL
_SEQ_LEN_COL = common.SEQ_LEN_COL


class DIFMultiHeadAttention(attention.MultiHeadAttention):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_side_features: int,
        d_side_features: List[int],
        dropout: float,
        attn_dropout: float,
        layer_norm_eps: float,
        fusion_type: str,
        max_len: int,
    ):
        super().__init__(d_model, num_heads, dropout)

        if d_side_features:
            assert all(dim % num_heads for dim in d_side_features) == 0, "'d_side_features' should be divisible by 'num_heads'."

        self.attr_d_k = [dim // num_heads for dim in d_side_features]
        self.fusion_type = fusion_type
        self.max_len = max_len

        self.pos_q_linear = nn.Linear(d_model, d_model)
        self.pos_k_linear = nn.Linear(d_model, d_model)

        self.feature_q_linears = nn.ModuleList([copy.deepcopy(nn.Linear(d, d)) for d in d_side_features])
        self.feature_k_linears = nn.ModuleList([copy.deepcopy(nn.Linear(d, d)) for d in d_side_features])

        if self.fusion_type == "concat":
            self.fusion_layer = nn.Linear(self.max_len * (2 + num_side_features), self.max_len)
        elif self.fusion_type == "gate":
            self.fusion_layer = attention.VanillaAttention(self.max_len, self.max_len)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attr_table: List[torch.Tensor],
        position_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):  # pylint: disable=arguments-differ
        batch_size = x.size(0)

        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        pos_q = self.pos_q_linear(position_embedding).view(batch_size, -1, self.num_heads, self.d_k)
        pos_k = self.pos_k_linear(position_embedding).view(batch_size, -1, self.num_heads, self.d_k)

        pos_q = pos_q.transpose(1, 2)
        pos_k = pos_k.transpose(1, 2)

        base_attn_scores = torch.matmul(q, k.transpose(-2, -1))
        pos_scores = torch.matmul(pos_q, pos_k.transpose(-2, -1))

        feature_attn_scores = self._get_feature_scores(batch_size, attr_table)
        attn_scores = self._get_attn_score(base_attn_scores, pos_scores, feature_attn_scores)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attn_scores.float())

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.transpose(1, 2)
        context_layer = context_layer.contiguous().view(batch_size, -1, self.d_model)
        hidden_states = self.out(context_layer)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + x)
        return hidden_states

    def _get_feature_scores(self, batch_size: int, attr_table: List[torch.Tensor]):
        feature_scores = []

        for tensor, a_q, a_k, attr_d_k in zip(attr_table, self.feature_q_linears, self.feature_k_linears, self.attr_d_k):
            tensor = tensor.squeeze(-2)
            attr_q = a_q(tensor).view(batch_size, -1, self.num_heads, attr_d_k)
            attr_k = a_k(tensor).view(batch_size, -1, self.num_heads, attr_d_k)

            attr_q = attr_q.transpose(1, 2)
            attr_k = attr_k.transpose(1, 2)

            scores = torch.matmul(attr_q, attr_k.transpose(-2, -1))
            feature_scores.append(scores.unsqueeze(-2))

        return torch.cat(feature_scores, dim=-2) if feature_scores else None

    def _get_attn_score(
        self, base_attn_scores: torch.Tensor, pos_scores: torch.Tensor, feature_scores: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.fusion_type == "sum":
            attn_scores = torch.sum(feature_scores, dim=-2) if feature_scores else torch.empty([]).to(self.device, non_blocking=True)
            attn_scores = attn_scores + base_attn_scores + pos_scores
        elif self.fusion_type == "concat":
            tensors = [base_attn_scores, pos_scores]
            if feature_scores:
                attn_scores = feature_scores.view(feature_scores.shape[:-2] + (feature_scores.shape[-2] * feature_scores.shape[-1],))
                tensors.insert(0, attn_scores)
            attn_scores = self.fusion_layer(torch.cat(tensors, dim=-1))
        elif self.fusion_type == "gate":
            tensors = [base_attn_scores.unsqueeze(-2), pos_scores.unsqueeze(-2)]
            if feature_scores is not None:
                tensors.insert(0, feature_scores)
            attn_scores, _ = self.fusion_layer(torch.cat(tensors, dim=-2))
        else:
            raise RuntimeError("fusion_type must be one of ('sum', 'concat', 'gate')")

        return attn_scores / math.sqrt(self.d_k)


class DIFTransformerEncoderLayer(pl.LightningModule):
    """
    One decoupled transformer layer consists of a decoupled multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_side_features: int,
        d_side_features: List[int],
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        fusion_type: str = "gate",
        max_len: int = 50,
    ):
        super().__init__()
        self.dif_attn = DIFMultiHeadAttention(
            d_model,
            num_heads,
            num_side_features,
            d_side_features,
            dropout,
            attn_dropout,
            layer_norm_eps,
            fusion_type,
            max_len,
        )

        self._ff = feed_forward.FeedForward(d_model, d_ff, activation, 0.0, dropout)
        self._norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(  # pylint: disable=arguments-differ
        self,
        x: torch.Tensor,
        attr_emb: List[torch.Tensor],
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out = self.dif_attn(x, attr_emb, pos_emb, mask)
        return self._norm(self._ff(attn_out) + attn_out)


class DIFTransformerEncoder(pl.LightningModule):
    """One decoupled TransformerEncoder consists of several decoupled TransformerLayers.

    Args:
    - d_model(num): the input and output hidden size. Default: 64
    - num_heads(num): num of attention heads for multi-head attention layer. Default: 2
    - num_encoder_layers(num): num of transformer layers in transformer encoder. Default: 2
    - d_ff(num): the dimensionality in feed-forward layer. Default: 256
    - num_side_features(list): the hidden size of attributes. Default:[64]
    - feat_num(num): the number of attributes. Default: 1
    - dropout(float): probability of an element to be zeroed. Default: 0.5
    - attn_dropout(float): probability of an attention score to be zeroed. Default: 0.5
    - activation(str): activation function in feed-forward layer. Default: 'gelu'
                candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
    - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                        candidates: 'sum','concat','gate'
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_side_features: int,
        d_side_features: List[int],
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        fusion_type: str = "gate",
        max_len: int = 10,
    ):
        super().__init__()
        layer = DIFTransformerEncoderLayer(
            d_model,
            num_heads,
            num_side_features,
            d_side_features,
            d_ff,
            dropout,
            attn_dropout,
            activation,
            layer_norm_eps,
            fusion_type,
            max_len,
        )
        self._layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_encoder_layers)])

    def forward(
        self,
        x: torch.Tensor,
        attr_x: List[torch.Tensor],
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        output_all_encoded_layers: bool = True,
    ):  # pylint: disable=arguments-differ
        """
        Args:
            x (torch.Tensor): the input of the TransformerEncoder
            mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer in self._layers:
            x = layer(x, attr_x, pos_emb, mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(x)

        if not output_all_encoded_layers:
            return [x]
        return all_encoder_layers


_ITEM_IDX = common.ITEM_ID_COL


@gin.configurable
class DIF(model_base.RecsysModelBase):  # pylint: disable=too-many-instance-attributes
    """
    DIF-SR moves the side information from the input to the attention layer and decouples the attention calculation of
    various side information and item representation

    Args:
    - d_model(num): the input and output hidden size. Default: 64
    - num_heads(num): num of attention heads for multi-head attention layer. Default: 2
    - num_encoder_layers(num): num of transformer layers in transformer encoder. Default: 2
    - d_ff(num): the dimensionality in feed-forward layer. Default: 256
    - side_features(list): side features to use e.g. ['item_category','author']
    - d_side_features(list): hidden size of each feature
    - dropout(float): probability of an element to be zeroed. Default: 0.5
    - attn_dropout(float): probability of an attention score to be zeroed. Default: 0.5
    - activation(str): activation function in feed-forward layer. Default: 'gelu'
                candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
    - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    - fusion_type(str): fusion function used in attention fusion module. Default: 'sum'
                        candidates: 'sum','concat','gate'
    - lamdas(list): loss scaler for each feature
    - feature_predictor(str): model to use to predict next feature token
    - max_seq_len: maximum length of sequence to use
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        d_ff: int,
        side_features: List[str],
        d_side_features: List[int],
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        pooling_mode: str = "sum",
        fusion_type: str = "gate",
        lambdas: List[int] = None,
        feature_predictor: str = "linear",
        max_seq_len: int = 50,
        target_genre: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert len(side_features) == len(d_side_features)
        assert len(side_features) == len(lambdas)

        self._d_model = d_model
        self._num_encoder_layers = num_encoder_layers
        self._d_side_features = d_side_features
        self._side_features = side_features
        self._initializer_range = initializer_range
        self._lambdas = lambdas

        self._feature_tokens = [self._meta.get_meta_size(f.replace("_idx", "")) + 1 for f in side_features]

        self._item_emb = nn.Embedding(self._item_size, d_model)
        self._pos_emb = nn.Embedding(max_seq_len, d_model)

        self._feature_embs = self._create_feature_embeddings()

        self._encoder = DIFTransformerEncoder(
            d_model,
            num_heads,
            num_encoder_layers,
            len(side_features),
            d_side_features,
            d_ff,
            dropout,
            attn_dropout,
            activation,
            layer_norm_eps,
            fusion_type,
            max_seq_len,
        )

        self._feature_predictor = self._create_feature_predictor(feature_predictor)
        self._norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        self._criterion = nn.CrossEntropyLoss()
        self._feature_criterion = nn.BCEWithLogitsLoss(reduction="none")
        self._target_genre = target_genre
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self._initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _create_feature_embeddings(self) -> nn.ParameterList:
        embeddings = []
        for token, dim in zip(self._feature_tokens, self._d_side_features):
            embeddings.append(nn.Embedding(token, dim, padding_idx=0))

        return nn.ParameterList(embeddings)

    def _create_feature_predictor(self, predictor_type: str) -> nn.ModuleList:
        if predictor_type == "linear":
            return nn.ModuleList(
                [copy.deepcopy(nn.Linear(in_features=self._d_model, out_features=token)) for token in self._feature_tokens]
            )

        assert False

    def _gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self._lr)

    @functools.cached_property
    def _target_item_start_idx(self) -> int:
        assert self._meta

        start_idx = self._meta.get_start_idx("item_id_prefix", lambda x: x == self._target_genre)
        assert start_idx is not None
        return start_idx

    @functools.cached_property
    def _target_item_size(self) -> int:
        assert self._meta
        return self._meta.get_meta_size("item_id_prefix", cond=lambda x: x == self._target_genre)

    def _get_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        extended_attn_mask = transformer._get_extended_attention_mask(attention_mask)  # pylint: disable=protected-access
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).to(self.device, non_blocking=True)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        extended_attn_mask = extended_attn_mask * subsequent_mask.long()
        return extended_attn_mask

    def forward(self, data, steps):  # pylint: disable=arguments-differ
        item_emb = self._item_emb(data[_ITEM_IDX].long())

        # position embedding
        pos_ids = torch.arange(data[_ITEM_IDX].size(1), dtype=torch.long, device=self.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(data[_ITEM_IDX])
        pos_emb = self._pos_emb(pos_ids)

        feature_emb = []
        for i, feature in enumerate(self._side_features):
            feature_emb.append(self._feature_embs[i](data[feature].long()))

        x = self._norm(item_emb)
        x = self._dropout(x)

        extended_mask = self._get_attention_mask((data[_ITEM_IDX] > 0).long())
        output = self._encoder(x, feature_emb, pos_emb, extended_mask, True)
        output = self._gather_indexes(output[-1], steps - 1)
        return output

    def loss(self, y_pred, y_true):
        test_item_emb = self._item_emb.weight
        logits = torch.matmul(y_pred, test_item_emb.transpose(0, 1))
        loss = self._criterion(logits, y_true[_ITEM_IDX].long())

        losses = {"item_loss": loss}

        for feature_name, num_classes, f_pred in zip(self._side_features, self._feature_tokens, self._feature_predictor):
            feature_logits = f_pred(y_pred)
            feature_labels = nn.functional.one_hot(y_true[feature_name].long(), num_classes=num_classes)

            if len(feature_labels.shape) > 2:
                feature_labels = feature_labels.sum(dim=1)

            feature_loss = self._feature_criterion(feature_logits, feature_labels.float())
            feature_loss = torch.mean(feature_loss[:, 1:])
            losses[feature_name] = feature_loss

        feature_loss_sum = 0
        for feature, l in zip(self._side_features, self._lambdas):
            feature_loss_sum += l * losses[feature]

        total_loss = loss + feature_loss_sum
        losses["total_loss"] = total_loss

        return total_loss

    def training_step(self, batch_data: Dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        logits = self(batch_data[common.INPUTS_COL], batch_data[_SEQ_LEN_COL])
        loss = self.loss(logits, batch_data[common.TARGETS_COL])

        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch_data: Dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        logits = self(batch_data[common.INPUTS_COL], batch_data[_SEQ_LEN_COL])
        loss = self.loss(logits, batch_data[common.TARGETS_COL])

        self.log("loss/dev", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if self._target_genre is not None:
            test_item_emb = self._item_emb.weight[self._target_item_start_idx : self._target_item_start_idx + self._target_item_size]
        else:
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

            self.log_dict(metric_ndcg, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log_dict(metric_recall, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0):
        logits = self(batch[common.INPUTS_COL], batch[_SEQ_LEN_COL])
        logits = torch.matmul(logits, self._item_emb.weight.transpose(0, 1))
        topk_score, topk_idx = torch.topk(logits, k=self._predict_top_k)

        dframe = pd.DataFrame(
            {
                _USER_ID_COL: (
                    batch[common.USER_ID_COL].detach().cpu().numpy().tolist()
                    if isinstance(batch[_USER_ID_COL], torch.Tensor)
                    else batch[_USER_ID_COL]
                ),
                _ITEM_IDX_COL: topk_idx.detach().cpu().numpy().tolist(),
                "score": topk_score.detach().cpu().numpy().tolist(),
            }
        )

        dframe = dframe.set_index(_USER_ID_COL).apply(pd.Series.explode).reset_index().astype({_ITEM_IDX_COL: "int", "score": "float32"})

        return dframe
