""" Implementation of Bert4Rec algorithm.

    Followed huggingface implemenation as explained in the paper:
        A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation. Recsys `22
        (https://arxiv.org/pdf/2207.07483.pdf)
"""

import enum
from typing import Any

import gin
import torch
import transformers

from data.pylib.constant import recsys as common
from data.ml.model_runner.base import model_base
from data.ml.model_runner.generators import base as base_generator
from data.ml.model_runner.utils import sagemaker_utils
from data.ml.utils import metadata


class SageMakerModelLoggingMixin(model_base.ModelBase, sagemaker_utils.MetricLoggingMixin):
    def _collect_metric_on_train_step(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if key not in self.training_step_outputs:
                self.training_step_outputs[key] = value
            else:
                self.training_step_outputs[key] += value

    def _collect_metric_on_validation_step(self, metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if key not in self.validation_step_outputs:
                self.validation_step_outputs[key] = value
            else:
                self.validation_step_outputs[key] += value

    def _log_metric_on_train_epoch_end(self) -> None:
        for key, value in self.training_step_outputs.items():
            epoch_mean = value / self.train_steps_per_epoch
            self.run.log_metric(key, epoch_mean, step=self.current_epoch)
        self.training_step_outputs.clear()

    def _log_metric_on_validation_epoch_end(self, **kwargs) -> None:
        for key, value in self.validation_step_outputs.items():
            epoch_mean = value / self.validation_steps_per_epoch
            self.run.log_metric(key, epoch_mean, step=self.current_epoch)
        self.validation_step_outputs.clear()


class RecsysModelBase(model_base.ModelBase):
    def __init__(
        self,
        meta: metadata.Meta,
        generator: base_generator.Generator | None = None,
        predict_top_k: int | None = None,  # Deprecated
        exclude_inputs_from_predictions: bool = True,  # Deprecated
        **kwargs,
    ):
        """Model base class.

        Args:
            meta: metadata of the model.
        """
        super().__init__(**kwargs)

        self._meta = meta
        self._generator = generator
        self._predict_top_k = min(self._item_size, predict_top_k) if predict_top_k else None
        self.exclude_inputs_from_predictions = exclude_inputs_from_predictions
        self.save_hyperparameters(ignore="meta")
        self._print_hparams()

    @property
    def _item_size(self) -> int:
        return self._meta.get_meta_size(common.ITEM_ID_COL)

    @property
    def _user_size(self):
        return self._meta.get_meta_size(common.USER_ID_COL)


class RecsysSageMakerModelBase(RecsysModelBase, SageMakerModelLoggingMixin):
    def on_train_epoch_end(self):
        self._log_metric_on_train_epoch_end()

    def on_validation_epoch_end(self):
        self._log_metric_on_validation_epoch_end()


_USER_ID_COL = common.USER_ID_COL
_ALL_TARGETS_COL = common.ALL_TARGETS_COL
_NEXT_TARGET_COL = common.NEXT_TARGET_COL
_SEQ_LEN_COL = common.SEQ_LEN_COL
_ITEM_ID_COL = common.ITEM_ID_COL
_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_TARGET_IDX_COL = common.TARGET_IDX_COL


@gin.constants_from_enum
class BertVariant(enum.Enum):
    BERT = "bert"
    DEBERTA = "deberta"
    DEBERTA_V2 = "deberta-v2"


_BERT_CONFIGS = {
    BertVariant.BERT: (transformers.BertConfig, transformers.BertForMaskedLM),
    BertVariant.DEBERTA: (transformers.DebertaConfig, transformers.DebertaForMaskedLM),
    BertVariant.DEBERTA_V2: (transformers.DebertaV2Config, transformers.DebertaV2ForMaskedLM),
}


@gin.configurable
class Bert4Rec(RecsysSageMakerModelBase):
    """Bert4Rec uses a masked language model to predict the next item in the sequence.

    Args:
    - d_model(num): the input and output hidden size.
    - num_heads(num): num of attention heads for multi-head attention layer.
    - num_encoder_layers(num): num of transformer layers in transformer encoder.
    - d_ff(num): the dimension of the feedforward network model. Default: 128
    - dropout(float): probability of an element to be zeroed. Default: 0.5
    - attn_dropout(float): dropout probability for transformer encoder. Default: 0.3
    - max_seq_len: maximum length of sequence to use
    - bert_variant: bert variant to use. Default: "bert", options: "bert", "deberta", "deberta-v2"
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        max_seq_len: int = 50,
        bert_variant: BertVariant = BertVariant.BERT,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._pad_token_id = self._item_size
        self._mask_token_id = self._item_size + 1

        config, masked_lm = transformers.BertConfig, transformers.BertForMaskedLM

        self._config = config(
            vocab_size=self._item_size + 2,  # padding at position -2, mask_token_id at postition -1
            hidden_size=d_model,
            num_attention_heads=num_heads,
            num_hidden_layers=num_encoder_layers,
            hidden_act="gelu_pytorch_tanh",
            attention_probs_dropout_prob=attn_dropout,
            intermediate_size=d_ff,
            hidden_dropout_prob=dropout,
            max_position_embeddings=max_seq_len + 2,
            pad_token_id=self._pad_token_id,
        )

        self._bert = masked_lm(self._config)

    def get_name(self) -> str:
        return "_".join([self.__class__.__name__, self._bert.__class__.__name__])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)

    def forward(self, inputs: torch.Tensor, target_idx: torch.Tensor | None = None, targets: torch.Tensor | None = None, ):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)

        output = self._bert(
            inputs,
            labels=targets,
            position_ids=pos_ids,
            attention_mask=inputs != self._pad_token_id,
        )

        if target_idx is not None:
            gather_index = target_idx.view(-1, 1, 1).expand(-1, -1, output.logits.shape[-1])
            output.logits = output.logits.gather(dim=1, index=gather_index).squeeze(1)[:, :-2]

        if targets is None:
            output.loss = torch.tensor(0, dtype=torch.int8)  # dummy loss not used in predict_step.

        return output.logits, output.loss


    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        _, loss = self(batch[_INPUTS_COL], targets=batch[_TARGETS_COL])

        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._collect_metric_on_train_step({"loss/train": loss})
        return loss
