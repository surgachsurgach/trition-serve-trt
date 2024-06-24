""" Implemetation of 'Contrastive Learning for Sequential Recommendation'. SIGIR `21
    https://arxiv.org/abs/2010.14395
"""
import functools

import gin
import torch
from torch import nn

from data.pylib.constant import recsys as common
from data.ml.model_runner.metrics import ndcg
from data.ml.model_runner.metrics import recall
from data.ml.model_runner.models import model_base
from data.ml.model_runner.models import transformer

_ITEM_IDX = common.ITEM_ID_COL
_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_ALL_TARGETS_COL = common.ALL_TARGETS_COL
_NEXT_TARGET_COL = common.NEXT_TARGET_COL
_SEQ_LEN_COL = common.SEQ_LEN_COL


@gin.configurable
class CL4SRec(model_base.RecsysModelBase):
    """
    CL4SRec not only takes advantage of the traditional next item prediction task
    but also utilizes the contrastive learning framework to derive self-supervision signals from
    the original user behavior sequences. Therefore, it can extract more
    meaningful user patterns and further encode the user representation effectively

    Args:
    - d_model(num): the input and output hidden size. Default: 64
    - num_heads(num): num of attention heads for multi-head attention layer. Default: 2
    - num_encoder_layers(num): num of transformer layers in transformer encoder. Default: 2
    - dropout(float): probability of an element to be zeroed. Default: 0.5
    - attn_dropout(float): dropout probability for transformer encoder. Default: 0.3
    - max_seq_len: maximum length of sequence to use
    - lmd: proportion of contrastive loss to add to main loss. Default: 0.1
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        dropout: float = 0.5,
        attn_dropout: float = 0.3,
        max_seq_len: int = 50,
        lmd: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._item_emb = nn.Embedding(self._item_size + 2, d_model, padding_idx=0)
        self._pos_emb = nn.Embedding(max_seq_len, d_model)

        self._norm = nn.LayerNorm(d_model, eps=1e-12)
        self._dropout = nn.Dropout(dropout)
        self._lmd = lmd

        self._encoder = transformer.TransformerEncoder(num_encoder_layers, d_model, num_heads, dropout=attn_dropout)
        self._criterion = nn.CrossEntropyLoss()
        self._criterion_cl = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @functools.cache  # pylint: disable=method-cache-max-size-none
    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def _gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, data, steps, lookup_key: str = _ITEM_IDX) -> torch.Tensor:  # pylint: disable=arguments-differ
        item_emb = self._item_emb(data[lookup_key].long())

        # position embedding
        pos_ids = torch.arange(data[lookup_key].size(1), dtype=torch.long, device=self.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(data[lookup_key])
        pos_emb = self._pos_emb(pos_ids)

        x = self._norm(item_emb + pos_emb)
        x = self._dropout(x)

        output = self._encoder(x, (data[lookup_key] > 0).long())
        output = self._gather_indexes(output, steps - 1)
        return output

    def _rcmd_loss(self, y_pred, y_true):
        test_item_emb = self._item_emb.weight
        logits = torch.matmul(y_pred, test_item_emb.transpose(0, 1))
        loss = self._criterion(logits, y_true[_ITEM_IDX].long())

        return loss

    def _constrastive_loss(self, z_i: torch.Tensor, z_j: torch.Tensor, batch_size: int):
        dim = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.mm(z, z.T)
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(dim, 1)
        mask = self.mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(dim, -1)

        labels = torch.zeros(dim).to(self.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        info_nce_loss = self._criterion_cl(logits, labels)
        return info_nce_loss

    def training_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        logits = self(batch_data[_INPUTS_COL], batch_data[_SEQ_LEN_COL])
        loss = self._rcmd_loss(logits, batch_data[_TARGETS_COL])

        # NCE
        seq_output1 = self.forward(batch_data[_INPUTS_COL], batch_data[_INPUTS_COL]["augmented_seq_len_0"], lookup_key="augmented_seq_0")
        seq_output2 = self.forward(batch_data[_INPUTS_COL], batch_data[_INPUTS_COL]["augmented_seq_len_1"], lookup_key="augmented_seq_1")

        cl_loss = self._lmd * self._constrastive_loss(seq_output1, seq_output2, batch_data["inputs"]["augmented_seq_0"].size(0))

        self.log("loss/train", loss + cl_loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss + cl_loss

    def validation_step(self, batch_data: dict[str, torch.Tensor], batch_idx: int):  # pylint: disable=arguments-differ
        logits = self(batch_data[_INPUTS_COL], batch_data[_SEQ_LEN_COL])
        loss = self._rcmd_loss(logits, batch_data[_TARGETS_COL])

        self.log("loss/dev", loss, on_step=False, on_epoch=True, prog_bar=True)

        test_item_emb = self._item_emb.weight
        logits = torch.matmul(logits, test_item_emb[: self._item_size + 1].transpose(0, 1))

        target_types = []
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
