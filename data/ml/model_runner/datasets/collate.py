import abc

import gin
import torch

from data.pylib.constant import recsys as common
from data.ml.utils import metadata

_USER_ID_COL = common.USER_ID_COL
_ITEM_ID_COL = common.ITEM_ID_COL
_TARGETS_COL = common.TARGETS_COL
_POSITIVE_CONTEXTS_COL = common.POSITIVE_CONTEXTS_COL
_NEGATIVE_CONTEXTS_COL = common.NEGATIVE_CONTEXTS_COL


class Collator(abc.ABC):
    def __init__(
        self,
        meta: metadata.Meta,
    ):
        self._meta = meta

    @abc.abstractmethod
    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pass


@gin.configurable
class Item2VecCollator(Collator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._item_size = self._meta.get_meta_size(_ITEM_ID_COL)

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate function.
        B   : batch size
        K   : length of sequence
        C_p : number of contexts (positive batch). In paper, C = K - 1
        C_n : number of contexts (negative batch).
        *_max : max length of *_col
        For each sequence, K, C can be different.
        by padding sequence to `item_size + 1` and masking, we can make all sequences have same length.
        Args:
            batch: list of dict, each dict contains following keys:
                - targets: torch.Tensor, shape of (K)
                - positive_contexts: torch.Tensor, shape of (K, C_p)
                - negative_contexts: torch.Tensor, shape of (K, C_n)
        Returns:
            batch_data: dict, contains following keys:
                - targets: torch.Tensor, shape of (B, K_max)
                - positive_contexts: torch.Tensor, shape of (B, K_max, C_p_max)
                - negative_contexts: torch.Tensor, shape of (B, K_max, C_n_max)
        """

        batch_targets = torch.nn.utils.rnn.pad_sequence(
            [sample[_TARGETS_COL] for sample in batch if len(sample[_TARGETS_COL]) > 0],
            batch_first=True,
            padding_value=self._item_size,
        )

        K_max = batch_targets.shape[1]  # pylint: disable=invalid-name

        def _padding_contexts(contexts):
            K = contexts.shape[0]  # pylint: disable=invalid-name
            C = contexts.shape[1]  # pylint: disable=invalid-name
            contexts_pad = torch.ones((K_max - K, C), dtype=torch.int) * self._item_size
            return torch.cat([contexts, contexts_pad], dim=0)

        batch_positive_contexts = []
        batch_negative_contexts = []

        for sample in batch:
            positive_contexts = sample[_POSITIVE_CONTEXTS_COL]
            negative_contexts = sample[_NEGATIVE_CONTEXTS_COL]

            positive_contexts = _padding_contexts(positive_contexts)
            negative_contexts = _padding_contexts(negative_contexts)

            positive_contexts = positive_contexts.transpose(0, 1)
            negative_contexts = negative_contexts.transpose(0, 1)

            batch_positive_contexts.append(positive_contexts)
            batch_negative_contexts.append(negative_contexts)

        batch_positive_contexts = torch.nn.utils.rnn.pad_sequence(
            batch_positive_contexts,
            batch_first=True,
            padding_value=self._item_size,
        ).transpose(1, 2)

        batch_negative_contexts = torch.nn.utils.rnn.pad_sequence(
            batch_negative_contexts,
            batch_first=True,
            padding_value=self._item_size,
        ).transpose(1, 2)

        batch_data = {
            _TARGETS_COL: batch_targets,
            _POSITIVE_CONTEXTS_COL: batch_positive_contexts,
            _NEGATIVE_CONTEXTS_COL: batch_negative_contexts,
        }
        return batch_data
