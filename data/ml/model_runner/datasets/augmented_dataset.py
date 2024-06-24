import math
import random

import gin
import torch

from data.ml.model_runner.datasets import parquet_dataset

_NUM_AUGMENTATION = 2
_NUM_AUGMENT_TYPES = 3


def _item_crop(seq: torch.Tensor, length: int, eta: float = 0.6) -> tuple[torch.Tensor, int]:
    num_left = math.floor(length * eta)
    crop_begin = random.randint(0, length - num_left)
    croped_item_seq = torch.zeros(seq.shape[0], dtype=torch.long)
    if crop_begin + num_left < seq.shape[0]:
        croped_item_seq[:num_left] = seq[crop_begin : crop_begin + num_left]
    else:
        croped_item_seq[:num_left] = seq[crop_begin:]
    return croped_item_seq, num_left


def _item_mask(seq: torch.Tensor, length: int, item_size: int, gamma: float = 0.3) -> tuple[torch.Tensor, int]:
    num_mask = math.floor(length * gamma)
    mask_index = random.sample(range(length), k=num_mask)
    masked_item_seq = seq.clone().detach()
    masked_item_seq[mask_index] = item_size + 1  # token 0 has been used for semantic masking
    return masked_item_seq, length


def _item_reorder(seq: torch.Tensor, length: int, beta: float = 0.6) -> tuple[torch.Tensor, int]:
    num_reorder = math.floor(length * beta)
    reorder_begin = random.randint(0, length - num_reorder)
    reordered_item_seq = seq.clone().detach()
    shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
    random.shuffle(shuffle_index)
    reordered_item_seq[reorder_begin : reorder_begin + num_reorder] = reordered_item_seq[shuffle_index]
    return reordered_item_seq, length


@gin.configurable
class AugmentedDataset(parquet_dataset.ParquetIterableSequentialDataSet):
    def __init__(self, augment_type: str = "cl4srec", **kwargs):
        """Provides augmented dataset from existing data.

        Args:
            augment_type: Types of augmentation to use. (ex. cl4srec, duorec, ...)
                          Only cl4srec is available for now.
        """
        super().__init__(**kwargs)
        self._augment_type = augment_type

    def _cl4srec_aug(self, seq: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """CL4SRec augmentation

        From given sequence of user logs, randomly crop, mask or reorder to produce augmented data.
        """
        aug_data = []
        aug_steps = []

        switch = random.sample(range(_NUM_AUGMENT_TYPES), k=_NUM_AUGMENTATION) if length > 1 else [3, 3]

        for i in range(_NUM_AUGMENTATION):
            match switch[i]:
                case 0:
                    aug_seq, aug_len = _item_crop(seq, length)
                case 1:
                    aug_seq, aug_len = _item_mask(seq, length, self._item_size)
                case 2:
                    aug_seq, aug_len = _item_reorder(seq, length)
                case _:
                    aug_seq, aug_len = seq, length

            aug_data.append(aug_seq)
            aug_steps.append(torch.tensor(aug_len))

        return torch.stack(aug_data), torch.stack(aug_steps)

    def _augment_data(self, seq: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._augment_type == "cl4srec":
            return self._cl4srec_aug(seq, length)

        assert False

    def _transform_input_features(self, inputs: dict, seq_len: int) -> dict:
        feature_dict = {}
        aug_seq, aug_sed_len = self._augment_data(inputs[self._target_col], min(seq_len - 1, self._max_seq_len))

        for i in range(_NUM_AUGMENTATION):
            feature_dict[f"augmented_seq_{i}"] = aug_seq[i]
            feature_dict[f"augmented_seq_len_{i}"] = aug_sed_len[i]

        return feature_dict
