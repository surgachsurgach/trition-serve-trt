import abc
import copy
import functools
import random
from typing import Any, Literal

import gin
from loguru import logger  # pylint: disable=unused-import
import numpy as np
import torch
from torch.nn import functional as F

from data.pylib.constant import recsys as common
from data.ml.model_runner.datasets import dataset_type as dt
from data.ml.model_runner.datasets import sampling_utils
from data.ml.model_runner.utils import torch_utils
from data.ml.utils import metadata

_ITEM_ID_COL = common.ITEM_ID_COL
_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_NEXT_TARGET_COL = common.NEXT_TARGET_COL
_ALL_TARGETS_COL = common.ALL_TARGETS_COL
_SEQ_LEN_COL = common.SEQ_LEN_COL
_TARGET_IDX_COL = common.TARGET_IDX_COL

_EXCLUSION_COL = common.EXCLUSION_COL


def _ensure_list(val: Any) -> list[Any]:
    if isinstance(val, list):
        return val
    return [val]


class DataTransformer(abc.ABC):
    def __init__(
        self,
        meta: metadata.Meta,
        cols: list[str] | None = None,
    ):
        if cols is None:
            cols = [_INPUTS_COL, _TARGETS_COL]

        self._cols = cols
        self._meta = meta

    @property
    def _item_size(self) -> int:
        return self._meta.get_meta_size(_ITEM_ID_COL)

    @abc.abstractmethod
    def transform_features(
        self,
        features: dict[str, str | int | list[int]],
        dataset_type: dt.DatasetType = dt.DatasetType.TRAIN_DATASET,
    ) -> dict[str, torch.Tensor]:
        pass

    def postprocess(self, row, dataset_type: dt.DatasetType = dt.DatasetType.TRAIN_DATASET):
        return row


@gin.configurable
class IdentityDataTransformer(DataTransformer):
    def transform_features(
        self,
        features: dict[str, str | int | list[int]],
        dataset_type: dt.DatasetType = dt.DatasetType.TRAIN_DATASET,
    ):
        return {col: features[col] for col in self._cols}


@gin.configurable
class SparseDataTransformer(DataTransformer):
    def __init__(self, ignore=None, **kwargs):
        super().__init__(**kwargs)
        self._ignore = ignore if ignore else []

    def transform_features(
        self,
        features: dict[str, str | int | list[int]],
        dataset_type: dt.DatasetType = dt.DatasetType.TRAIN_DATASET,
    ):
        feature_dict = {}
        for col in self._cols:
            if col not in self._ignore:
                feature = _ensure_list(features[col])
                feature_dict[col] = torch_utils.multi_hot_encoding(feature, self._item_size)
            else:
                feature_dict[col] = features[col]

        return feature_dict


@gin.configurable
class SparseWeightedDataTransformer(DataTransformer):
    def __init__(
        self,
        sparse_cols: list[str] | None = None,
        weight_cols: list[str] | None = None,
        meta_keys: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._sparse_cols = sparse_cols if sparse_cols else []
        self._weight_cols = weight_cols if weight_cols else []
        self._meta_keys = meta_keys if meta_keys else []
        assert len(self._sparse_cols) == len(self._weight_cols), "sparse_cols and weight_cols must have same length."
        assert len(self._sparse_cols) == len(self._meta_keys), "sparse_cols and meta_keys must have same length."

    def transform_features(
        self,
        features: dict[str, str | int | list[int]],
        dataset_type: dt.DatasetType = dt.DatasetType.TRAIN_DATASET,
    ):
        feature_dict = {}
        for sparse_col, weight_col, meta_key in zip(self._sparse_cols, self._weight_cols, self._meta_keys):
            if weight_col:
                feature_dict[sparse_col] = torch_utils.multi_hot_weighted_encoding(
                    features[sparse_col],
                    features[weight_col],
                    self._meta.get_meta_size(meta_key),
                )
            else:
                feature_dict[sparse_col] = torch_utils.multi_hot_encoding(
                    features[sparse_col],
                    self._meta.get_meta_size(meta_key),
                )

        for col in self._cols:
            if col not in self._sparse_cols and col not in self._weight_cols:
                feature_dict[col] = features[col]

        return feature_dict


@gin.configurable
class SamplingDataTransformer(DataTransformer):
    """Dataset for sampling."""

    def __init__(
        self,
        sampler: sampling_utils.DatasetSampler,
        ignore: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._sampler = sampler
        self._ignore = ignore if ignore else []

    def transform_features(
        self,
        features: dict[str, str | int | list[int]],
        dataset_type: dt.DatasetType = dt.DatasetType.TRAIN_DATASET,
    ):
        return self._sampler.transform({col: features[col] for col in self._cols if col not in self._ignore})

    def postprocess(self, row, dataset_type: dt.DatasetType = dt.DatasetType.TRAIN_DATASET):
        return self._sampler.postprocess(row)


@gin.configurable
class SequentialDataTransformer(DataTransformer):
    """Sequential Data Transformer for unidirectional language model training.

    TODO(fivessun): 현재 사용처가 없어서 표준적인 Unidirection Model (GPT) 기준으로 구현해두었음. 추후 사용처에 맞게 개선 혹은 상속 후 사용할 것.

    Note:
        This transformer is designed to be used with a single target sequential data column.
        In huggingface transformer models, default pad_token_id is 0.
        However, we may need to add more special token id, such as mask_token_id or so on, in the future.
        So, for consistency and convenience, the added special token id will be defined as 1 plus from the largest token id.

    Args:
        max_seq_len: Maximum number of logs to use.
        input_col: column name to generate sequential data.
        overlength_handing_method: Method to handle overlength sequence.
            - "latest": Use the latest target as the target.
            - "random": Randomly select the target from the sequence.
    """

    def __init__(
        self,
        input_col: str,
        meta_key: str,
        max_seq_len: int,
        target_col: str | None = None,
        overlength_handing_method: Literal["latest", "random"] = "random",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_col = input_col
        self._target_col = target_col
        self._meta_key = meta_key
        self._max_seq_len = max_seq_len
        self._overlength_handing_method = overlength_handing_method
        self._pad_token_id = self._target_size

    @functools.cached_property
    def _target_size(self) -> int:
        return self._meta.get_meta_size(self._meta_key)

    def _padding_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.shape[0] < self._max_seq_len:
            return F.pad(sequence, (0, self._max_seq_len - sequence.shape[0]), value=self._pad_token_id)
        return sequence

    def _extract_input_and_target(self, sequence: list) -> tuple[torch.Tensor, torch.Tensor, int]:
        seq_len = len(sequence)

        if seq_len <= self._max_seq_len:
            target_idx = seq_len - 1
        else:
            target_idx = random.randint(self._max_seq_len, seq_len - 1)

        target = torch.tensor(sequence[target_idx])
        inputs = torch.tensor(sequence[:target_idx])
        inputs = self._padding_sequence(inputs)
        return inputs, target, target_idx

    def _get_validation_targets(self, next_target: torch.Tensor, total_targets: list[int], timestamp_decay=0.01) -> dict[str, torch.Tensor]:
        total_targets.insert(0, next_target.item())
        timestamp_weights = (1 - timestamp_decay * np.arange(len(total_targets))).tolist()
        return {
            _NEXT_TARGET_COL: F.one_hot(next_target, self._target_size),
            _ALL_TARGETS_COL: torch_utils.multi_hot_weighted_encoding(total_targets, timestamp_weights, self._target_size),
        }

    def _get_train_features(self, features: dict[str, str | int | list[int]]) -> dict[str, torch.Tensor]:
        """Get Train Features"""
        sequence = copy.deepcopy(features[self._input_col])
        inputs, target, target_idx = self._extract_input_and_target(sequence)
        return {
            _INPUTS_COL: inputs,
            _TARGETS_COL: target,
            _TARGET_IDX_COL: target_idx,
        }

    def _get_dev_features(self, features: dict[str, str | int | list[int]]) -> dict[str, torch.Tensor]:
        """Get Dev Features"""
        sequence = copy.deepcopy(features[self._input_col])
        inputs, target, target_idx = self._extract_input_and_target(sequence)
        return {
            _INPUTS_COL: inputs,
            _TARGETS_COL: target,
            _TARGET_IDX_COL: target_idx,
            **self._get_validation_targets(target, features[self._target_col]),
        }

    def _get_predict_features(self, features: dict[str, str | int | list[int]]) -> dict[str, torch.Tensor]:
        """Get Dev Features"""
        sequence = copy.deepcopy(features[self._input_col])
        sequence.append(self._pad_token_id)
        inputs, _, target_idx = self._extract_input_and_target(sequence)
        return {
            _INPUTS_COL: inputs,
            _TARGET_IDX_COL: target_idx,
        }

    def transform_features(
        self,
        features: dict[str, str | int | list[int]],
        dataset_type: dt.DatasetType = dt.DatasetType.TRAIN_DATASET,
    ) -> dict[str, torch.Tensor]:
        feature_dict = {col: features[col] for col in self._cols if col not in [self._input_col, self._target_col]}
        if dataset_type == dt.DatasetType.TRAIN_DATASET:
            feature_dict.update(self._get_train_features(features))
        elif dataset_type == dt.DatasetType.DEV_DATASET:
            feature_dict.update(self._get_dev_features(features))
        elif dataset_type == dt.DatasetType.PREDICT_DATASET:
            feature_dict.update(self._get_predict_features(features))
        return feature_dict


@gin.configurable
class MaskedSequentialDataTransformer(SequentialDataTransformer):
    """Masked Sequential Data Transformer for bidirectional language model training."""

    _IGNORE_INDEX = -100  # torch.nn.CrossEntropyLoss's default value for ignore_index

    def __init__(self, mask_prob: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self._mask_prob = mask_prob
        self._mask_token_id = self._target_size + 1

    # pylint: disable=arguments-differ
    def _extract_input_and_target(self, sequence: list, mask_last: bool) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = len(sequence)
        if seq_len <= self._max_seq_len:
            end_idx = seq_len - 1
            start_idx = 0
        else:
            end_idx = random.randint(self._max_seq_len, seq_len - 1)
            start_idx = end_idx - self._max_seq_len + 1

        targets = torch.tensor([self._IGNORE_INDEX] * self._max_seq_len)

        if mask_last:
            mask_pos = np.array([min(end_idx, self._max_seq_len - 1)]).tolist()
        else:
            num_masks = max(1, int(min(seq_len, self._max_seq_len) * self._mask_prob))
            mask_pos = np.random.choice(min(seq_len, self._max_seq_len), num_masks, replace=False).tolist()

        inputs = torch.tensor(sequence[start_idx : end_idx + 1])
        inputs = self._padding_sequence(inputs)

        targets[mask_pos] = inputs[mask_pos]
        inputs[mask_pos] = self._mask_token_id

        return inputs, targets

    def _get_train_features(self, features: dict[str, str | int | list[int]]) -> dict[str, torch.Tensor]:
        """Get Train Features

        For training, we mask random positions in the sequence and set the target to the original value.
        """
        sequence = copy.deepcopy(features[self._input_col])
        inputs, targets = self._extract_input_and_target(sequence, mask_last=False)
        return {
            _INPUTS_COL: inputs,
            _TARGETS_COL: targets,
        }

    def _get_dev_features(self, features: dict[str, str | int | list[int]]) -> dict[str, torch.Tensor]:
        """Get Dev Features

        For validation, we mask the last position in the sequence and set the target to the original value.
        Since length of the sequence is not fixed, last position is also not fixed. So, target_idx is used.

        To estimate (ndcg) metric, we need to know the next target and all targets.
        """
        sequence = copy.deepcopy(features[self._input_col])

        inputs, targets = self._extract_input_and_target(sequence, mask_last=True)
        target_idx = torch.tensor(min(len(sequence) - 1, self._max_seq_len - 1))
        return {
            _INPUTS_COL: inputs,
            _TARGETS_COL: targets,
            _TARGET_IDX_COL: target_idx,
            **self._get_validation_targets(targets[target_idx], features[self._target_col]),
        }

    def _get_predict_features(self, features: dict[str, str | int | list[int]]) -> dict[str, torch.Tensor]:
        """Get Predict Features

        To prevent masking the last position in the sequence, we add pad token at the end of the sequence.
        """
        sequence = copy.deepcopy(features[self._input_col])
        sequence.append(self._pad_token_id)  # Append pad token which is removed after extracting inputs and targets

        inputs, _ = self._extract_input_and_target(sequence, mask_last=True)
        target_idx = torch.tensor(min(len(sequence) - 1, self._max_seq_len - 1))
        return {
            _INPUTS_COL: inputs,
            _TARGET_IDX_COL: target_idx,
            _EXCLUSION_COL: torch_utils.multi_hot_encoding(features[self._input_col], self._item_size),
        }
