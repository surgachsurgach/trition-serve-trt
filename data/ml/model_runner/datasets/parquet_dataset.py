import abc
import functools
import itertools
import os
import random
from typing import Any, Callable, Generator, Iterator, Optional

import gin
from loguru import logger
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils import data

from data.pylib.constant import recsys as common
from data.ml.model_runner.datasets import parquet_utils
from data.ml.model_runner.utils import torch_utils
from data.ml.utils import file_utils
from data.ml.utils import metadata
from data.ml.utils import token_utils

_USER_ID_COL = common.USER_ID_COL
_USER_INPUTS_COL = common.USER_INPUTS_COL
_ITEM_ID_COL = common.ITEM_ID_COL
_ITEM_INPUTS_COL = common.ITEM_INPUTS_COL
_INPUTS_COL = common.INPUTS_COL
_ALL_TARGETS_COL = common.ALL_TARGETS_COL
_TARGETS_COL = common.TARGETS_COL
_NEXT_TARGET_COL = common.NEXT_TARGET_COL
_SEQ_LEN_COL = common.SEQ_LEN_COL


def _pick(buf: list, rng: np.random.Generator = np.random.default_rng()) -> Any:
    k = rng.integers(0, len(buf))
    return buf.pop(k)


def _ensure_list(val: Any) -> list[Any]:
    if isinstance(val, list):
        return val
    return [val]


def _get_parquet_files(parquet_dir: str, shuffle: bool, seed: Optional[int] = None) -> list[str]:
    filesystem = file_utils.get_filesystem(parquet_dir)
    logger.info(f"Load parquet files from {parquet_dir}.")
    parquet_files = sorted(filesystem.glob(os.path.join(parquet_dir, "*.parquet")))

    logger.info(f"Loaded {len(parquet_files)} files.")

    if not parquet_files:
        raise ValueError(f"parquet_dir {parquet_dir} does not have parquet files.")

    if not file_utils.has_all_same_scheme(parquet_files):
        raise ValueError("`parquet_files` do not have the same schema.")

    if shuffle:
        np.random.default_rng(seed).shuffle(parquet_files)

    return [filesystem.unstrip_protocol(parquet_file) for parquet_file in parquet_files]


def _split_by_gpu_and_worker(inputs: list, cur_gpu: int, num_gpus: int, cur_worker: int, num_workers: int):
    assert cur_worker < num_workers, "cur_worker must be less than num_workers."
    assert cur_gpu < num_gpus, "cur_gpu must be less than num_gpus."
    return itertools.islice(inputs, cur_gpu + num_gpus * cur_worker, None, num_gpus * num_workers)


class ParquetIterableDataSet(data.IterableDataset, abc.ABC):
    def __init__(
        self,
        parquet_dir: str,
        cols=None,
        shuffle_buffer_size: int = 10000,
        shuffle: bool = True,
        seed: Optional[int] = None,
        collate_fn: Callable | None = None,
        **kwargs,
    ):
        """Torch IterableDataSet for loading parquet files.

        If DataLoader is set to use multiple workers, files in parquet_dir are split into N groups where N is
        the number of workers. Each worker reads parquet rows from the files in the group and returns transformed rows.

        file1 --+
        file2 --+ --> shuffle buffer  --> data loader --> model
        ...
        fileN --+

        Args:
            parquet_dir: Directory path of parquet files. Files in this directory which end with
                `.parquet` are used.
            cols: Column names to retrieve from parquet table. Default: ['inputs', 'targets'].
            shuffle_buffer_size: If shuffle is True, this number of parquet rows is loaded to shuffle buffer.
                If shuffle is False, shuffle_buffer_size is not used.
            shuffle: If True, data are shuffled before feeding to a model.
            seed: A seed to initialize numpy's BitGenerator.
        """
        assert parquet_dir, "`parquet_dir` is not provided."
        assert isinstance(parquet_dir, str), "`parquet_dir` should be str."

        if cols is None:
            cols = [_INPUTS_COL, _TARGETS_COL]

        self._parquet_dir = parquet_dir
        self._shuffle_buffer_size = shuffle_buffer_size
        self._shuffle = shuffle
        self._cols = cols
        self._seed = seed
        self._collate_fn = collate_fn

    @functools.cached_property
    def _parquet_files(self):
        return _get_parquet_files(self._parquet_dir, self._shuffle, self._seed)

    @functools.cached_property
    def _total_len(self):
        return parquet_utils.get_num_rows(self._parquet_files)

    @property
    def seed(self):
        return self._seed

    @property
    def collate_fn(self) -> Callable | None:
        return self._collate_fn

    def __len__(self):
        if len(self._parquet_files) == 1:  # dev case
            # TODO(swkang): currently dev dataset has only 1 file,
            # therefore distribution is not necessary.
            # Fix this when dev dataset has multiple files.
            return self._total_len

        if torch.distributed.is_initialized():
            num_gpus = torch.distributed.get_world_size()
        else:
            num_gpus = 1

        return self._total_len // num_gpus

    def _split_by_worker(self) -> Generator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            if torch.distributed.is_initialized():
                cur_gpu = torch.distributed.get_rank()
                num_gpus = torch.distributed.get_world_size()
            else:
                cur_gpu = 0
                num_gpus = 1

            for parquet_file in _split_by_gpu_and_worker(self._parquet_files, cur_gpu, num_gpus, worker_info.id, worker_info.num_workers):
                yield parquet_file

        else:
            for parquet_file in self._parquet_files:
                yield parquet_file

    def _shuffle_rows(self, rows: Iterator) -> Generator:
        rng = np.random.default_rng(self._seed)
        buffer = []
        for row in rows:
            buffer.append(row)
            if len(buffer) < self._shuffle_buffer_size:
                try:
                    buffer.append(next(rows))
                except StopIteration:
                    pass
            if len(buffer) >= self._shuffle_buffer_size:
                yield _pick(buffer, rng)
        while len(buffer) > 0:
            yield _pick(buffer, rng)

    def __iter__(self):
        files_for_worker = list(self._split_by_worker())

        if self._shuffle:
            # For each epoch, the list of dataset files are shuffled
            # so that rows are randomly fed.
            rng = np.random.default_rng(self._seed)
            rng.shuffle(files_for_worker)
            rows = self._shuffle_rows(parquet_utils.iter_rows(files_for_worker))
        else:
            rows = parquet_utils.iter_rows(files_for_worker)

        for row in rows:
            features = self._transform_features(row)
            if features:
                yield features

    @abc.abstractmethod
    def _transform_features(self, features: dict[str, int | list[int]]) -> dict[str, torch.Tensor]:
        """Transform features."""


@gin.configurable
class ParquetIterableSparseDataSet(ParquetIterableDataSet):
    def __init__(self, meta: metadata.Meta, ignore: list[str] = None, **kwargs):
        """Parquet data loader for space datasets.

        Args:
            item_size: The number of items in dataset.
            inputs_col_name: The `inputs` column name. Default: 'inputs'.
        """
        super().__init__(**kwargs)
        self._meta = meta
        self._ignore = ignore if ignore else []

    @property
    def _item_size(self) -> int:
        return self._meta.get_meta_size(_ITEM_ID_COL)

    def _transform_features(self, features):
        feature_dict = {}
        for col in self._cols:
            if col not in self._ignore:
                feature = _ensure_list(features[col])
                feature_dict[col] = torch_utils.multi_hot_encoding(feature, self._item_size)
            else:
                feature_dict[col] = features[col]

        return feature_dict


class ParquetIterableIdentityDataSet(ParquetIterableDataSet):
    def _transform_features(self, features):
        feature_dict = {}
        for col in self._cols:
            feature_dict[col] = features[col]

        return feature_dict


@gin.configurable
class ParquetIterableSequentialDataSet(ParquetIterableDataSet):
    def __init__(
        self,
        max_seq_len: int,
        reserved_labels: dict[str, int] | None = None,
        ignore: Optional[list[str]] = None,
        meta: Optional[metadata.Meta] = None,
        target_col: str = _ITEM_ID_COL,
        **kwargs,
    ):
        """Parquet data loader for sequential datasets.

        Values in each column are already replaced as index values.

        Args:
            max_seq_len: Maximum number of logs to use.
            reserved_labels: number of labels to reserve for features.
                Parameter must be dict object having feature as key and number of labels to reserve as value.
                ex) { "col1": 1, "col2": 2, "col3": 1 }
            ignore: cols to ignore from feature generation

            args for dev purpose:
            item_size: The number of items in dataset.
            target_col: column name to generate data for dev metrics
        """

        super().__init__(**kwargs)
        self._max_seq_len = max_seq_len
        self._reserved_labels = reserved_labels if reserved_labels else {}
        self._ignore = set(ignore) if ignore else set()
        self._cols_in_use = set(self._cols) - self._ignore
        self._meta = meta
        self._target_col = target_col

    @property
    def _item_size(self) -> int:
        if not self._meta:
            return 0

        item_size = self._meta.get_meta_size(self._target_col)
        if self._target_col in self._reserved_labels:
            return item_size + self._reserved_labels[self._target_col]
        return item_size

    def _trunc_or_pad(self, feature: list[int], pad_token_id: int = 0) -> torch.Tensor:
        if len(feature) > self._max_seq_len:  # truncate
            feature = feature[-self._max_seq_len :]
            return torch.tensor(feature)
        elif len(feature) == self._max_seq_len:
            return torch.tensor(feature)
        else:  # pad
            tensor = torch.tensor(feature)
            return F.pad(tensor, (pad_token_id, self._max_seq_len - len(tensor)))  # pylint: disable=not-callable

    def _apply_reserved_label(self, col, feature):
        if col in self._reserved_labels:
            if isinstance(feature, list):
                return [f + self._reserved_labels[col] for f in feature]
            return feature + self._reserved_labels[col]
        return feature

    def _extract_target(self, features: dict[str, str | int | list[int]], seq_len: int) -> tuple[dict[str, torch.Tensor], int]:
        target = {}

        if seq_len <= self._max_seq_len:
            target_idx = seq_len - 1
        else:
            target_idx = random.randint(self._max_seq_len, seq_len - 1)

        for col in self._cols_in_use:
            feature = _ensure_list(features[col])
            feature = self._apply_reserved_label(col, feature[target_idx])
            target[col] = torch.tensor(feature)

        return target, target_idx

    def _extract_inputs(self, features: dict[str, str | int | list[int]], target_idx: int) -> dict[str, torch.Tensor]:
        inputs = {}

        for col in self._cols_in_use:
            feature = _ensure_list(features[col])
            feature = self._apply_reserved_label(col, feature[:target_idx])
            inputs[col] = self._trunc_or_pad(feature)

        return inputs

    def _transform_dev_features(self, features: dict[str, str | int | list[int]], target_idx: int) -> dict[str, torch.Tensor]:
        dev_feature_dict = {}

        feature = _ensure_list(features[self._target_col])
        feature = self._apply_reserved_label(self._target_col, feature)

        dev_feature_dict[_ALL_TARGETS_COL] = torch_utils.multi_hot_encoding(feature, self._item_size)
        dev_feature_dict[_NEXT_TARGET_COL] = F.one_hot(torch.tensor(feature[target_idx]), self._item_size)  # pylint: disable=not-callable
        return dev_feature_dict

    def _transform_input_features(self, inputs: dict, seq_len: int) -> dict[str, str | int | list[int]]:
        return {}

    def _transform_features(self, features: dict[str, str | int | list[int]]) -> dict[str, torch.Tensor]:
        feature_dict = {}
        total_seq_len = len(_ensure_list(features[list(self._cols_in_use)[0]]))

        target, target_idx = self._extract_target(features, total_seq_len)

        if not target_idx:
            return feature_dict

        inputs = self._extract_inputs(features, target_idx)
        input_feature_dict = self._transform_input_features(inputs, total_seq_len)
        inputs.update(**input_feature_dict)

        feature_dict[_SEQ_LEN_COL] = torch.tensor(min(total_seq_len - 1, self._max_seq_len))
        feature_dict[_INPUTS_COL] = inputs
        feature_dict[_TARGETS_COL] = target

        for col in self._ignore:
            feature_dict[col] = features[col]

        # for dev dataset
        if self._item_size and self._target_col:
            dev_feature_dict = self._transform_dev_features(features, target_idx)
            feature_dict.update(**dev_feature_dict)

        return feature_dict


@gin.configurable
class ParquetIterableSequentialGroupedDataSet(ParquetIterableSequentialDataSet):
    def __init__(self, target_genre: str, **kwargs):
        super().__init__(**kwargs)

        self._target_genre = target_genre
        self._target_genre_idx = self._get_genre_idx(target_genre)

    def _get_genre_idx(self, target_genre: str) -> int:
        assert self._meta

        genres = self._meta.get_id_to_idx("item_genre")
        assert target_genre in genres
        return genres[target_genre]

    def _filter_target(self, features: list[int]) -> np.ndarray:
        target_indices = np.where(np.array(_ensure_list(features)) == self._target_genre_idx)[0]
        return target_indices[target_indices >= 1]

    @functools.cached_property
    def _total_len(self):
        return parquet_utils.get_filtered_num_rows(self._parquet_files, "item_genre", lambda x: len(self._filter_target(x)) > 0)

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

    def _extract_target(self, features: dict[str, str | int | list[int]], seq_len: int) -> tuple[dict[str, torch.Tensor], int]:
        target = {}
        target_indices = self._filter_target(features["item_genre"])

        if len(target_indices) < 1:
            return target, None

        target_idx = random.choice(target_indices)

        for col in self._cols_in_use:
            feature = _ensure_list(features[col])
            feature = self._apply_reserved_label(col, feature[target_idx])
            target[col] = torch.tensor(feature)

        return target, target_idx

    def _transform_dev_features(self, features: dict[str, str | int | list[int]], target_idx: int) -> dict[str, torch.Tensor]:
        dev_feature_dict = {}

        feature = _ensure_list(features[self._target_col])
        feature = self._apply_reserved_label(self._target_col, feature)

        dev_feature_dict[_NEXT_TARGET_COL] = F.one_hot(  # pylint: disable=not-callable
            torch.tensor(feature[target_idx] - self._target_item_start_idx),
            self._target_item_size,
        )
        return dev_feature_dict


@gin.configurable
class ParquetIterableMaskedDataset(ParquetIterableSequentialDataSet):
    _IGNORE_INDEX = -100  # torch.nn.CrossEntropyLoss's default value for ignore_index

    def __init__(
        self,
        mask_last: bool = False,
        mask_prob: float = 0.2,
        pad_token_id: int = token_utils.DEFAULT_PAD_TOKEN_ID,
        is_dev_dataset: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mask_last = mask_last
        self._mask_token_id = self._item_size  # reserved labels are already added
        self._pad_token_id = pad_token_id
        self._mask_prob = mask_prob
        self._is_dev_dataset = is_dev_dataset

    def _extract_input_and_target(
        self,
        features: dict[str, str | int | list[int]],
        seq_len: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        start_idx = 0
        if seq_len <= self._max_seq_len:
            target_idx = seq_len - 1
        else:
            target_idx = random.randint(self._max_seq_len, seq_len - 1)
            start_idx = target_idx - self._max_seq_len + 1

        targets = torch.tensor([ParquetIterableMaskedDataset._IGNORE_INDEX] * self._max_seq_len)

        if self._mask_last:
            mask_pos = np.array([min(target_idx, self._max_seq_len - 1)]).tolist()
        else:
            num_masks = int(max(1, min(seq_len, self._max_seq_len) * self._mask_prob))
            mask_pos = np.random.choice(min(seq_len, self._max_seq_len), num_masks, replace=False).tolist()

        inputs = {}
        for col in self._cols_in_use:
            feature = _ensure_list(features[col])
            feature = self._apply_reserved_label(col, feature[start_idx : target_idx + 1])
            feature = self._trunc_or_pad(feature, pad_token_id=self._pad_token_id)

            if col == self._target_col:
                targets[mask_pos] = feature[mask_pos]

            feature[mask_pos] = self._mask_token_id
            inputs[col] = feature

        return inputs, targets

    def _transform_features(self, features: dict[str, str | int | list[int]]) -> dict[str, torch.Tensor]:
        feature_dict = {}
        total_seq_len = len(_ensure_list(features[list(self._cols_in_use)[0]]))

        inputs, targets = self._extract_input_and_target(features, total_seq_len)

        feature_dict[_SEQ_LEN_COL] = torch.tensor(min(total_seq_len - 1, self._max_seq_len - 1))
        feature_dict[_INPUTS_COL] = inputs
        feature_dict[_TARGETS_COL] = targets

        for col in self._ignore:
            feature_dict[col] = features[col]

        if self._is_dev_dataset:
            feature_dict[_NEXT_TARGET_COL] = F.one_hot(  # pylint: disable=not-callable
                feature_dict[_TARGETS_COL][feature_dict[_SEQ_LEN_COL]], self._item_size
            )

            if self._target_col in self._reserved_labels:
                idx = self._reserved_labels[self._target_col]
                feature_dict[_NEXT_TARGET_COL] = feature_dict[_NEXT_TARGET_COL][idx:]

        return feature_dict
