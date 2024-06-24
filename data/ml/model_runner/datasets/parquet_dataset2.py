import abc
import functools
import itertools
from typing import Any, Generator, Iterator

import gin
from loguru import logger
import numpy as np
import torch
from torch.utils import data

from data.ml.model_runner.datasets import data_transformer
from data.ml.model_runner.datasets import dataset_type as dt
from data.ml.model_runner.datasets import parquet_utils


def _pick(buf: list, rng: np.random.Generator = np.random.default_rng()) -> Any:
    k = rng.integers(0, len(buf))
    return buf.pop(k)


def _split_by_gpu_and_worker(inputs: list, cur_gpu: int, num_gpus: int, cur_worker: int, num_workers: int):
    assert cur_worker < num_workers, "cur_worker must be less than num_workers."
    assert cur_gpu < num_gpus, "cur_gpu must be less than num_gpus."
    return itertools.islice(inputs, cur_gpu + num_gpus * cur_worker, None, num_gpus * num_workers)


class DatasetBase(abc.ABC):
    def __init__(
        self,
        transformer: data_transformer.DataTransformer,
        parquet_dir: str,
        shuffle: bool = True,
        seed: int = 0,
        dataset_type: dt.DatasetType = dt.DatasetType.TRAIN_DATASET,
    ):
        assert parquet_dir, "`parquet_dir` is not provided."
        assert isinstance(parquet_dir, str), "`parquet_dir` should be str."
        assert transformer, "`transformer` is not provided."

        self._transformer = transformer
        self._parquet_dir = parquet_dir
        self._shuffle = shuffle
        self._seed = seed
        self._dataset_type = dataset_type

        logger.info(f"Data Transformer: {transformer.__class__.__name__}")
        logger.info(f"Dataset Type: {dataset_type}")

    @functools.cached_property
    def _parquet_files(self) -> list[str]:
        return parquet_utils.get_parquet_files(self._parquet_dir, self._shuffle, self._seed)

    @functools.cached_property
    def _total_len(self) -> int:
        return parquet_utils.get_num_rows(self._parquet_files)

    @property
    def seed(self) -> int:
        return self._seed


@gin.configurable
class ParquetDataset2(data.Dataset, DatasetBase):
    def __init__(self, cache_transformed_data: bool = False, **kwargs):
        super().__init__(**kwargs)

        self._cache_transformed_data = cache_transformed_data
        logger.info(f"Cache Transformed Data: {cache_transformed_data}")

    @functools.cached_property
    def data(self) -> list:
        logger.info(f"Preloading {self._dataset_type} starts.")
        if self._cache_transformed_data:
            rows = []
            for row in parquet_utils.iter_rows(self._parquet_files):
                row = self._transformer.transform_features(row, self._dataset_type)
                if row:
                    rows.append(row)
        else:
            rows = list(parquet_utils.iter_rows(self._parquet_files))
        logger.info(f"Preloading {self._dataset_type} Done.")

        return rows

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self._cache_transformed_data:
            row = self.data[idx]
        else:
            row = self._transformer.transform_features(self.data[idx], self._dataset_type)
        return self._transformer.postprocess(row, self._dataset_type)


@gin.configurable
class ParquetIterableDataset2(data.IterableDataset, DatasetBase):
    def __init__(self, shuffle_buffer_size: int = 10000, **kwargs):
        super().__init__(**kwargs)
        self._shuffle_buffer_size = shuffle_buffer_size

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
            features = self._transformer.transform_features(row)
            if features:
                yield features

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
