import os
from typing import Callable, Dict, Generator, List

from loguru import logger
import numpy as np
import pandas as pd
import pyarrow
from pyarrow import parquet

from data.ml.utils import file_utils


def make_parquet_file(data_: Dict, filepath: str):
    df = pd.DataFrame(data=data_)
    table = pyarrow.Table.from_pandas(df)
    parquet.write_table(table, filepath)


def get_parquet_files(parquet_dir: str, shuffle: bool, seed: int | None = None) -> list[str]:
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


def get_num_rows(files: List[str]) -> int:
    num_rows = 0
    for file in files:
        parquet_file = parquet.ParquetFile(file)
        num_rows += parquet_file.metadata.num_rows

    logger.info(f"rows: {num_rows}")
    return num_rows


def get_filtered_num_rows(files: List[str], col: str, filter_fn: Callable[..., bool]) -> int:
    num_rows = 0
    logger.info("counting")
    for file in files:
        parquet_file = parquet.ParquetFile(file)
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx)
            for row in row_group[col]:
                if filter_fn(row.as_py()):
                    num_rows += 1

    logger.info(f"rows: {num_rows}")
    return num_rows


def iter_rows(parquet_files: List[str]) -> Generator:
    for file in parquet_files:
        parquet_file = parquet.ParquetFile(file)
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx)
            for row in row_group.to_pylist():
                yield row
