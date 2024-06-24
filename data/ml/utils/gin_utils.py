import datetime
import os
from typing import Dict, List

import gin


@gin.configurable
def join_paths(base_path: str, sub_paths: List[str], sub_path_rows: List[str] | None = None) -> str | List[str]:
    """Generate concatenated path string from given parameters.
        base_path and sub_paths are concatenated to return generated path.
        If sub_path_rows are given, each path in sub_path_rows is concatenated to (base_path + sub_paths)
        and returns list of paths.

        ex) base_path: "a", sub_paths: ["b", "c", "d"]
         -> "a/b/c/d"
        ex2) base_path: "a", sub_paths: ["b", "c", "d"], sub_path_rows = ["1", "2", "3"]
         -> ["a/b/c/d/1", "a/b/c/d/2", "a/b/c/d/3"]

    Args:
        base_path: base path
        sub_paths: list of string to expand base path
        sub_path_rows: rows of sub_path where each row is concatenated with given base_path and sub_paths (returns list of paths)

    Returns:
        Path string if sub_path_rows is None,
        List of path strings if sub_path_rows are given.
    """
    if sub_path_rows is None:
        return os.path.join(base_path, *sub_paths)
    return [os.path.join(os.path.join(base_path, *sub_paths), sub) for sub in sub_path_rows]


@gin.configurable
def join_partitions(partitions: Dict[str, str | list[str]]) -> str:
    """Generate concatenated partition string from given parameters.
    Args:
        partitions: dictionary containing partition key and value pairs.

    Returns:
        Partition string. e.g. "date=2023-01-30/genre=general"
    """

    return concat_strings(
        [f"{k}={','.join(sorted(v))}" if isinstance(v, list) else f"{k}={v}" for k, v in partitions.items()], delimiter="/"
    )


@gin.configurable
def create_date_paths(partition: str, end_date: str, days_to_collect: int = 1, date_pattern="%Y-%m-%d") -> List[str]:
    """Generates list of partitions containing multiple dates based on given partition and end_date.
        Date range in result partition is [end_date - (days_to_collect - 1) days, end_date].

        ex) partition: "date=2023-01-30/genre=general", end_date: "2023-01-30", days_to_collect: 2
          -> ["date=2023-01-30/genre=general", "date=2023-01-29/genre=general"]

    Args:
        partition: string containing prefix, suffix and end_date.
        end_date: base date where date generation starts. (past dates relative to end_date are generated.)
        days_to_collect: number of days to collect (subtracted from end_date)
        date_pattern: date format of end_date

    Returns:
        List of partition strings containing multiple dates. (end_date is included)
    """
    assert end_date in partition

    prefix, suffix = partition.split(end_date)
    end_dt = datetime.datetime.strptime(end_date, date_pattern)
    return [
        "".join([prefix, datetime.datetime.strftime(end_dt - datetime.timedelta(days=days), date_pattern), suffix])
        for days in range(days_to_collect)
    ]


@gin.configurable
def concat_strings(strings: List[str], delimiter: str = "") -> str:
    return delimiter.join(strings)


def parse_gin_file(gin_file: str) -> list[str]:
    """Parses given gin_file into list."""
    assert gin_file, "GIN_FILE is not set."
    return [gin_file]


def parse_gin_params(gin_params: str) -> list[str]:
    """Parses given gin_params string into dictionary.

    Args:
        gin_params: gin_params string where each parameter is separated by "|".

    Returns:
        List of gin_params. e.g. ["TABLE_NAME='table_name'", "PARTITION='2023-11-22'"]
    """
    assert gin_params, "GIN_PARAMS is not set."
    return gin_params.split("|") if gin_params is not None else []
