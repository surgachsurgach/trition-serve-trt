from itertools import groupby
import pathlib
from typing import Any, List, Union
from urllib import parse

import fsspec
import s3fs


def get_path(filepath: str) -> str:
    parsed = parse.urlparse(filepath)
    return parsed.path


def get_scheme(filepath: str) -> str:
    parsed = parse.urlparse(filepath)
    scheme = parsed.scheme

    if not scheme:
        scheme = "file"

    if scheme != "file" and not is_s3_scheme(scheme):
        raise ValueError(f"Unsupported scheme `{scheme}`")

    return scheme


def is_s3_scheme(scheme: str):
    return scheme and scheme in ("s3", "s3a", "s3n")


def has_all_same_scheme(filepaths: List[str]) -> bool:
    grp = groupby([get_scheme(filepath) for filepath in filepaths])
    return next(grp, True) and not next(grp, False)


def get_filesystem(path: Union[str, pathlib.Path], **kwargs: Any) -> fsspec.AbstractFileSystem:
    fs, _ = fsspec.core.url_to_fs(str(path), **kwargs)
    return fs


def copy_from_s3(s3_path: str, local_path: str):
    s3 = s3fs.S3FileSystem(anon=False)
    s3.get_file(s3_path, local_path)
    return local_path
