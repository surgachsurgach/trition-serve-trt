from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

import gin
from loguru import logger

from data.ml.utils import file_utils

DEFAULT_FILENAME = "meta.json"
ITEM_ID_PREFIX_DELIM = "/"


def _read_file(filepath: str) -> Dict:
    fs = file_utils.get_filesystem(filepath)
    with fs.open(filepath) as file:
        return json.load(file)


@gin.configurable
class Meta:
    def __init__(self, metadata: Optional[Dict] = None, filepath: Optional[str] = None):
        self._metadata = metadata if metadata else {}

        if filepath:
            logger.info(f"Loading meta from {filepath}.")
            self._metadata = _read_file(filepath)

        self._split_item_id()

    @property
    def metadata(self):
        return self._metadata

    def _split_item_id(self):
        if "item_id" in self._metadata:
            self._metadata["item_id_prefix"] = [
                item_id.split(ITEM_ID_PREFIX_DELIM)[0] if ITEM_ID_PREFIX_DELIM in item_id else "" for item_id in self._metadata["item_id"]
            ]
            self._metadata["item_id"] = [
                item_id.split(ITEM_ID_PREFIX_DELIM)[1] if ITEM_ID_PREFIX_DELIM in item_id else item_id
                for item_id in self._metadata["item_id"]
            ]

    def add_meta(self, key: str, value: Any):
        if key in self._metadata:
            raise RuntimeError("Key already exists in metadata.")
        self._metadata[key] = value

    def update_meta(self, key: str, value: Any):
        self._metadata[key] = value

    def get_meta_size(self, key: str, cond: Callable[..., bool] | None = None) -> int:
        if key in self._metadata and self._metadata[key]:
            if cond is None:
                return len(self._metadata[key])

            return sum(1 for v in self._metadata[key] if cond(v))
        return 0

    def get_meta(self, key: str, out_type: Optional[Callable] = None) -> List[Union[int, str]]:
        if key not in self._metadata:
            raise RuntimeError("Key does not exist in metadata.")

        if out_type:
            return list(map(out_type, self._metadata[key]))
        return self._metadata[key]

    def get_idx_to_id(self, key: str, out_type=None) -> Dict[int, Union[int, str]]:
        meta = self.get_meta(key, out_type)
        return dict(enumerate(meta))

    def get_id_to_idx(self, key, out_type=None) -> Dict[Union[int, str], int]:
        meta = self.get_meta(key, out_type)
        return {id_: idx for idx, id_ in enumerate(meta)}

    def get_start_idx(self, key: str, cond: Callable[..., bool]) -> int | None:
        meta = self.get_meta(key, str)
        for i, id_ in enumerate(meta):
            if cond(id_):
                return i
        return None

    def save(self, path: str, filename: str = DEFAULT_FILENAME):
        filepath = os.path.join(path, filename)
        logger.info(f"Writing meta to {filepath}.")

        fs = file_utils.get_filesystem(filepath)
        if fs.exists(filepath) and fs.isfile(filepath):
            fs.rm(filepath)

        with fs.open(filepath, "w") as file:
            json.dump(self._metadata, file, indent=4)

    def __eq__(self, other: object):
        if isinstance(other, Meta):
            return self._metadata == other._metadata
        return False

    @classmethod
    def load(cls, path: str, filename: str = DEFAULT_FILENAME) -> Meta:
        filepath = os.path.join(path, filename)
        logger.info(f"Loading meta from {filepath}.")
        return cls(metadata=_read_file(filepath))

    @classmethod
    def from_json(cls, json_obj: Dict):
        return cls(metadata=json_obj)

    def reset(self):
        self._metadata = {}
