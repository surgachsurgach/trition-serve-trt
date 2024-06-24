import abc
import functools
from typing import Any

import gin
from loguru import logger
import numpy as np
import torch

from data.pylib.constant import recsys as common
from data.ml.utils import metadata

_ITEM_ID_COL = common.ITEM_ID_COL

_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_POSITIVE_CONTEXTS_COL = common.POSITIVE_CONTEXTS_COL
_NEGATIVE_CONTEXTS_COL = common.NEGATIVE_CONTEXTS_COL

_ITEM_FREQUENCY_META_KEY = common.ITEM_FREQUENCY_META_KEY


class DatasetSampler(abc.ABC):
    def __init__(self, **kwargs):
        """Base class for sampler."""
        logger.info(f"{self.__class__.__name__} configurations:")

    @abc.abstractmethod
    def transform(self, features: dict[str, Any]) -> dict[str, Any]:
        """Transform inputs."""

    def postprocess(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Post process from inputs."""
        return row


@gin.configurable
class NegativeSampler(DatasetSampler):
    def __init__(
        self,
        meta: metadata.Meta,
        frequency_exponent: float = 0.75,
        num_negative_samples: int = 20,
        **kwargs,
    ):
        """Negative-sampler based on item frequency.

        Args:
            meta: Metadata. It should have `item frequency` and `item size`.
            frequency_exponent: Exponent of item frequency.
            num_negative_samples: Number of negative samples.
        """
        super().__init__(**kwargs)
        self._meta = meta
        self._frequency_exponent = frequency_exponent
        self._num_negative_samples = num_negative_samples

        logger.info(f"  item_size: {self._item_size}")
        logger.info(f"  frequency_exponent: {self._frequency_exponent}")
        logger.info(f"  num_negative_samples: {self._num_negative_samples}")

    @functools.cached_property
    def _item_size(self):
        return self._meta.get_meta_size(_ITEM_ID_COL)

    @functools.cached_property
    def _frequency(self):
        return self._meta.get_meta(_ITEM_FREQUENCY_META_KEY)

    @functools.cached_property
    def _powered_frequency(self) -> np.ndarray:
        """Powered frequency of items.

        In paper, uniform distribution of 0.75 powered frequency shows the best performance,
        but for us, we use 0.5 powered frequency to alleviate the gap between high and low frequency items.
        """
        return np.array(self._frequency) ** self._frequency_exponent

    @functools.cached_property
    def _total_item_array(self) -> np.ndarray:
        return np.array(list(range(self._item_size)))

    def _get_negative_samples(self, negative_item_indices: np.ndarray, size: int | tuple[int, ...]) -> torch.Tensor:
        frequency = self._powered_frequency[negative_item_indices]
        p = frequency / np.sum(frequency)
        return torch.from_numpy(np.random.choice(negative_item_indices, size=size, p=p))

    def transform(self, features: dict[str, str | int | list[int]]) -> dict[str, torch.Tensor]:
        inputs = np.array(features[_INPUTS_COL])
        negative_item_indices = np.setdiff1d(self._total_item_array, inputs)
        return {
            _TARGETS_COL: torch.tensor(features[_TARGETS_COL]),
            _POSITIVE_CONTEXTS_COL: torch.tensor(features[_POSITIVE_CONTEXTS_COL]),
            _NEGATIVE_CONTEXTS_COL: torch.from_numpy(negative_item_indices),
        }

    def postprocess(self, row):
        negative_item_indices = row.pop(_NEGATIVE_CONTEXTS_COL)
        targets_len = row[_TARGETS_COL].size(0)
        size = (targets_len, self._num_negative_samples)
        row[_NEGATIVE_CONTEXTS_COL] = (
            self._get_negative_samples(negative_item_indices, size)
            if negative_item_indices.nelement()
            else torch.empty((0, 0), dtype=torch.int)
        )
        return row


@gin.configurable
class SkipGramNegativeSampler(NegativeSampler):
    def __init__(
        self,
        discard_frequency_threshold: float | None = 0.8,
        max_discard_probability: float = 1.0,
        max_length: int | None = None,
        **kwargs,
    ):
        """Skip-Gram Negative-sampler based on item frequency.

        Args:
            max_length: Maximum length of sequence.
            discard_frequency_threshold: Frequency threshold for discarding items.
            max_discard_probability: Maximum probability of discarding an item.
        """
        super().__init__(**kwargs)
        self._max_length = max_length
        self._discard_frequency_threshold = discard_frequency_threshold
        self._max_discard_probability = max_discard_probability

        logger.info(f"  max_length: {self._max_length}")
        logger.info(f"  discard_frequency_threshold: {self._discard_frequency_threshold}")
        logger.info(f"  max_discard_probability: {self._max_discard_probability}")
        logger.info(f"  rho: {self._rho}")

    @functools.cached_property
    def _frequency_ratio(self):
        return np.array(self._frequency) / np.sum(self._frequency)

    @functools.cached_property
    def _rho(self) -> float:
        if self._discard_frequency_threshold is None:
            return 0
        return np.percentile(self._frequency_ratio, self._discard_frequency_threshold * 100)

    @functools.cached_property
    def _p_discard(self) -> np.ndarray:
        """Probability of discarding a item.

        If `rho` is zero, item is not discarded at all.


        """
        if self._discard_frequency_threshold is None:
            return np.zeros(self._item_size)

        p = 1 - np.sqrt(self._rho / self._frequency_ratio)
        return np.clip(p, 0, self._max_discard_probability)

    def _dropout(self, sequence: np.ndarray) -> np.ndarray:
        """Dropout items from sequence."""
        if self._discard_frequency_threshold is None:
            return sequence

        p = np.random.random(size=sequence.shape)
        p_discard = self._p_discard[sequence]
        return sequence[p >= p_discard]

    @staticmethod
    def _get_skip_gram_sample(targets: np.ndarray) -> np.ndarray:
        """Get skip-gram sample from sequence.

        In word2vec, positive samples are words that are adjacent to the target.
        But in item2vec, positive samples are all items in sequence except the target.
        Ref: SECTION 3. ITEM2VEC – SGNS FOR ITEM SIMILARITY
        """
        samples = []
        for i in range(len(targets)):
            samples.append(np.delete(targets.copy(), i))
        return np.stack(samples)

    def transform(self, features: dict[str, str | int | list[int]]) -> dict[str, torch.Tensor] | None:
        inputs = np.array(features[_INPUTS_COL])
        targets = self._dropout(inputs)

        targets_len = len(targets)

        if targets_len == 0:
            return None

        if self._max_length is not None and targets_len > self._max_length:
            targets = np.random.choice(targets, size=self._max_length, replace=False)

        positive_contexts = self._get_skip_gram_sample(targets)
        negative_item_indices = np.setdiff1d(self._total_item_array, inputs)
        return {
            _TARGETS_COL: torch.from_numpy(targets),
            _POSITIVE_CONTEXTS_COL: torch.from_numpy(positive_contexts),
            _NEGATIVE_CONTEXTS_COL: torch.from_numpy(negative_item_indices),
        }
