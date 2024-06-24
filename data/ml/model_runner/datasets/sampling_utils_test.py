from absl.testing import absltest
import numpy as np
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.datasets import sampling_utils
from data.ml.model_runner.utils import test_utils
from data.ml.utils import metadata

_INPUTS_COL = common.INPUTS_COL
_TARGETS_COL = common.TARGETS_COL
_POSITIVE_CONTEXTS_COL = common.POSITIVE_CONTEXTS_COL
_NEGATIVE_CONTEXTS_COL = common.NEGATIVE_CONTEXTS_COL


class SamplingUtilsTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        np.random.seed(1)

    def test_negative_sampler(self):
        sampler = sampling_utils.NegativeSampler(
            meta=metadata.Meta.from_json({"item_id": ["2", "1", "3", "0"], "item_frequency": [1, 2, 3, 2]}),
            num_negative_samples=2,
        )

        actual = sampler.transform({_INPUTS_COL: [2, 0], _TARGETS_COL: [2, 0], _POSITIVE_CONTEXTS_COL: [[0], [2]]})
        expected = {
            _TARGETS_COL: torch.tensor([2, 0]),
            _POSITIVE_CONTEXTS_COL: torch.tensor([[0], [2]]),
            _NEGATIVE_CONTEXTS_COL: torch.tensor([1, 3]),
        }
        test_utils.assert_dict_equals(actual, expected)

        actual = sampler.postprocess(actual)
        expected = {
            _TARGETS_COL: torch.tensor([2, 0]),
            _POSITIVE_CONTEXTS_COL: torch.tensor([[0], [2]]),
            _NEGATIVE_CONTEXTS_COL: torch.tensor([[1, 3], [1, 1]]),
        }

        test_utils.assert_dict_equals(actual, expected)

    def test_skip_gram_negative_sampler(self):
        sampler = sampling_utils.SkipGramNegativeSampler(
            meta=metadata.Meta.from_json({"item_id": ["2", "1", "3", "0"], "item_frequency": [1, 2, 3, 2]}),
            num_negative_samples=2,
            discard_frequency_threshold=None,
        )

        actual = sampler.transform({_INPUTS_COL: [2, 0]})
        expected = {
            _TARGETS_COL: torch.tensor([2, 0]),
            _POSITIVE_CONTEXTS_COL: torch.tensor([[0], [2]]),
            _NEGATIVE_CONTEXTS_COL: torch.tensor([1, 3]),
        }
        test_utils.assert_dict_equals(actual, expected)

        actual = sampler.postprocess(actual)
        expected = {
            _TARGETS_COL: torch.tensor([2, 0]),
            _POSITIVE_CONTEXTS_COL: torch.tensor([[0], [2]]),
            _NEGATIVE_CONTEXTS_COL: torch.tensor([[1, 3], [1, 1]]),
        }

        test_utils.assert_dict_equals(actual, expected)


if __name__ == "__main__":
    absltest.main()
