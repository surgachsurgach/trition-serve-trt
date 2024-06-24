from absl.testing import absltest
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.datasets import collate
from data.ml.model_runner.utils import test_utils
from data.ml.utils import metadata

_TARGETS_COL = common.TARGETS_COL
_POSITIVE_CONTEXTS_COL = common.POSITIVE_CONTEXTS_COL
_NEGATIVE_CONTEXTS_COL = common.NEGATIVE_CONTEXTS_COL


class CollateTest(absltest.TestCase):
    def test_item2vec_collate(self):
        collator = collate.Item2VecCollator(
            meta=metadata.Meta.from_json({"item_id": ["2", "1", "3", "0"], "item_frequency": [1, 2, 3, 2]}),
        )

        actual = collator.collate(
            [
                {
                    common.TARGETS_COL: torch.tensor([2, 0]),
                    common.POSITIVE_CONTEXTS_COL: torch.tensor([[0], [2]]),
                    common.NEGATIVE_CONTEXTS_COL: torch.tensor([[3, 4], [3, 3]]),
                },
                {
                    common.TARGETS_COL: torch.tensor([1, 2, 3]),
                    common.POSITIVE_CONTEXTS_COL: torch.tensor([[2, 3], [1, 3], [1, 2]]),
                    common.NEGATIVE_CONTEXTS_COL: torch.tensor([[0, 0], [0, 0], [0, 0]]),
                },
            ]
        )

        expected = {
            _TARGETS_COL: torch.tensor(
                [
                    [2, 0, 4],
                    [1, 2, 3],
                ]
            ),
            _POSITIVE_CONTEXTS_COL: torch.tensor(
                [
                    [[0, 4], [2, 4], [4, 4]],
                    [[2, 3], [1, 3], [1, 2]],
                ]
            ),
            _NEGATIVE_CONTEXTS_COL: torch.tensor(
                [
                    [[3, 4], [3, 3], [4, 4]],
                    [[0, 0], [0, 0], [0, 0]],
                ]
            ),
        }
        test_utils.assert_dict_equals(actual, expected)


if __name__ == "__main__":
    absltest.main()
