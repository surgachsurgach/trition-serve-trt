from absl.testing import absltest
import pandas as pd
import torch

from data.ml.model_runner.generators import item2user
from data.ml.utils import metadata
from data.pylib.constant import recsys as common


class Item2UserGeneratorTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _create_meta(self, item_size: int) -> metadata.Meta:
        data = {common.ITEM_ID_COL: [str(i) for i in range(item_size)]}
        return metadata.Meta.from_json(data)

    def test_topk_generator(self):
        meta = self._create_meta(5)
        generator = item2user.TopKGenerator(meta=meta, revert_idx_to_id=False)
        logits = torch.tensor([[0.1, 1.0, 0.2, 0.3, 0.4], [0.4, 0.3, 0.1, 1.0, 0.2], [1.0, 0.2, 0.3, 0.4, 0.1]])
        batch_data = {common.USER_ID_COL: torch.tensor([0, 1, 2])}

        actual = generator(logits, batch_data)
        expected = pd.DataFrame(
            [
                {"user_id": 0, "item_idx": 1, "score": 1.0},
                {"user_id": 0, "item_idx": 4, "score": 0.4},
                {"user_id": 0, "item_idx": 3, "score": 0.3},
                {"user_id": 0, "item_idx": 2, "score": 0.2},
                {"user_id": 0, "item_idx": 0, "score": 0.1},
                {"user_id": 1, "item_idx": 3, "score": 1.0},
                {"user_id": 1, "item_idx": 0, "score": 0.4},
                {"user_id": 1, "item_idx": 1, "score": 0.3},
                {"user_id": 1, "item_idx": 4, "score": 0.2},
                {"user_id": 1, "item_idx": 2, "score": 0.1},
                {"user_id": 2, "item_idx": 0, "score": 1.0},
                {"user_id": 2, "item_idx": 3, "score": 0.4},
                {"user_id": 2, "item_idx": 2, "score": 0.3},
                {"user_id": 2, "item_idx": 1, "score": 0.2},
                {"user_id": 2, "item_idx": 4, "score": 0.1},
            ],
        ).astype(
            {
                "user_id": "int64",
                "item_idx": "int64",
                "score": "float32",
            }
        )

        actual = actual.sort_values(by=["user_id", "score"], ignore_index=True)
        expected = expected.sort_values(by=["user_id", "score"], ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected)

        generator = item2user.TopKGenerator(meta=meta, top_k=2, revert_idx_to_id=False)

        actual = generator(logits, batch_data)
        expected = pd.DataFrame(
            [
                {"user_id": 0, "item_idx": 1, "score": 1.0},
                {"user_id": 0, "item_idx": 4, "score": 0.4},
                {"user_id": 1, "item_idx": 3, "score": 1.0},
                {"user_id": 1, "item_idx": 0, "score": 0.4},
                {"user_id": 2, "item_idx": 0, "score": 1.0},
                {"user_id": 2, "item_idx": 3, "score": 0.4},
            ],
        ).astype(
            {
                "user_id": "int64",
                "item_idx": "int64",
                "score": "float32",
            }
        )
        pd.testing.assert_frame_equal(actual, expected)
        generator = item2user.TopKGenerator(meta=meta, top_k=2, revert_idx_to_id=True)

        actual = generator(logits, batch_data)
        expected = pd.DataFrame(
            [
                {"user_id": 0, "item_id": "1", "score": 1.0},
                {"user_id": 0, "item_id": "4", "score": 0.4},
                {"user_id": 1, "item_id": "3", "score": 1.0},
                {"user_id": 1, "item_id": "0", "score": 0.4},
                {"user_id": 2, "item_id": "0", "score": 1.0},
                {"user_id": 2, "item_id": "3", "score": 0.4},
            ],
        ).astype(
            {
                "user_id": "int64",
                "item_id": "object",
                "score": "float32",
            }
        )

        actual = actual.sort_values(by=["user_id", "score"], ignore_index=True)
        expected = expected.sort_values(by=["user_id", "score"], ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_exclusion_generator(self):
        meta = self._create_meta(5)
        generator = item2user.ExclusionGenerator(meta=meta, top_k=2, revert_idx_to_id=False)
        logits = torch.tensor([[0.1, 1.0, 0.3, 0.0, 0.0], [0.4, 0.0, 0.1, 1.0, 0.0], [1.0, 0.0, 0.4, 0.0, 0.1]])
        batch_data = {
            common.USER_ID_COL: torch.tensor([0, 1, 2]),
            common.INPUTS_COL: torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]]),
        }

        actual = generator(logits, batch_data)
        expected = pd.DataFrame(
            [
                {"user_id": 0, "item_idx": 2, "score": 0.3},
                {"user_id": 0, "item_idx": 0, "score": 0.1},
                {"user_id": 1, "item_idx": 0, "score": 0.4},
                {"user_id": 1, "item_idx": 2, "score": 0.1},
                {"user_id": 2, "item_idx": 2, "score": 0.4},
                {"user_id": 2, "item_idx": 4, "score": 0.1},
            ]
        ).astype(
            {
                "user_id": "int64",
                "item_idx": "int64",
                "score": "float32",
            }
        )

        actual = actual.sort_values(by=["user_id", "score"], ignore_index=True)
        expected = expected.sort_values(by=["user_id", "score"], ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    absltest.main()
