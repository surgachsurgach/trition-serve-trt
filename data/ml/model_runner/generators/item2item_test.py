from absl.testing import absltest
import pandas as pd
import torch
from torch.nn import functional as F

from data.ml.model_runner.generators import item2item
from data.ml.utils import metadata
from data.pylib.constant import recsys as common


def _mock_embedding_fn(x, num_classes=5):
    return F.one_hot(x, num_classes=num_classes).double()  # pylint: disable=not-callable


class EmbeddingOutputGeneratorTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_printoptions(precision=10)

    def _create_meta(self, item_size: int) -> metadata.Meta:
        data = {common.ITEM_ID_COL: [str(i) for i in range(item_size)]}
        return metadata.Meta.from_json(data)

    def test_embedding_vector_generator(self):
        meta = self._create_meta(5)
        generator = item2item.EmbeddingVectorGenerator(meta=meta, revert_idx_to_id=False)
        batch_data = {common.ITEM_IDX_COL: torch.tensor([1, 3, 0])}
        actual = generator(batch_data, lambda x: _mock_embedding_fn(x, num_classes=5))  # pylint: disable=not-callable
        expected = pd.DataFrame(
            [
                {"item_idx": 1, "embedding_vector": [0, 1, 0, 0, 0]},
                {"item_idx": 3, "embedding_vector": [0, 0, 0, 1, 0]},
                {"item_idx": 0, "embedding_vector": [1, 0, 0, 0, 0]},
            ]
        )
        actual = actual.sort_values(by=["item_idx"], ignore_index=True)
        expected = expected.sort_values(by=["item_idx"], ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected)

        generator = item2item.EmbeddingVectorGenerator(meta=meta, revert_idx_to_id=True)
        batch_data = {common.ITEM_IDX_COL: torch.tensor([1, 3, 0])}
        actual = generator(batch_data, lambda x: _mock_embedding_fn(x, num_classes=5))  # pylint: disable=not-callable
        expected = pd.DataFrame(
            [
                {"item_id": "1", "embedding_vector": [0, 1, 0, 0, 0]},
                {"item_id": "3", "embedding_vector": [0, 0, 0, 1, 0]},
                {"item_id": "0", "embedding_vector": [1, 0, 0, 0, 0]},
            ]
        )
        actual = actual.sort_values(by=["item_id"], ignore_index=True)
        expected = expected.sort_values(by=["item_id"], ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_similarity_generator(self):
        meta = self._create_meta(5)
        generator = item2item.SimilarityGenerator(meta=meta, revert_idx_to_id=False)
        batch_data = {common.ITEM_IDX_COL: torch.tensor([1, 3, 0])}
        actual = generator(batch_data, lambda x: _mock_embedding_fn(x, num_classes=5))  # pylint: disable=not-callable
        expected = pd.DataFrame(
            [
                {"item_idx": 1, "other_item_idx": 0, "similarity": 0.0},
                {"item_idx": 3, "other_item_idx": 0, "similarity": 0.0},
                {"item_idx": 0, "other_item_idx": 0, "similarity": 1.0},
                {"item_idx": 1, "other_item_idx": 1, "similarity": 1.0},
                {"item_idx": 3, "other_item_idx": 1, "similarity": 0.0},
                {"item_idx": 0, "other_item_idx": 1, "similarity": 0.0},
                {"item_idx": 1, "other_item_idx": 2, "similarity": 0.0},
                {"item_idx": 3, "other_item_idx": 2, "similarity": 0.0},
                {"item_idx": 0, "other_item_idx": 2, "similarity": 0.0},
                {"item_idx": 1, "other_item_idx": 3, "similarity": 0.0},
                {"item_idx": 3, "other_item_idx": 3, "similarity": 1.0},
                {"item_idx": 0, "other_item_idx": 3, "similarity": 0.0},
                {"item_idx": 1, "other_item_idx": 4, "similarity": 0.0},
                {"item_idx": 3, "other_item_idx": 4, "similarity": 0.0},
                {"item_idx": 0, "other_item_idx": 4, "similarity": 0.0},
            ],
        ).astype(
            {
                "item_idx": "int64",
                "other_item_idx": "int64",
                "similarity": "float32",
            }
        )
        actual = actual.sort_values(by=["item_idx", "similarity"], ignore_index=True)
        expected = expected.sort_values(by=["item_idx", "similarity"], ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected)

        generator = item2item.SimilarityGenerator(meta=meta, top_k=1, revert_idx_to_id=False)
        batch_data = {common.ITEM_IDX_COL: torch.tensor([1, 3, 0])}
        actual = generator(batch_data, lambda x: _mock_embedding_fn(x, num_classes=5))  # pylint: disable=not-callable
        expected = pd.DataFrame(
            [
                {"item_idx": 1, "other_item_idx": 1, "similarity": 1.0},
                {"item_idx": 3, "other_item_idx": 3, "similarity": 1.0},
                {"item_idx": 0, "other_item_idx": 0, "similarity": 1.0},
            ],
        ).astype(
            {
                "item_idx": "int64",
                "other_item_idx": "int64",
                "similarity": "float32",
            }
        )
        actual = actual.sort_values(by=["item_idx", "similarity"], ignore_index=True)
        expected = expected.sort_values(by=["item_idx", "similarity"], ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected)

        generator = item2item.SimilarityGenerator(meta=meta, top_k=1, revert_idx_to_id=True)
        batch_data = {common.ITEM_IDX_COL: torch.tensor([1, 3, 0])}
        actual = generator(batch_data, lambda x: _mock_embedding_fn(x, num_classes=5))  # pylint: disable=not-callable
        expected = pd.DataFrame(
            [
                {"item_id": "1", "other_item_id": "1", "similarity": 1.0},
                {"item_id": "3", "other_item_id": "3", "similarity": 1.0},
                {"item_id": "0", "other_item_id": "0", "similarity": 1.0},
            ],
        ).astype(
            {
                "item_id": "object",
                "other_item_id": "object",
                "similarity": "float32",
            }
        )
        actual = actual.sort_values(by=["item_id", "similarity"], ignore_index=True)
        expected = expected.sort_values(by=["item_id", "similarity"], ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    absltest.main()
