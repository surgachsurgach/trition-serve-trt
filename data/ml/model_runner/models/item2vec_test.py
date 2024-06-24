from absl.testing import absltest
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.generators import item2item as i2i_generator
from data.ml.model_runner.models import item2vec
from data.ml.utils import metadata

_ITEM_ID_COL = common.ITEM_ID_COL

_TARGETS_COL = common.TARGETS_COL
_POSITIVE_CONTEXTS_COL = common.POSITIVE_CONTEXTS_COL
_NEGATIVE_CONTEXTS_COL = common.NEGATIVE_CONTEXTS_COL


class Item2VecTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_printoptions(precision=10)

    def _create_meta(self, item_size):
        return metadata.Meta.from_json({_ITEM_ID_COL: [str(i) for i in range(item_size)]})

    def test_item2vec(self):
        meta = self._create_meta(5)
        model = item2vec.Item2Vec(
            meta=meta,
            total_training_steps=10,
            generator=i2i_generator.EmbeddingVectorGenerator(meta=meta),
            embedding_dim=2,
        )
        # zero grad
        model.zero_grad()

        targets = torch.tensor([[1, 3, 0]])
        contexts = torch.tensor([[[3, 0], [1, 0], [1, 3]]])

        actual = model(targets, contexts)
        expected = torch.tensor(
            [
                [
                    [-0.4024474025, -3.5242998600],
                    [-0.4024474025, 0.3756546974],
                    [-3.5242998600, 0.3756546974],
                ]
            ]
        )
        torch.testing.assert_close(actual, expected)

        batch_data = {
            _TARGETS_COL: torch.tensor([[1, 3, 0]]),
            _POSITIVE_CONTEXTS_COL: torch.tensor([[[3, 0], [1, 0], [1, 3]]]),
            _NEGATIVE_CONTEXTS_COL: torch.tensor([[[2, 4], [4, 2], [2, 4]]]),
        }
        model.training_step(batch_data, 1)


if __name__ == "__main__":
    absltest.main()
