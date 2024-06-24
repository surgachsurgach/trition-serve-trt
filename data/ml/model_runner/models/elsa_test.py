from absl.testing import absltest
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.models import elsa
from data.ml.utils import metadata


class ElsaTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _create_meta(self, item_size):
        return metadata.Meta.from_json({common.ITEM_ID_COL: [str(i) for i in range(item_size)]})

    def test_elsa(self):
        model = elsa.ELSA(meta=self._create_meta(5), num_dims=5)

        actual = model(torch.Tensor([[0.0, 0.0, 1.0, 0.0, 1.0]])).detach()
        torch.testing.assert_close(actual, torch.tensor([[0.883786, -0.960347, -0.326738, -0.192507, -0.326738]]))

    def test_elsa_training_step(self):
        model = elsa.ELSA(meta=self._create_meta(5), num_dims=5)
        actual = model.training_step(
            {
                common.INPUTS_COL: torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0]),
                common.TARGETS_COL: torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
            },
            0,
        )

        torch.testing.assert_close(actual, torch.tensor(0.343561))


if __name__ == "__main__":
    absltest.main()
