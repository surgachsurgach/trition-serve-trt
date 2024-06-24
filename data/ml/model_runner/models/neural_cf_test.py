from absl.testing import absltest
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.models import neural_cf
from data.ml.utils import metadata


class NeuralCfTest(absltest.TestCase):
    def _create_meta(self, item_size, user_size):
        return metadata.Meta.from_json(
            {
                common.ITEM_ID_COL: [str(i) for i in range(item_size)],
                common.USER_ID_COL: [str(i) for i in range(user_size)],
            }
        )

    def test_ncf(self):
        torch.manual_seed(0)
        model = neural_cf.NeuralCF(
            meta=self._create_meta(10, 15),
            mlp_latent_dim=5,
            mf_latent_dim=5,
            mlp_dims=[10, 5],
            dropout=0.0,
        )

        actual = model(torch.tensor([5, 11, 13]), torch.tensor([1, 0, 9])).detach()
        assert actual.shape == (3, 1)
        torch.testing.assert_close(actual, torch.tensor([[0.132507], [0.090964], [0.238254]]))

    def test_ncf_training_step(self):
        torch.manual_seed(0)
        model = neural_cf.NeuralCF(
            meta=self._create_meta(10, 15),
            mlp_latent_dim=5,
            mf_latent_dim=5,
            mlp_dims=[10, 5],
            dropout=0.0,
        )

        actual = model.training_step(
            {
                common.USER_INPUTS_COL: torch.tensor([5, 11, 13]),
                common.ITEM_INPUTS_COL: torch.tensor([1, 0, 9]),
                common.TARGETS_COL: torch.tensor([0.0, 1.0, 1.0]),
            },
            0,
        )

        torch.testing.assert_close(actual, torch.tensor(0.663797))


if __name__ == "__main__":
    absltest.main()
