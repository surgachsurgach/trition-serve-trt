from absl.testing import absltest
import torch

from data.ml.model_runner.modules.layers import mlp


class MLPTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        return super().setUp()

    def test_mlp(self):
        layer = mlp.MLP([2, 4])
        output = layer(torch.tensor([1.0, 2.0]))
        assert output.size(dim=0) == 4


if __name__ == "__main__":
    absltest.main()
