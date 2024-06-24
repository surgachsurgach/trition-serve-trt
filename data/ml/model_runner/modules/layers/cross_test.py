from absl.testing import absltest
import torch

from data.ml.model_runner.modules.layers import cross


class CrossTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        return super().setUp()

    def test_cross(self):
        layer = cross.Cross(2, 3)
        output = layer(torch.tensor([1.0, 2.0]))
        assert output.size(dim=0) == 2


if __name__ == "__main__":
    absltest.main()
