from absl.testing import absltest
import torch

from data.ml.model_runner.utils import torch_utils


class TorchUtilsTest(absltest.TestCase):
    def test_multi_hot_encoding(self):
        indices = [3, 5, 7]
        size = 10

        expected = torch.Tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        actual = torch_utils.multi_hot_encoding(indices, size)

        assert torch.equal(expected, actual)
        assert expected.dtype == actual.dtype

    def test_multi_hot_weighted_encoding(self):
        indices = [3, 5, 7]
        weights = [0.1, 0.2, 0.3]
        size = 10

        expected = torch.Tensor([0.0, 0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.3, 0.0, 0.0])
        actual = torch_utils.multi_hot_weighted_encoding(indices, weights, size)

        assert torch.equal(expected, actual)
        assert expected.dtype == actual.dtype


if __name__ == "__main__":
    absltest.main()
