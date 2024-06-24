from absl.testing import absltest
import torch

from data.ml.model_runner.metrics import utils


class UtilsTest(absltest.TestCase):
    def test_assert_metric_inputs(self):
        with self.assertRaises(AssertionError):
            utils.assert_metric_inputs(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
            utils.assert_metric_inputs(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3, 4]))
            utils.assert_metric_inputs(torch.tensor([[1, 2, 3]]), torch.tensor([1, 2, 3, 4]))
            utils.assert_metric_inputs(torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2, 3, 4]]))

        utils.assert_metric_inputs(torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2, 3]]))


if __name__ == "__main__":
    absltest.main()
