from absl.testing import absltest
import torch

from data.ml.model_runner.modules.layers import feed_forward


class FeedForwardTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_printoptions(6)
        return super().setUp()

    def test_feed_forward(self):
        layer = feed_forward.FeedForward(2, 4, dropout1=0, dropout2=0)
        output = layer(torch.tensor([[2.0, 2.0], [1.0, 1.0]]))

        torch.testing.assert_close(output, torch.tensor([[-0.128069, 0.342875], [0.040470, 0.387964]]))


if __name__ == "__main__":
    absltest.main()
