from absl.testing import absltest
import numpy as np
import torch

from data.ml.model_runner.metrics import mean_ap


class MeanApTest(absltest.TestCase):
    def test_map(self):
        np.testing.assert_almost_equal(
            torch.tensor(1.0),
            mean_ap.mean_average_precision_at_k(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[0.2, 0.3, 0.5]])),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.8333333),
            mean_ap.mean_average_precision_at_k(torch.tensor([[1.0, 0.0, 1.0]]), torch.tensor([[0.2, 0.3, 0.5]])),
        )

        np.testing.assert_almost_equal(
            torch.tensor(1.0),
            mean_ap.mean_average_precision_at_k(torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([[0.2, 0.3, 0.5]])),
        )

        np.testing.assert_almost_equal(
            torch.tensor(1.0),
            mean_ap.mean_average_precision_at_k(torch.tensor([[1.0, 0.0, 1.0]]), torch.tensor([[0.2, 0.3, 0.5]]), k=2),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.75),
            mean_ap.mean_average_precision_at_k(
                torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
                torch.tensor([[0.2, 0.5, 0.3], [0.2, 0.3, 0.5]]),
            ),
        )


if __name__ == "__main__":
    absltest.main()
