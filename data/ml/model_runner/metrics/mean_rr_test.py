from absl.testing import absltest
import numpy as np
import torch

from data.ml.model_runner.metrics import mean_rr


class MeanRrTest(absltest.TestCase):
    def test_mrr(self):
        np.testing.assert_almost_equal(
            torch.tensor(1.0),
            mean_rr.mean_reciprocal_rank_at_k(torch.tensor([[0, 0, 1]]), torch.tensor([[0.2, 0.3, 0.5]])),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.5),
            mean_rr.mean_reciprocal_rank_at_k(torch.tensor([[0, 0, 1]]), torch.tensor([[0.5, 0.3, 0.4]])),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.3333333),
            mean_rr.mean_reciprocal_rank_at_k(torch.tensor([[0, 0, 1]]), torch.tensor([[0.5, 0.4, 0.3]])),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.5),
            mean_rr.mean_reciprocal_rank_at_k(torch.tensor([[0, 0, 1]]), torch.tensor([[0.3, 0.6, 0.4]]), k=2),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.75),
            mean_rr.mean_reciprocal_rank_at_k(
                torch.tensor([[1, 1, 0], [0, 1, 1]]),
                torch.tensor([[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]),
            ),
        )


if __name__ == "__main__":
    absltest.main()
