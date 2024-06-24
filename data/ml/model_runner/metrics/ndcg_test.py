from absl.testing import absltest
import numpy as np
import torch

from data.ml.model_runner.metrics import ndcg


class NdcgTest(absltest.TestCase):
    def test_ndcg(self):
        np.testing.assert_almost_equal(
            torch.tensor(0.695694),
            ndcg.normalized_dcg_at_k(torch.tensor([[10, 0, 0, 1, 5]]), torch.tensor([[0.1, 0.2, 0.3, 4, 70]])),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.4936802),
            ndcg.normalized_dcg_at_k(torch.tensor([[10, 0, 0, 1, 5]]), torch.tensor([[0.05, 1.1, 1.0, 0.5, 0.0]])),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.0366176),
            ndcg.normalized_dcg_at_k(
                torch.tensor([[10, 0, 0, 1, 5]]),
                torch.tensor([[0.05, 1.1, 1.0, 0.5, 0.0]]),
                k=3,
            ),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.0),
            ndcg.normalized_dcg_at_k(
                torch.tensor([[10, 0, 0, 1, 5]]),
                torch.tensor([[0.05, 1.1, 1.0, 0.5, 0.0]]),
                k=2,
            ),
        )

        np.testing.assert_almost_equal(
            torch.tensor(1.0),
            ndcg.normalized_dcg_at_k(torch.tensor([[10, 0, 0, 1, 5]]), torch.tensor([[10, 0, 0, 1, 5]])),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.6182886),
            ndcg.normalized_dcg_at_k(torch.tensor([[1, 0, 0, 1, 1]]), torch.tensor([[0.2, 0.4, 0.6, 0.1, 0.3]])),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.2346394),
            ndcg.normalized_dcg_at_k(
                torch.tensor([[1, 0, 0, 1, 1]]),
                torch.tensor([[0.2, 0.4, 0.6, 0.1, 0.3]]),
                k=3,
            ),
        )


if __name__ == "__main__":
    absltest.main()
