from absl.testing import absltest
import numpy as np
import torch

from data.ml.model_runner.metrics import recall


class RecallTest(absltest.TestCase):
    def test_recall(self):
        np.testing.assert_almost_equal(
            torch.tensor(0.33333334),
            recall.recall_at_k(
                torch.tensor([[0, 0, 1, 1, 1]]),
                torch.tensor([[0.2, 0.3, 0.9, 0, 0.1]]),
                k=1,
            ),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.33333334),
            recall.recall_at_k(
                torch.tensor([[0, 0, 1, 1, 1]]),
                torch.tensor([[0.2, 0.3, 0.9, 0, 0.1]]),
                k=2,
            ),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.16666667),
            recall.recall_at_k(
                torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1]]),
                torch.tensor([[0.2, 0.3, 0.9, 0, 0.1], [1, 1, 0, 0, 0.1]]),
                k=2,
            ),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.6666667),
            recall.recall_at_k(torch.tensor([[0, 0, 1, 1, 1]]), torch.tensor([[1, 1, 1, 0, 0.1]]), k=4),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.0),
            recall.recall_at_k(torch.tensor([[0, 0, 1, 1, 1]]), torch.tensor([[1, 1, 0, 0, 0.1]]), k=2),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.3333333),
            recall.recall_at_k(torch.tensor([[0, 0, 1, 1, 1]]), torch.tensor([[1, 1, 0, 0, 0.1]]), k=3),
        )

        np.testing.assert_almost_equal(
            torch.tensor(0.3333333),
            recall.recall_at_k(torch.tensor([[0, 0, 1, 2, 3]]), torch.tensor([[1, 1, 0, 0, 0.1]]), k=3),
        )

        np.testing.assert_almost_equal(
            torch.tensor(1.0),
            recall.recall_at_k(torch.tensor([[0, 0, 1, 1, 1]]), torch.tensor([[0, 0, 1, 1, 1]])),
        )


if __name__ == "__main__":
    absltest.main()
