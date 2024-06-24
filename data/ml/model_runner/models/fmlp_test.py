from absl.testing import absltest
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.models import fmlp
from data.ml.utils import metadata


class FMLPTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_printoptions(6)
        self.meta = metadata.Meta.from_json(
            {
                common.ITEM_ID_COL: [str(i) for i in range(10)],
                "item_category": list(range(10)),
            }
        )

    def test_fmlp_model(self):
        model = fmlp.FMLP(
            d_model=4,
            num_encoder_layers=2,
            max_seq_len=2,
            dropout=0.0,
            mlp_dropout=0.0,
            meta=self.meta,
        )

        x = {
            common.ITEM_ID_COL: torch.tensor([[1, 3, 4], [5, 8, 9]]),
            "item_category": torch.tensor([[0, 2, 3], [2, 5, 1]]),
            common.SEQ_LEN_COL: torch.tensor([[3], [3]]),
        }

        data = {k: v[..., :-1] for k, v in x.items()}
        targets = {k: v[..., -1] for k, v in x.items()}

        actual = model(data, torch.tensor([2, 2]))
        expected = torch.tensor(
            [
                [0.209965, 0.417862, -0.480199, -0.348656],
                [1.215536, 0.936609, 0.301732, -0.417463],
            ]
        )
        torch.testing.assert_close(actual, expected)

        loss = model.loss(actual, targets)
        torch.testing.assert_close(loss, torch.tensor(2.007282))

    def test_fmlp_model_train_step(self):
        model = fmlp.FMLP(
            d_model=4,
            num_encoder_layers=2,
            max_seq_len=2,
            dropout=0.0,
            mlp_dropout=0.0,
            meta=self.meta,
        )

        x = {
            common.ITEM_ID_COL: torch.tensor([[1, 3, 4], [5, 8, 9]]),
            "item_category": torch.tensor([[0, 2, 3], [2, 5, 1]]),
            common.SEQ_LEN_COL: torch.tensor([[3], [3]]),
        }

        data = {k: v[..., :-1] for k, v in x.items()}
        targets = {k: v[..., -1] for k, v in x.items()}

        batch_data = {
            common.INPUTS_COL: data,
            common.TARGETS_COL: targets,
            common.SEQ_LEN_COL: torch.tensor([2, 2]),
        }

        actual = model.training_step(batch_data, 0)
        torch.testing.assert_close(actual, torch.tensor(2.007282))


if __name__ == "__main__":
    absltest.main()
