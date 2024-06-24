import random

from absl.testing import absltest
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.models import cl4srec
from data.ml.utils import metadata


class CL4SRecTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)

        self.meta = metadata.Meta.from_json({"item_id": [str(i) for i in range(10154)]})

    def test_contrastive_loss(self):
        model = cl4srec.CL4SRec(
            d_model=4,
            num_heads=2,
            num_encoder_layers=2,
            max_seq_len=10,
            dropout=0.0,
            attn_dropout=0.0,
            meta=self.meta,
        )

        x1 = {
            common.ITEM_ID_COL: torch.tensor(
                [
                    [1, 17, 62, 141, 10155, 0, 0, 0, 0, 0],
                    [5351, 17, 62, 141, 10155, 0, 0, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
        }

        x2 = {
            common.ITEM_ID_COL: torch.tensor(
                [
                    [10155, 141, 62, 141, 10155, 0, 0, 0, 0, 0],
                    [2222, 17, 62, 141, 1, 0, 0, 0, 0, 0],
                ],
                dtype=torch.long,
            ),
        }

        output1 = model(x1, torch.tensor([[5], [5]]))
        output2 = model(x2, torch.tensor([[5], [5]]))

        output = model._constrastive_loss(output1, output2, 2)  # pylint: disable=protected-access

        torch.testing.assert_close(output, torch.tensor(1.097471))


if __name__ == "__main__":
    absltest.main()
