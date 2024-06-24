from absl.testing import absltest
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.generators import item2user as item2user_generator
from data.ml.model_runner.models import bert4rec
from data.ml.utils import metadata

_INPUTS_COL = common.INPUTS_COL
_ITEM_ID_COL = common.ITEM_ID_COL
_USER_ID_COL = common.USER_ID_COL
_SEQ_LEN_COL = common.SEQ_LEN_COL
_TARGETS_COL = common.TARGETS_COL
_TARGET_IDX_COL = common.TARGET_IDX_COL
_NEXT_TARGET_COL = common.NEXT_TARGET_COL
_ALL_TARGETS_COL = common.ALL_TARGETS_COL

_EXCLUSION_COL = common.EXCLUSION_COL


class Bert4RecTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_printoptions(precision=10)

    def test_bert4rec_model(self):
        meta = metadata.Meta.from_json(
            {
                _ITEM_ID_COL: [str(i) for i in range(10)],
            }
        )

        model = bert4rec.Bert4Rec(
            meta=meta,
            generator=item2user_generator.ExclusionGenerator(meta=meta, exclusion_col=_EXCLUSION_COL, threshold=0.0),
            d_model=4,
            num_heads=2,
            num_encoder_layers=2,
            d_ff=4,
            max_seq_len=8,
        )

        batch_data = {
            _INPUTS_COL: torch.tensor([[0, 11, 2, 11, 4, 10, 10, 10], [1, 2, 11, 4, 5, 10, 10, 10]]),
            _TARGETS_COL: torch.tensor([[-100, 2, -100, 4, -100, -100, -100, -100], [-100, -100, 4, -100, -100, -100, -100, -100]]),
            _TARGET_IDX_COL: torch.tensor([[5], [5]]),
        }

        model(batch_data[_INPUTS_COL], targets=batch_data[_TARGETS_COL])

        batch_data = {
            _INPUTS_COL: torch.tensor([[0, 11, 10, 10, 10]]),
            _TARGETS_COL: torch.tensor([[-100, 1, -100, -100, -100]]),
            _TARGET_IDX_COL: torch.tensor([[1]]),
            _NEXT_TARGET_COL: torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
            _ALL_TARGETS_COL: torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        }
        model.validation_step(batch_data, 0)

        batch_data = {
            _USER_ID_COL: torch.tensor([1]),
            _INPUTS_COL: torch.tensor([[0, 11, 10, 10, 10]]),
            _TARGET_IDX_COL: torch.tensor([[1]]),
            _EXCLUSION_COL: torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        }
        model.predict_step(batch_data, 0)


if __name__ == "__main__":
    absltest.main()
