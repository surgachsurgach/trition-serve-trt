from absl.testing import absltest
import torch

from data.pylib.constant import recsys as common
from data.ml.model_runner.models import dif_sr
from data.ml.utils import metadata


class DIFTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_dif_transformer_layer(self):
        x = torch.tensor(
            [
                [[-1.2991e-01, 3.0695e-01, 1.3031e00, -1.4801e00], [7.6027e-01, 1.2126e00, -9.3454e-01, -1.0384e00]],
                [[5.2208e-01, 1.0788e00, 1.7309e-04, -1.6011e00], [4.1088e-02, 2.6094e-01, -1.5447e00, 1.2427e00]],
            ]
        )
        attr_embed = [
            torch.tensor(
                [
                    [[[-0.0576, -2.1619, 0.4419, 0.1133]], [[1.7243, -0.5977, -0.7355, 0.3348]]],
                    [[[0.7070, 0.3586, -0.3149, 0.6512]], [[1.2102, -0.3115, 1.2194, -1.8096]]],
                ]
            )
        ]

        positional_embedding = torch.tensor(
            [
                [[-0.0231, 0.0268, -0.0028, -0.0030], [-0.0343, 0.0232, -0.0118, 0.0370]],
                [[-0.0231, 0.0268, -0.0028, -0.0030], [-0.0343, 0.0232, -0.0118, 0.0370]],
            ]
        )
        attention_mask = torch.tensor([[[[1, 0], [1, 1]]], [[[1, 0], [1, 1]]]])

        encoder_layer = dif_sr.DIFTransformerEncoderLayer(4, 4, 1, [4], 4, max_len=2)
        actual = encoder_layer(x, attr_embed, positional_embedding, attention_mask)
        expected = torch.tensor(
            [
                [[0.425890, -0.220505, 1.266414, -1.471798], [1.200781, 0.596073, -0.355989, -1.440865]],
                [[0.887398, 0.448739, 0.361137, -1.697274], [1.123512, -0.173642, -1.537113, 0.587242]],
            ]
        )

        torch.testing.assert_close(actual, expected)

    def test_dif_transformer_encoder(self):
        x = torch.tensor(
            [
                [[-1.2991e-01, 3.0695e-01, 1.3031e00, -1.4801e00], [7.6027e-01, 1.2126e00, -9.3454e-01, -1.0384e00]],
                [[5.2208e-01, 1.0788e00, 1.7309e-04, -1.6011e00], [4.1088e-02, 2.6094e-01, -1.5447e00, 1.2427e00]],
            ]
        )
        attr_embed = [
            torch.tensor(
                [
                    [[[-0.0576, -2.1619, 0.4419, 0.1133]], [[1.7243, -0.5977, -0.7355, 0.3348]]],
                    [[[0.7070, 0.3586, -0.3149, 0.6512]], [[1.2102, -0.3115, 1.2194, -1.8096]]],
                ]
            )
        ]

        positional_embedding = torch.tensor(
            [
                [[-0.0231, 0.0268, -0.0028, -0.0030], [-0.0343, 0.0232, -0.0118, 0.0370]],
                [[-0.0231, 0.0268, -0.0028, -0.0030], [-0.0343, 0.0232, -0.0118, 0.0370]],
            ]
        )
        attention_mask = torch.tensor([[[[1, 0], [1, 1]]], [[[1, 0], [1, 1]]]])

        encoder = dif_sr.DIFTransformerEncoder(4, 2, 2, 1, [4], 4, max_len=2)
        actual = encoder(x, attr_embed, positional_embedding, attention_mask)

        expected = [
            torch.tensor(
                [
                    [[0.425890, -0.220505, 1.266414, -1.471798], [1.197798, 0.604393, -0.364419, -1.437772]],
                    [[0.887398, 0.448739, 0.361137, -1.697274], [1.089508, -0.191337, -1.538353, 0.640182]],
                ]
            ),
            torch.tensor(
                [
                    [[0.859906, -0.581413, 1.061533, -1.340026], [1.322819, 0.108065, 0.063903, -1.494787]],
                    [[1.083571, -0.013567, 0.526420, -1.596424], [1.670911, -0.519586, -0.946684, -0.204641]],
                ]
            ),
        ]

        torch.testing.assert_close(actual, expected)

    def test_dif_model(self):
        meta = metadata.Meta.from_json(
            {
                common.ITEM_ID_COL: [str(i) for i in range(12100)],
                "item_category": list(range(2076)),
            }
        )
        model = dif_sr.DIF(
            d_model=4,
            num_heads=2,
            num_encoder_layers=2,
            d_ff=4,
            side_features=["item_category"],
            d_side_features=[4],
            lambdas=[10],
            max_seq_len=2,
            meta=meta,
        )

        x = {
            common.ITEM_ID_COL: torch.tensor([[7075, 6570, 9165], [11858, 11763, 11898]]),
            "item_category": torch.tensor([[14, 525, 525], [2065, 39, 0]]),
            common.SEQ_LEN_COL: torch.tensor([[3], [3]]),
        }

        data = {k: v[..., :-1] for k, v in x.items()}
        targets = {k: v[..., -1] for k, v in x.items()}

        actual = model(data, torch.tensor([2, 2]))
        expected = torch.tensor(
            [
                [-0.666097, -0.980979, 1.610165, 0.036911],
                [-1.364443, 1.246138, 0.596943, -0.478639],
            ]
        )

        torch.testing.assert_allclose(actual, expected)

        loss = model.loss(actual, targets)
        torch.testing.assert_close(loss, torch.tensor(16.333965))

        batch_data = {
            common.INPUTS_COL: data,
            common.TARGETS_COL: targets,
            common.SEQ_LEN_COL: torch.tensor([2, 2]),
        }

        actual = model.training_step(batch_data, 0)
        torch.testing.assert_close(actual, torch.tensor(16.333965))


if __name__ == "__main__":
    absltest.main()
