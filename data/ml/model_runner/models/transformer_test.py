from absl.testing import absltest
import torch

from data.ml.model_runner.models import transformer


class TransformerTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_transformer_encoder_layer(self):
        x = torch.rand(2, 2, 4)
        encoder_layer = transformer.TransformerEncoderLayer(4, 2, d_ff=2048, dropout=0.0)
        out = encoder_layer(x, mask=None)
        assert out.shape == (2, 2, 4)
        torch.testing.assert_close(
            out,
            torch.tensor(
                [
                    [[0.52741563, 0.81423539, -0.91809350, 0.09324713], [0.33240348, 0.94186622, -0.61993301, 0.52030545]],
                    [[0.61176729, 0.75025922, -0.67240858, 0.28009096], [0.25523773, 0.58053011, -0.68229413, 0.12760280]],
                ]
            ),
        )

    def test_transformer_encoder_layer_invalid_shape(self):
        with self.assertRaises(AssertionError):
            transformer.TransformerEncoderLayer(5, 2, d_ff=2048, dropout=0.0)

    def test_transformer_encoder(self):
        x = torch.rand(2, 2, 4)
        encoder = transformer.TransformerEncoder(2, 4, 2, d_ff=2048, dropout=0.0)
        out = encoder(x)
        assert out.shape == (2, 2, 4)
        torch.testing.assert_close(
            out,
            torch.tensor(
                [
                    [[1.33499038, 0.06594975, -1.48533034, 0.08438999], [0.82968587, 0.38197237, -1.70842254, 0.49676445]],
                    [[1.37035048, -0.17520836, -1.42736328, 0.23222132], [1.28884172, -0.05886258, -1.50345266, 0.27347347]],
                ]
            ),
        )

    def test_transformer_encoder_mask(self):
        x = torch.rand(2, 4, 4)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        encoder = transformer.TransformerEncoder(2, 4, 2, d_ff=2048, dropout=0.0)
        out = encoder(x, mask=mask)
        assert out.shape == (2, 4, 4)
        torch.testing.assert_close(
            out,
            torch.tensor(
                [
                    [
                        [0.40173727, 1.43504214, -0.70353866, -1.13324070],
                        [0.19497824, 1.50301993, -1.21034968, -0.48764896],
                        [0.39368647, 1.44088292, -0.70997947, -1.12458968],
                        [0.36434841, 1.45703435, -1.11734104, -0.70404136],
                    ],
                    [
                        [0.41014689, 1.37682962, -0.48155573, -1.30542064],
                        [0.24896574, 1.52731228, -0.77087414, -1.00540400],
                        [-0.01298999, 0.95372796, 0.68021154, -1.62094951],
                        [0.01009226, 1.13887000, -1.58497322, 0.43601084],
                    ],
                ]
            ),
        )


if __name__ == "__main__":
    absltest.main()
