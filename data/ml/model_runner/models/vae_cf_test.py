from absl.testing import absltest
import torch

from data.ml.model_runner.generators import item2user as item2user_generator
from data.ml.model_runner.models import vae_cf
from data.ml.utils import metadata
from data.pylib.constant import recsys as common


class VaeCfTest(absltest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _create_meta(self, item_size: int) -> metadata.Meta:
        data = {common.ITEM_ID_COL: [str(i) for i in range(item_size)]}
        return metadata.Meta.from_json(data)

    def test_vae(self):
        meta = self._create_meta(10)
        model = vae_cf.VAE(
            meta=meta,
            generator=item2user_generator.TopKGenerator(meta=meta, revert_idx_to_id=False),
            encoder_dims=[5, 2],
            decoder_dims=[2, 5],
            total_training_steps=10,
        )

        assert model(torch.Tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]))

    def test_cvae(self):
        meta = self._create_meta(10)
        meta.add_meta("gender", ["unknown", "F", "M"])
        meta.add_meta("generation", ["10", "20", "40", "unknown"])

        model = vae_cf.CVAE(
            meta=meta,
            generator=item2user_generator.TopKGenerator(meta=meta, revert_idx_to_id=False),
            encoder_dims=[5, 2],
            decoder_dims=[2, 5],
            total_training_steps=10,
            label_cols=["gender", "generation"],
        )

        assert model(torch.Tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]), torch.Tensor([[0, 0, 1, 0, 1, 0, 0]]))

    def test_vae_invalid_dims(self):
        with self.assertRaises(AssertionError):
            meta = self._create_meta(100)
            vae_cf.VAE(
                meta=meta,
                generator=item2user_generator.TopKGenerator(meta=meta, revert_idx_to_id=False),
                encoder_dims=[50, 20],
                decoder_dims=[30, 50],
                total_training_steps=10,
            )

    def test_vae_exclude_inputs_from_prediction(self):
        meta = self._create_meta(10)
        model = vae_cf.VAE(
            meta=meta,
            generator=item2user_generator.ExclusionGenerator(meta=meta, revert_idx_to_id=False, top_k=5),
            encoder_dims=[5, 2],
            decoder_dims=[2, 5],
            total_training_steps=10,
        )

        actual = model.predict_step(
            {
                common.USER_ID_COL: torch.LongTensor([1234]),
                common.INPUTS_COL: torch.Tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            },
            batch_idx=0,
        )

        self.assertSameElements([1234, 1234, 1234, 1234, 1234], actual[common.USER_ID_COL].tolist())
        self.assertFalse(set([2, 5]).issubset(set(actual["item_idx"].tolist())))

    def test_vae_not_exclude_inputs_from_prediction(self):
        meta = self._create_meta(10)
        model = vae_cf.VAE(
            meta=meta,
            generator=item2user_generator.TopKGenerator(meta=meta, revert_idx_to_id=False, top_k=10),
            encoder_dims=[5, 2],
            decoder_dims=[2, 5],
            total_training_steps=10,
        )

        actual = model.predict_step(
            {
                common.USER_ID_COL: torch.LongTensor([1234]),
                common.INPUTS_COL: torch.Tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
            },
            batch_idx=0,
        )

        self.assertSameElements([1234, 1234, 1234, 1234, 1234], actual[common.USER_ID_COL].tolist())
        self.assertTrue(set([2, 5]).issubset(set(actual["item_idx"].tolist())))


if __name__ == "__main__":
    absltest.main()
