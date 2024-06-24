import json
import os
import random

from absl.testing import absltest
import numpy as np
import pytest
import torch

from data.ml.model_runner.inference.handler import vae_cf as handler
from data.ml.model_runner.models import vae_cf

_TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data")
_CONTENT_TYPE = "application/json"


class HandlerTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["META_PATH"] = _TEST_DATA_DIR
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    @pytest.mark.skip(reason="Test results are not reproducible in github actions")
    def test_transform_fn(self):
        model = handler.model_fn(_TEST_DATA_DIR)
        self.assertIsInstance(model, vae_cf.VAE)
        self.assertEqual(model.hparams["encoder_dims"], [512, 256])
        self.assertEqual(model.hparams["decoder_dims"], [256, 512])

        input_data = handler.input_fn('{"item_id": ["1", "2", "3"]}', _CONTENT_TYPE)
        self.assertEqual(input_data.shape, (1, 429))
        self.assertEqual(input_data.sum(), 3)  # [1, 2, 3] -> [1, 1, 1]

        expected = {
            "item_ids": ["4", "6", "7", "20", "8", "35", "9", "19", "36", "13"],  # default_top_k=10
            "scores": [
                4.293556213378906,
                3.7262840270996094,
                3.691073417663574,
                3.6658968925476074,
                3.579657554626465,
                3.5763931274414062,
                3.4642810821533203,
                3.4230518341064453,
                3.412893772125244,
                3.298957347869873,
            ],
        }
        actual = handler.predict_fn(input_data, model)

        self.assertCountEqual(actual.keys(), expected.keys())
        self.assertSequenceEqual(actual["item_ids"], expected["item_ids"])
        torch.testing.assert_close(torch.tensor(actual["scores"]), torch.tensor(expected["scores"]))

        actual = json.loads(
            handler.output_fn(
                actual,
                _CONTENT_TYPE,
            )
        )
        self.assertSequenceEqual(actual["item_ids"], expected["item_ids"])


if __name__ == "__main__":
    absltest.main()
