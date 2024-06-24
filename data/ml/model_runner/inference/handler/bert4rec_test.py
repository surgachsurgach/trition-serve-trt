import json
import os
import random

from absl.testing import absltest
import numpy as np
import pytest
import torch

from data.ml.model_runner.inference.handler import bert4rec as handler
from data.ml.model_runner.models import bert4rec

_TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_data/bert4rec")
_CONTENT_TYPE = "application/json"


class HandlerTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["META_PATH"] = _TEST_DATA_DIR
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    # @pytest.mark.skip(reason="Test results are not reproducible in github actions")
    def test_transform_fn(self):
        model = handler.model_fn(_TEST_DATA_DIR)
        self.assertIsInstance(model, bert4rec.Bert4Rec)
        self.assertEqual(model.hparams["d_model"], 256)
        self.assertEqual(model.hparams["num_heads"], 4)
        self.assertEqual(model.hparams["num_encoder_layers"], 4)
        self.assertEqual(model.hparams["d_ff"], 256)

        input_data = handler.input_fn('{"input_sequence": ["1", "2", "3"]}', _CONTENT_TYPE)
        self.assertEqual(input_data["model_input"].shape, (1, 4))  # input sequence + masking token

        expected = {
            "item_ids": ["12", "9", "14", "6", "16", "15", "48", "18", "43", "29"],  # default_top_k=10
            "scores": [
                8.6917142868042,
                8.589170455932617,
                8.562948226928711,
                8.239452362060547,
                8.207462310791016,
                8.00611400604248,
                7.833898544311523,
                7.754785537719727,
                7.738406658172607,
                7.728820323944092,
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
