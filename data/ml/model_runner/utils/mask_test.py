import os
import random
import tempfile

from absl.testing import absltest
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision import utils as vutils

from data.ml.model_runner.utils import mask

_BATCH_SIZE = 1
_IMAGE_SIZE = 512

_totensor = transforms.ToTensor()


class MaskTest(absltest.TestCase):
    def setUp(self):
        super().setUp()
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        processed_img = mask.mask_gen(_BATCH_SIZE, _IMAGE_SIZE).type(torch.float32)[0]

        # Save Ground Truth Processed Image as .TIFF(no compressing format)
        self._tempdirname = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self._sample_hint_mask_img_path = os.path.join(self._tempdirname.name, "hint_mask.tiff")
        vutils.save_image(processed_img, self._sample_hint_mask_img_path, "TIFF")

    def tearDown(self):
        self._tempdirname.cleanup()
        return super().tearDown()

    def test_mask_gen(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        processed_img = mask.mask_gen(_BATCH_SIZE, _IMAGE_SIZE).type(torch.float32)

        assert len(processed_img.shape) == 4 and processed_img.shape[0] == _BATCH_SIZE
        assert processed_img.shape[-2] == _IMAGE_SIZE // 4 and processed_img.shape[-1] == _IMAGE_SIZE // 4

        mask_hint_img = _totensor(Image.open(self._sample_hint_mask_img_path).convert("L"))

        torch.testing.assert_close(processed_img[0], mask_hint_img)


if __name__ == "__main__":
    absltest.main()
