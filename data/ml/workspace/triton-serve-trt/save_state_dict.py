import os

import torch

from data.ml.model_runner.models import bert4rec
from data.ml.model_runner.generators import item2user
from data.ml.model_runner.inference.utils import model as model_utils


_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "torch_ckpt")
os.environ["META_PATH"] = _TEST_DATA_DIR

_MAX_SEQ_LEN = 40
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = model_utils.load_model(
    _TEST_DATA_DIR,
    model_klass=bert4rec.Bert4Rec,
    generator_klass=item2user.ExclusionGenerator,
    device=_DEVICE
)

torch.save(model.state_dict(), f"{_TEST_DATA_DIR}/bert4rec.pth")
