import os

import torch
import torch_tensorrt

from data.ml.model_runner.models import bert4rec
from data.ml.model_runner.generators import item2user
from data.ml.model_runner.inference.utils import model as model_utils


_MAX_SEQ_LEN = 40
_DEVICE = "cpu"
_TEST_DIR = os.path.dirname(__file__)
_TEST_CKPT_DIR = os.path.join(_TEST_DIR, "torch_ckpt")
_TEST_MODEL_REPO = os.path.join(_TEST_DIR, "model_repository")

# load model
model = model_utils.load_model(
    _TEST_CKPT_DIR,
    model_klass=bert4rec.Bert4Rec,
    generator_klass=item2user.ExclusionGenerator,
    device=_DEVICE
)
model.load_state_dict(torch.load(f"{_TEST_CKPT_DIR}/bert4rec.pth"))
model.eval()

# Compile with Torch TensorRT;
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, _MAX_SEQ_LEN))],
    enabled_precisions={torch.float32}  # Run with FP32
)

# Save the model
torch.jit.save(trt_model, f"{_TEST_MODEL_REPO}/bert4rec/1/model.pt")
