import os

import torch
import torch_tensorrt
import numpy as np

from data.ml import bert4rec_model
from data.ml.model_runner.generators import item2user
from data.ml.model_runner.inference.utils import model as model_utils


_DEVICE="cuda"
_MAX_SEQ_LEN = 40
_TEST_DIR = os.path.dirname(__file__)
_TEST_CKPT_DIR = os.path.join(_TEST_DIR, "torch_ckpt")
_TEST_MODEL_REPO = os.path.join(_TEST_DIR, "model_repository")
os.environ["META_PATH"] = _TEST_CKPT_DIR

# load model
model = model_utils.load_model(
    _TEST_CKPT_DIR,
    model_klass=bert4rec_model.Bert4Rec,
    generator_klass=item2user.ExclusionGenerator,
    device=_DEVICE
)
model.eval()

# Compile with Torch TensorRT;
inputs = [
    torch_tensorrt.Input(shape=[1, _MAX_SEQ_LEN], dtype=torch.int64),
    torch_tensorrt.Input(shape=[1, 1], dtype=torch.int64),
]

trt_model = torch_tensorrt.compile(
    model,
    ir="dynamo",
    inputs=inputs,
    enabled_precisions={torch.float32},
    workspace_size=2000000000,
    truncate_long_and_double=True,
)

inputs = [
    torch.rand(1,40).long().cuda(),
    torch.rand(1,1).long().cuda()
]
# Save the model
# torch_tensorrt.save(trt_model, f"{_TEST_MODEL_REPO}/bert4rec/1/model.ep", inputs=inputs)
torch_tensorrt.save(trt_model, f"{_TEST_MODEL_REPO}/bert4rec/1/model.pt", output_format="torchscript", inputs=inputs)
