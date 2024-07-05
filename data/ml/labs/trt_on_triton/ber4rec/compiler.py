"""Compile Pytorch Module with Torch TensorRT and save it to the model repository."""

import os

import torch
import torch_tensorrt

from data.ml.model_runner.generators import item2user
from data.ml.model_runner.inference.utils import model as model_utils
from data.ml.model_runner.models import bert4rec

_DEVICE = "cuda"

_MODEL_NAME = "bert4rec"
_MODEL_VERSION = "1"
_INPUT_MAX_SEQ_LEN = 40  # fixed when training model.

_MODEL_ROOT_DIR = os.path.dirname(__file__)
_MDOEL_CKPT_PATH = os.path.join(_MODEL_ROOT_DIR, "model_ckpt")
_MODEL_REPO_PATH = os.path.join(_MODEL_ROOT_DIR, "model_repository")
os.environ["META_PATH"] = _MDOEL_CKPT_PATH


class CustomBert4rec(bert4rec.Bert4Rec):
    def forward(
        self,
        inputs: torch.Tensor,
        target_idx: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
        pos_ids = pos_ids.unsqueeze(0).expand(batch_size, -1)

        output = self._bert(
            inputs,
            labels=targets,  # only used for training and validation steps
            position_ids=pos_ids,
            attention_mask=inputs != self._pad_token_id,
        )

        if target_idx is not None:
            gather_index = target_idx.view(-1, 1, 1).expand(-1, -1, output.logits.shape[-1])
            # replace top_k logic with fixed slicing size 10.
            output.logits = output.logits.gather(dim=1, index=gather_index).squeeze(1)[:, :-2][:, :10]

        if targets is None:
            output.loss = torch.tensor(0, dtype=torch.int8)  # dummy loss not used in predict_step.

        return output.logits, output.loss


def main():
    # 1. load model
    model = model_utils.load_model(
        _MDOEL_CKPT_PATH, model_klass=CustomBert4rec, generator_klass=item2user.ExclusionGenerator, device=_DEVICE
    )
    model.eval()

    # 2. Compile with Torch TensorRT;
    # TODO(hyesung): Support dynamic shape.
    inputs = [
        torch_tensorrt.Input(shape=[1, _INPUT_MAX_SEQ_LEN], dtype=torch.int64),
        torch_tensorrt.Input(shape=[1, 1], dtype=torch.int64),
    ]

    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={torch.half, torch.float32},
        workspace_size=2000000000,
        truncate_long_and_double=True,
    )

    inputs = [torch.rand(1, _INPUT_MAX_SEQ_LEN).long().cuda(), torch.rand(1, 1).long().cuda()]
    # 3. Save the model
    # - Save the model with TorchScript format which is supported on only C++ backend.
    torch_tensorrt.save(
        trt_model,
        f"{_MODEL_REPO_PATH}/{_MODEL_NAME}/{_MODEL_VERSION}/model.pt",
        output_format="torchscript",
        inputs=inputs,
    )
    # - Save the model with ExportedProgram format.
    # torch_tensorrt.save(trt_model, f"{model_repo_path}/model.ep", inputs=inputs)


if __name__ == "__main__":
    main()
