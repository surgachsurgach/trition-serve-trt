"""Implementation of Pytorch Serve handler for Bert4Rec model used by handler service class."""

import copy
import itertools
import json
import os

from loguru import logger
import numpy as np
import torch

from data.ml.model_runner.generators import item2user
from data.ml.model_runner.inference.utils import meta as meta_utils
from data.ml.model_runner.inference.utils import model as model_utils
from data.ml.model_runner.models import bert4rec
from data.ml.model_runner.models import model_base
from data.ml.model_runner.utils import torch_utils
from data.pylib import watcher

_INFERENCE_TOP_K = os.environ.get("SM_HP_INFERENCE_TOP_K", 10)
_INPUT_SEQUENCE = "input_sequence"
_MODEL_INPUT = "model_input"
_TARGET_IDX = "target_idx"
_OUTPUT_EXCLUSIONS = "output_exclusions"
_MODEL_NAME = bert4rec.Bert4Rec.__name__


@watcher.report_error
def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    return model_utils.load_model(
        model_path=model_dir,
        model_klass=bert4rec.Bert4Rec,
        generator_klass=item2user.ExclusionGenerator,
        device=device,
    ).eval()


@watcher.report_error
def input_fn(
    input_data,
    content_type,
):
    try:
        input_data = json.loads(input_data) if isinstance(input_data, str) else input_data
    except ValueError:
        logger.error(f"Failed to parse input_data: {input_data}")

    logger.info(f"input_data: {input_data}")

    assert _INPUT_SEQUENCE in input_data, f"input_data must have {_INPUT_SEQUENCE} key, but got {input_data.keys()}"

    input_sequence = input_data.get(_INPUT_SEQUENCE, [])
    item_idxes = meta_utils.convert_id_to_idx(_MODEL_NAME, input_sequence)

    target_idx = len(input_sequence)  # last item should be masked token for prediction.
    item_size = meta_utils.get_item_size(_MODEL_NAME)  # also used as masked token
    model_input = copy.deepcopy(item_idxes)
    model_input.insert(target_idx, item_size)  # insert masked token to the end of sequence.
    # unsqueeze(0) to add batch dimension
    return {
        _MODEL_INPUT: torch.tensor(model_input, dtype=torch.int).unsqueeze(0),
        _OUTPUT_EXCLUSIONS: torch_utils.multi_hot_encoding(
            item_idxes,
            item_size,
        ).unsqueeze(0),
        # _TARGET_IDX: torch.tensor([target_idx]).unsqueeze(0),
    }


@watcher.report_error
def predict_fn(input_object, model: model_base.RecsysModelBase):
    output, _ = model(input_object[_MODEL_INPUT])

    if model.exclude_inputs_from_predictions:
        output[input_object[_OUTPUT_EXCLUSIONS] > 0] = -np.inf

    topk_score, topk_idx = torch.topk(output, k=_INFERENCE_TOP_K)
    topk_score, topk_idx = topk_score.detach().cpu().numpy().tolist(), topk_idx.detach().cpu().numpy().tolist()
    # Flatten the batch dimension
    topk_score, topk_idx = list(itertools.chain(*topk_score)), list(itertools.chain(*topk_idx))

    predict_result = {
        "item_ids": meta_utils.convert_idx_to_id(_MODEL_NAME, topk_idx),
        "scores": topk_score,
    }
    return predict_result


@watcher.report_error
def output_fn(predictions, response_content_type):
    return json.dumps(predictions)
