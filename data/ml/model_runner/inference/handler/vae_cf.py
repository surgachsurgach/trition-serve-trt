"""Implementation of Pytorch Serve handler for VAE CF model used by handler service class."""

import itertools
import json
import os

from loguru import logger
import numpy as np
import torch

from data.ml.model_runner.base import model_base
from data.ml.model_runner.generators import item2user
from data.ml.model_runner.inference.utils import meta as meta_utils
from data.ml.model_runner.inference.utils import model as model_utils
from data.ml.model_runner.models import vae_cf
from data.ml.model_runner.utils import torch_utils
from data.pylib import watcher

_INFERENCE_TOP_K = os.environ.get("SM_HP_INFERENCE_TOP_K", 10)
_INPUT_ITEM_IDS = "item_ids"
_MODEL_NAME = vae_cf.VAE.__name__


@watcher.report_error
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    return model_utils.load_model(
        model_path=model_dir,
        model_klass=vae_cf.VAE,
        generator_klass=item2user.ExclusionGenerator,
        device=device,
    ).eval()


@watcher.report_error
def input_fn(
    input_data,
    content_type,
):
    """input_fn that can handle JSON formats.

    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type
    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
    """

    try:
        input_data = json.loads(input_data) if isinstance(input_data, str) else input_data
    except ValueError:
        logger.error(f"Failed to parse input_data: {input_data}")

    logger.info(f"input_data: {input_data}")

    assert _INPUT_ITEM_IDS in input_data, f"input_data must have {_INPUT_ITEM_IDS} key, but got {input_data.keys()}"

    indices = meta_utils.convert_id_to_idx(_MODEL_NAME, input_data.get(_INPUT_ITEM_IDS, []))
    refined_input = torch_utils.multi_hot_encoding(
        indices,
        meta_utils.get_item_size(_MODEL_NAME),
    ).unsqueeze(0)

    return refined_input


@watcher.report_error
def predict_fn(input_object, model: model_base.ModelBase):
    """predict_fn for PyTorch. Calls a model on data deserialized in input_fn."""

    output, _, _ = model(input_object)
    if model.exclude_inputs_from_predictions:
        output[input_object > 0] = -np.inf

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
    """Serializes predictions from predict_fn into JSON format."""
    return json.dumps(predictions)
