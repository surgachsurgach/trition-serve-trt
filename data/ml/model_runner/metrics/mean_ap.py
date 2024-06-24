from typing import Optional

import torch

from data.ml.model_runner.metrics import utils
from data.ml.model_runner.utils import torch_utils


def mean_average_precision_at_k(y_true: torch.Tensor, y_pred: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """Computes Mean Average Precision.

    https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision

    Args:
        y_true: Tensor of shape (batch_size, item_size). Ground truth target values.
        y_pred: Tensor of shape (batch_size, item_size). Estimated targets as returned by a model.
        k: Consider the highest k scores in the prediction. If None, use all scores.

    Returns:
        The averaged MAP scores for all samples.
    """

    utils.assert_metric_inputs(y_true, y_pred)

    k = min(y_pred.shape[-1], y_pred.shape[-1] if k is None else k)

    _, topk_idx = torch.topk(y_pred, k)
    relevance = torch.take_along_dim(y_true, topk_idx, dim=-1)
    cum_relevance = torch.cumsum(relevance, dim=-1)
    rank_pos = torch.arange(1, k + 1, dtype=torch.float32, device=y_true.device)
    average_precision = (relevance * cum_relevance) / rank_pos

    return torch_utils.div_no_nan(average_precision.sum(dim=-1), relevance.sum(dim=-1)).mean()
