from typing import Optional

import torch

from data.ml.model_runner.metrics import utils
from data.ml.model_runner.utils import torch_utils


def recall_at_k(y_true: torch.Tensor, y_pred: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """Compute the recall.

    https://en.wikipedia.org/wiki/Discounted_cumulative_gain.

    Args:
        y_true: Tensor of shape (batch_size, item_size). Ground truth target values.
        y_pred: Tensor of shape (batch_size, item_size). Estimated targets as returned by a model.
        k: Consider the highest k scores in the prediction. If None, use all scores.

    Returns:
        The averaged recall of the positive class in binary classification.
    """

    utils.assert_metric_inputs(y_true, y_pred)

    k = min(y_pred.shape[-1], y_pred.shape[-1] if k is None else k)

    _, topk_idx = torch.topk(y_pred, k)
    y_true[y_true > 0] = 1.0
    relevance = torch.take_along_dim(y_true, topk_idx, dim=-1).sum(dim=-1).float()

    return torch_utils.div_no_nan(relevance, y_true.sum(dim=-1)).mean()
