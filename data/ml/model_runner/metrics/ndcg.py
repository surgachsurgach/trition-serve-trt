from typing import Optional

import torch

from data.ml.model_runner.metrics import utils
from data.ml.model_runner.utils import torch_utils


def _dcg(y_true: torch.Tensor) -> torch.Tensor:
    batch_size, k = y_true.shape
    denom = torch.tile(torch.log2(torch.arange(k, device=y_true.device) + 2), (batch_size, 1))
    return (y_true / denom).sum(dim=-1)


def normalized_dcg_at_k(y_true: torch.Tensor, y_pred: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """Compute Normalized Discounted Cumulative Gain.

    https://en.wikipedia.org/wiki/Discounted_cumulative_gain.

    Args:
        y_true: Tensor of shape (batch_size, item_size). Ground truth target values.
        y_pred: Tensor of shape (batch_size, item_size). Estimated targets as returned by a model.
        k: Consider the highest k scores in the prediction. If None, use all scores.

    Returns:
        The averaged NDCG scores for all samples.
    """

    utils.assert_metric_inputs(y_true, y_pred)

    k = min(y_pred.shape[-1], y_pred.shape[-1] if k is None else k)

    if not isinstance(k, int) and not k > 0:
        raise ValueError("`k` should be a positive integer nor None.")

    _, topk_idx = torch.topk(y_pred, k=k)
    sorted_y_true = torch.take_along_dim(y_true, topk_idx, dim=-1)
    ideal_y_true, _ = torch.topk(y_true, k)

    ideal_dcg = _dcg(ideal_y_true)
    pred_pcd = _dcg(sorted_y_true)

    return torch_utils.div_no_nan(pred_pcd, ideal_dcg).mean()
