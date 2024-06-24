from typing import Optional

import torch

from data.ml.model_runner.metrics import utils


def mean_reciprocal_rank_at_k(y_true: torch.Tensor, y_pred: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """Computes Mean Reciprocal Rank.

    https://en.wikipedia.org/wiki/Mean_reciprocal_rank

    Args:
        y_true: Tensor of shape (batch_size, item_size). Ground truth target values.
        y_pred: Tensor of shape (batch_size, item_size). Estimated targets as returned by a model.
        k: Consider the highest k scores in the prediction. If None, use all scores.

    Returns:
        The averaged MRR scores for all samples.
    """

    utils.assert_metric_inputs(y_true, y_pred)

    k = min(y_pred.shape[-1], y_pred.shape[-1] if k is None else k)

    _, topk_idx = torch.topk(y_pred, k)
    relevance = torch.take_along_dim(y_true, topk_idx, dim=-1)
    first_relevance_pos = torch.argmax(relevance, dim=-1) + 1
    valid_mask = torch.sum(relevance, dim=-1) > 0

    return torch.mean(valid_mask / first_relevance_pos)
