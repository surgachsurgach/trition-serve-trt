import torch


def assert_metric_inputs(y_true: torch.Tensor, y_pred: torch.Tensor):
    assert y_true.ndim == 2 and y_pred.ndim == 2, "`y_true` and `y_pred` must be 2-dimensional."
    assert y_true.shape == y_pred.shape, f"`y_true` and `y_pred` must be of the same shape. Got {y_true.shape} and {y_pred.shape}."
