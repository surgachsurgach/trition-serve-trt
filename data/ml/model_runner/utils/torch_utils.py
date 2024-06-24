import torch


def multi_hot_encoding(indices: list[int], size: int, dtype=torch.float32) -> torch.Tensor:
    multi_hot = torch.zeros(size=(size,), dtype=dtype)
    multi_hot[indices] = 1.0
    return multi_hot


def multi_hot_weighted_encoding(indices: list[int], weights: list[float], size: int, dtype=torch.float32) -> torch.Tensor:
    multi_hot = torch.zeros(size=(size,), dtype=dtype)
    multi_hot[indices] = torch.tensor(weights, dtype=dtype)
    return multi_hot


def div_no_nan(tensor_a: torch.Tensor, tensor_b: torch.Tensor, na_value: float | None = 0.0) -> torch.Tensor:
    """Divides each of `tensor_a` by the corresponding element of `tensor_b`.

    After division, `NaN`, positive infinity, and negative infinity values are replaced with `nan_value`.

    Args:
        tensor_a: The dividend Tensor.
        tensor_b: The divisor Tensor.
        na_value:  Value to replace NaN, positive infinity, negative infinity.

    Returns:
        The output Tensor.
    """
    return (tensor_a / tensor_b).nan_to_num(nan=na_value, posinf=na_value, neginf=na_value)
