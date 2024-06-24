# pylint: disable=invalid-name
from typing import Callable, List

from torch import nn


# https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
class MLP(nn.Sequential):
    """A multi-layer perceptron module.
    Args:
        dims (List[int]): List of the hidden channel dimensions
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.
    """

    def __init__(
        self,
        dims: List[int],
        dropout: float = 0.3,
        activation: Callable[..., nn.Module] = nn.ReLU,
        normalization: Callable[..., nn.Module] | None = None,
        kernel_initializer: Callable = nn.init.xavier_normal_,
        bias_initializer: Callable = nn.init.zeros_,
    ):
        super().__init__()

        layers = []

        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(in_dim, out_dim))
            if normalization:
                layers.append(normalization(out_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))

        for layer in layers:
            if isinstance(layer, nn.Linear):
                kernel_initializer(layer.weight)
                bias_initializer(layer.bias)

        super().__init__(*layers)
