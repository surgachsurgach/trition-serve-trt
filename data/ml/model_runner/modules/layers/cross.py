from typing import Callable

import torch
from torch import nn


class Cross(nn.Sequential):
    """Cross Layer in Deep & Cross Network to learn explicit feature interactions.
        https://www.tensorflow.org/recommenders/examples/dcn.

    Args:
        dim: dimension of the input.
        num_layers: number of cross layers to stack.
    """

    def __init__(
        self, dim: int, num_layers: int, kernel_initializer: Callable = nn.init.xavier_normal_, bias_initializer: Callable = nn.init.zeros_
    ):

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, dim))

        for layer in layers:
            if isinstance(layer, nn.Linear):
                kernel_initializer(layer.weight)
                bias_initializer(layer.bias)

        super().__init__(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        x_0 = x
        for layer in self:
            output = layer(x)
            x = x_0 * output + x
        return x
