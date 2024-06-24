import math
from typing import Callable

from lightning import pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(pl.LightningModule):
    """FeedForward layers for Transformer model."""

    def __init__(self, d_model: int, d_ff: int = 2048, activation: str = "relu", dropout1: float = 0.1, dropout2: float = 0.0):
        """Constructs FeedForward instance.

        Args:
            d_model: Dimensionality of the model.
            d_ff: Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
            dropout: The dropout probability.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout1)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout2)
        self.activation = self._get_activation(activation)

    def _get_activation(self, activation: str) -> Callable:
        activations = {
            "relu": F.relu,
            "gelu": self._gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return activations[activation]

    def _gelu(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        return self.dropout_2(x)
