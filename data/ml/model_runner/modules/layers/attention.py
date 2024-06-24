import math
from typing import Optional

from lightning import pytorch as pl
import torch
from torch import nn
from torch.nn import functional as F


class VanillaAttention(pl.LightningModule):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1))

    def forward(self, input_tensor):  # pylint: disable=arguments-differ
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights


class MultiHeadAttention(pl.LightningModule):
    """MultiHeadAttention for Transformer model."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """Constructs MultiHeadAttention instance.

        Args:
            d_model: Dimensionality of the model.
            num_heads: The number of attention heads for an attention layer.
        """
        super().__init__()

        assert d_model % num_heads == 0, "'d_model' should be divisible by 'num_heads'."

        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)

        # Perform linear and split int `num_heads` heads.
        # batch_size * seq_len * num_heads * d_model
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k)
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k)

        # Transpose to get dimensions batch_size * num_heads * seq_len * d_model
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        scores = torch.matmul(scores, v)

        # batch_size * seq_len * num_heads * d_model
        scores = scores.transpose(1, 2)

        concat = scores.contiguous().view(batch_size, -1, self.d_model)
        return self.out(concat)
