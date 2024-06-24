"""Transformer Encoder in Attention Is All You Need.

Paper: https://arxiv.org/abs/1706.03762
"""

from typing import Optional

from lightning import pytorch as pl
import torch
from torch import nn

from data.ml.model_runner.modules.layers import attention
from data.ml.model_runner.modules.layers import feed_forward


def _get_extended_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor | None:
    if attention_mask is None:
        extended_attention_mask = None
    elif attention_mask.dim() == 3:
        # If self-attention mask is provided as [batch_size, from_seq_length, to_seq_length],
        # we just need to make it broadcastable to all heads.
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length],
        # make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape}")

    return extended_attention_mask


class TransformerEncoderLayer(pl.LightningModule):
    """Transformer Encoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        """Constructs a Transformer encoder block.

        Args:
            d_model: Dimensionality of the model.
            num_heads: The number of attention heads for an attention layer.
            d_ff: Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
            dropout: The dropout probability for all internal layers in the encoder block.
        """
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = attention.MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff = feed_forward.FeedForward(d_model, d_ff=d_ff, dropout1=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x2 = self.norm_1(x)
        attn = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(attn)
        x2 = self.norm_2(x)
        return x + self.dropout_2(self.ff(x2))


class TransformerEncoder(pl.LightningModule):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encodes the input via Transformer encoder block.

        Args:
            x: torch.Tensor. |batch| x |seq len| x |d_model|
            mask: torch.Tensor of size |batch| x |seq_len| x |seq_len| or |batch| x |seq_len|.
                1s indicate tokens to attend to, 0s for tokens to ignore.

        Returns:
            torch.Tensor. |batch| x |seq len| x |d_model|
        """
        extended_mask = _get_extended_attention_mask(mask)
        for layer in self.layers:
            x = layer(x, extended_mask)
        return self.norm(x)
