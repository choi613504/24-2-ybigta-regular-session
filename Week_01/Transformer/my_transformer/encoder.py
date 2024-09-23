import torch.nn as nn
from torch import Tensor
from typing import Optional
from my_transformer.attention import MultiHeadAttention
from my_transformer.feedforward import FeedForwardLayer, DropoutLayer
from my_transformer.normalization import LayerNormalization
from my_transformer.residual import ResidualConnection

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
    
    def forward(self, x: Tensor) -> Tensor:
        mask = None
        #TODO
        # Self-Attention + Residual & Layer Normalization
        attn_output = self.self_attn(x, x, x, mask)  # Multi-head self-attention
        x = self.residual1(x, self.dropout1(attn_output))  # Residual connection and dropout
        x = self.norm1(x)  # Layer normalization

        # FeedForward + Residual & Layer Normalization
        ff_output = self.ff(x)  # Feed-forward network
        x = self.residual2(x, self.dropout2(ff_output))  # Residual connection and dropout
        x = self.norm2(x)  # Layer normalization

        return x