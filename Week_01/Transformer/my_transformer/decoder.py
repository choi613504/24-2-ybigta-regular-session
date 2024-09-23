import torch
import torch.nn as nn
from typing import Optional
from my_transformer.attention import MultiHeadAttention
from my_transformer.feedforward import FeedForwardLayer, DropoutLayer
from my_transformer.normalization import LayerNormalization
from my_transformer.residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        #TODO
        self.self_attn = MultiHeadAttention(d_model, n_heads)  # Self-attention
        self.cross_attn = MultiHeadAttention(d_model, n_heads)  # Encoder-Decoder attention
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.dropout3 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
        self.residual3 = ResidualConnection()
    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        # Self-Attention + Residual & Layer Normalization
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.residual1(x, self.dropout1(self_attn_output))
        x = self.norm1(x)

        # Encoder-Decoder Attention + Residual & Layer Normalization
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = self.residual2(x, self.dropout2(cross_attn_output))
        x = self.norm2(x)

        # FeedForward + Residual & Layer Normalization
        ff_output = self.ff(x)
        x = self.residual3(x, self.dropout3(ff_output))
        x = self.norm3(x)

        return x