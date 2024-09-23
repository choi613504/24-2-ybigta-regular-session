import torch
import torch.nn as nn
import math
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()
        #TODO
        # Positional Encoding 초기화
        # max_len과 d_model 크기의 텐서를 생성하고, requires_grad를 False로 설정
        self.P_E = torch.zeros(max_len, d_model)
        self.P_E.requires_grad = False  # 학습되지 않도록 설정

        # pos (0~max_len) 생성, row 방향으로 확장 (dim=1)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 2i 생성, d_model 크기에서 step=2로 짝수 인덱스들을 만듦
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float)

        # 짝수 -> sin 
        self.P_E[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))

        # 홀수 -> cos
        self.P_E[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x: Tensor) -> Tensor:
        #TODO one line!
        return self.P_E[:x.size(1), :].unsqueeze(0).repeat(x.size(0), 1, 1)
