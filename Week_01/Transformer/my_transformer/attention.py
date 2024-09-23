import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO
        d_k = q.size(-1)  # 키의 차원 수 (d_model / n_heads)
        
        # 쿼리와 키의 내적을 계산하고 d_k의 제곱근으로 나눠줌 (스케일링)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 마스크가 있으면, 패딩된 부분에 매우 작은 값 (-inf)을 추가하여 softmax에서 무시되도록 함
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # softmax를 통해 가중치로 변환
        attention_weights = F.softmax(scores, dim=-1)
        
        # 가중치를 값 벡터 v에 적용
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        batch_size = Q.size(0)
        
        # Query, Key, Value에 각각의 Linear layer 적용
        Q = self.query_layers(Q)  # (batch_size, seq_len, n_heads * d_model)
        K = self.key_layers(K)
        V = self.value_layers(V)
        
        # 헤드 수에 맞게 텐서의 모양을 변환
        Q = Q.view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_model)
        K = K.view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_model).transpose(1, 2)
        
        # ScaledDotProductAttention 적용
        attention_output, _ = self.attention(Q, K, V, mask)  # (batch_size, n_heads, seq_len, d_model)
        
        # 모든 헤드를 결합
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_model)
        
        # 최종 Linear layer 적용
        output = self.fc(attention_output)
        
        return output