import torch.nn as nn
from torch import Tensor

class ResidualConnection(nn.Module):
    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        #TODO one line!
        return x + sublayer  # 입력과 sublayer의 출력을 더하는 잔차 연결