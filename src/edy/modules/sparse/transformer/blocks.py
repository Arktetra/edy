import torch.nn as nn

from edy.modules.sparse.activations import SparseGELU
from edy.modules.sparse.linear import SparseLinear
from edy.modules.sparse.tensor import SparseTensor


class SparseFFN(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            SparseLinear(channels, int(channels * mlp_ratio)),
            SparseGELU(approximate="tanh"),
            SparseLinear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return self.mlp(x)
