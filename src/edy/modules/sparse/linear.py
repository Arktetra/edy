import torch
import torch.nn as nn

from .tensor import SparseTensor

class SparseLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ):
        super(SparseLinear, self).__init__(in_features, out_features, bias)

    def forward(
        self, x: SparseTensor
    ) -> SparseTensor:
        raise NotImplementedError("Implement Me!")
