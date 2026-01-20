import torch.nn as nn

from edy.modules.sparse.tensor import SparseTensor


class SparseReLU(nn.ReLU):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(super().forward(x.feats))


class SparseSiLU(nn.ReLU):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(super().forward(x.feats))


class SparseGELU(nn.GELU):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(super().forward(x.feats))


class SparseActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(self.activation(x.feats))
