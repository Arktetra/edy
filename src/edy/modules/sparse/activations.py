import torch
import torch.nn as nn

from . import SparseTensor

class SparseReLU(nn.ReLU):
    def forward(self, x: SparseTensor) -> SparseTensor:
        raise NotImplementedError("Implement Me!")

class SparseSiLU(nn.ReLU):
    def forward(self, x: SparseTensor) -> SparseTensor:
        raise NotImplementedError("Implement Me!")

class SparseGELU(nn.GELU):
    def forward(self, x: SparseTensor) -> SparseTensor:
        raise NotImplementedError("Implement Me!")

class SparseActivation(nn.Module):
    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, input: SparseTensor) -> SparseTensor:
        raise NotImplementedError("Implement Me!")
