import torch
import torch.nn as nn

from typing import List, Union

from .tensor import SparseTensor

class SparseLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super(SparseLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x: SparseTensor) -> SparseTensor:
        raise NotImplementedError("Implement Me!")

class SparseLayerNorm32(SparseLayerNorm):
    def forward(self, x: SparseTensor) -> SparseTensor:
        raise NotImplementedError("Implement Me!")
