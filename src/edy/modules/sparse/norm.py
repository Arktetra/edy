import torch
import torch.nn as nn

from typing import List, Union

from .tensor import SparseTensor


class SparseLayerNorm(nn.LayerNorm):
    def __init__(
        self, normalized_shape: Union[int, List[int], torch.Size], eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super(SparseLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine)

    def forward(self, x: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(x.feats)
        for i in range(x.shape[0]):
            bfeats = x.feats[x.layout[i]]
            bfeats = bfeats.permute(1, 0).reshape(1, x.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(x.shape[1], -1).permute(1, 0)
            nfeats[x.layout[i]] = bfeats
        return x.replace(bfeats)


class SparseLayerNorm32(SparseLayerNorm):
    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)
