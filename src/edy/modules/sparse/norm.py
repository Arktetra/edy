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


class SparseGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(SparseGroupNorm, self).__init__(num_groups, num_channels, eps, affine)

    def forward(self, input: SparseTensor) -> SparseTensor:
        nfeats = torch.zeros_like(input.feats)
        for k in range(input.shape[0]):
            bfeats = input.feats[input.layout[k]]
            bfeats = bfeats.permute(1, 0).reshape(1, input.shape[1], -1)
            bfeats = super().forward(bfeats)
            bfeats = bfeats.reshape(input.shape[1], -1).permute(1, 0)
            nfeats[input.layout[k]] = bfeats
        return input.replace(nfeats)


class SparseGroupNorm32(SparseGroupNorm):
    """
    A GroupNorm layer that converts to float32 before the forward pass.
    """

    def forward(self, x: SparseTensor) -> SparseTensor:
        return super().forward(x.float()).type(x.dtype)
