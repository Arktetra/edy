from typing import Union, overload

import torch

from edy.modules.sparse.tensor import SparseTensor

@overload
def sparse_scaled_dot_product_attention(
    qkv: SparseTensor
) -> SparseTensor:
    ...

@overload
def sparse_scaled_dot_product_attention(
    q: SparseTensor, kv: Union[SparseTensor, torch.Tensor]
) -> SparseTensor:
    ...

@overload
def sparse_scaled_dot_product_attention(
    q: torch.Tensor, kv: SparseTensor
) -> SparseTensor:
    ...

@overload
def sparse_scaled_dot_product_attention(
    q: SparseTensor, k: SparseTensor, v: SparseTensor
) -> SparseTensor:
    ...

@overload
def sparse_scaled_dot_product_attention(
    q: SparseTensor, k: torch.Tensor, v: torch.Tensor
) -> SparseTensor:
    ...

@overload
def sparse_scaled_dot_product_attention(
    q: torch.Tensor, k: SparseTensor, v: SparseTensor
) -> SparseTensor:
    ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    raise NotImplementedError("Implement Me!")

