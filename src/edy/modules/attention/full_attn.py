from typing import Union, overload

import torch


@overload
def scaled_dot_product_attention(
    qkv: torch.Tensor
) -> torch.Tensor:
    ...

@overload
def scaled_dot_product_attention(
    q: torch.Tensor, kv: Union[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    ...

@overload
def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    ...

def scaled_dot_product_attention(*args, **kwargs):
    raise NotImplementedError("Implement Me!")

