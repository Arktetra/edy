from typing import Literal, Optional, Union, Tuple
from torch import nn
import torch

from edy.modules.sparse.attention.serialized_attn import SerializeMode
from edy.modules.sparse.tensor import SparseTensor

class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(
        self, dim: int, heads: int
    ):
        super().__init__()
        raise NotImplementedError("Implement me!")

    def forward(
        self, x: Union[SparseTensor, torch.Tensor]
    ) -> Union[SparseTensor, torch.Tensor]:
        raise NotImplementedError("implement me!")

class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] ="full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False
    ):
        super().__init__()
        raise NotImplementedError("Implement me!")

    @staticmethod
    def _linear(
        module: nn.Linear, x: Union[SparseTensor, torch.Tensor]
    ) -> Union[SparseTensor, torch.Tensor]:
        raise NotImplementedError("Implement me!")

    @staticmethod
    def _reshape_chs(
        x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]
    ) -> Union[SparseTensor, torch.Tensor]:
        raise NotImplementedError("Implement me!")

    def _fused_pre(
        self, x: Union[SparseTensor, torch.Tensor], num_fused: int
    ) -> Union[SparseTensor, torch.Tensor]:
        raise NotImplementedError("Implement me!")

    def _rope(self, qkv: SparseTensor) -> SparseTensor:
        raise NotImplementedError("Implement me!")

    def forward(
        self,
        x: Union[SparseTensor, torch.Tensor],
        context: Optional[Union[SparseTensor, torch.Tensor]]
    ) -> Union[SparseTensor, torch.Tensor]:
        raise NotImplementedError("Implement me!")
