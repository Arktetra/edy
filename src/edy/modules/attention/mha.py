
from typing import Literal, Optional, Tuple, Union
from torch import nn
import torch

class MultiHeadRMSNorm(nn.Module):
    def __init__(
        self, dim: int, heads: int
    ):
        super().__init__()
        raise NotImplementedError("Implement me!")

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("implement me!")

class RotaryPositionEmbedder(nn.Module):
    def __init__(
        self, hidden_size: int, in_channels: int = 3
    ):
        super().__init__()
        raise NotImplementedError("Implement me!")

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("implement me!")

    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("implement me!")

    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("implement me!")

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] ="full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False
    ):
        super().__init__()
        raise NotImplementedError("Implement me!")

    @staticmethod
    def _linear(
        module: nn.Linear, x: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("Implement me!")

    @staticmethod
    def _reshape_chs(
        x: torch.Tensor, shape: Tuple[int, ...]
    ) -> torch.Tensor:
        raise NotImplementedError("Implement me!")

    def _fused_pre(
        self, x: torch.Tensor, num_fused: int
    ) -> torch.Tensor:
        raise NotImplementedError("Implement me!")

    def _rope(self, qkv: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement me!")

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError("Implement me!")
