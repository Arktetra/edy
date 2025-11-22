from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from edy.modules.sparse.attention.mha import SparseMultiHeadAttention
from edy.modules.sparse.attention.serialized_attn import SerializeMode
from edy.modules.sparse.norm import SparseLayerNorm32
from edy.modules.sparse.tensor import SparseTensor


class SparseFFN(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        raise NotImplementedError("Implement Me!")

    def forward(self, x: SparseTensor) -> SparseTensor:
        raise NotImplementedError("Implement Me!")

class ModulatedSparseTransformerCrossBlock(nn.Module):
    """Modulated Sparse Transformer Cross-Attention Block"""
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = SparseFFN(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, context: torch.Tensor):
        raise NotImplementedError("Implement Me!")

    def forward(self, x: SparseTensor, context: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, context, use_reentrant=False)
        else:
            return self._forward(x, context)

