
from torch import nn

from torch.utils.checkpoint import checkpoint

from edy.modules.attention.mha import MultiHeadAttention

class FFN(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        raise NotImplementedError("implement me!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("implement me!")

class ModulatedTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)

        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )

        self.mlp = FFN(
            channels,
            mlp_ratio=mlp_ratio,
        )

        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("implement me!")

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)
