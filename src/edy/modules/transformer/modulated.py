from typing import Literal, Optional, Tuple
from torch import nn

import torch
from torch.utils.checkpoint import checkpoint

from edy.modules.attention.modules import MultiHeadAttention
from edy.modules.norm import LayerNorm32


class AbsolutePositionEmbedder(nn.Module):
    """
    Creates absolute position embedding from spatial positions.
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**self.freqs)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        N, D = x.shape
        assert D == self.in_channels, "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat([embed, torch.zeros(N, self.channels - embed.shape[1], device=embed.device)], dim=-1)
        return embed


class FFN(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(channels * mlp_ratio), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


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
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
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
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            mod = mod.to(torch.float16)
            x = x.to(torch.float16)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Modulated transformer cross-attention block with adaptive layer norm
    conditioning.

    This block contains:

    1. Multi-head Self-Attention (MSA)
    2. Multi-head Cross-Attention (MCA)
    3. Feed-Forward Network (FFN)

    Args:
    ---
        channels (int): number of input channels. This is used as the hidden dimension of this transformer block.
        ctx_channels (int): number of context channels. This is used as the dimension for cross-attention keys and values.
        num_heads (int): number of heads to use in the MSA and MCA.
        mlp_ratio (float): used to determine the hidden dimension of the FFN. Defaults to `4.0`.
        attn_mode (Literal["full", "windowed"]): the attention mode to use in the attention layers. Can be "full" or "windowed". Defaults to `"full"`.
        window_size (Optional[int]): the size of window to use. Required if `attn_mode` is "windowed" defaults to `None`.
        shift_window (Optional[Tuple[int, int, int]]): optional tuple (shift_h, shift_w, shift_d) for implementing the shifted window attention. Defaults to `None`.
        use_checkpoint (bool): uses gradient checkpointing if `True`. defaults to `False`.
        use_rope (bool): uses RoPE in attention if `True`. defaults to `False`.
        qk_rms_norm (bool): uses RMS normalization in query and key projections in self-attention if True. defaults to `False`.
        qk_rms_norm_cross (bool): uses RMS normalization in query and key projections in cross-attention if True. defaults to `False`.
        qkv_bias (bool): adds a bias to the query, key and value projections if True. defaults to `True`.
        share_mod (bool): shares modulation parameters across different parts of the block if `True`. defaults to `False`.
        use_global (bool): adds a global attention if `True`. defaults to `False`.
        global_heads (Optional[int]): number of heads to use in global attention. Required if `use_global` is `True`. defaults to `None`.
        num_register_tokens (int): number of learnable register tokens to append to the input sequence. defaults to `0`.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        use_global: bool = False,
        global_heads: Optional[int] = None,
        num_register_tokens: int = 0,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.use_global = use_global
        self.num_register_tokens = num_register_tokens

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        if use_global:
            self.norm4 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
            self.norm5 = LayerNorm32(ctx_channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        if use_global:
            self.global_attn = MultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=global_heads if global_heads is not None else num_heads,
                type="global",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )
            self.global_cross_attn = MultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=global_heads if global_heads is not None else num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )

            nn.init.xavier_uniform_(self.global_attn.to_out.weight, gain=1.0)
            nn.init.xavier_uniform_(self.global_cross_attn.to_out.weight, gain=1.0)

            if self.global_attn.to_out.bias is not None:
                nn.init.normal_(self.global_attn.to_out.bias, mean=0.0, std=0.2)
            if self.global_cross_attn.to_out.bias is not None:
                nn.init.normal_(self.global_cross_attn.to_out.bias, mean=0.0, std=0.2)

            nn.init.xavier_uniform_(self.global_attn.to_qkv.weight)
            if self.global_attn.to_qkv.bias is not None:
                nn.init.normal_(self.global_attn.to_qkv.bias, mean=0.0, std=0.2)

            nn.init.xavier_uniform_(self.global_cross_attn.to_q.weight)
            if self.global_cross_attn.to_q.bias is not None:
                nn.init.normal_(self.global_cross_attn.to_q.bias, mean=0.0, std=0.2)
            nn.init.xavier_uniform_(self.global_cross_attn.to_kv.weight)
            if self.global_cross_attn.to_kv.bias is not None:
                nn.init.constant_(self.global_cross_attn.to_kv.bias, 0.1)

        self.mlp = FFN(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))
            if use_global:
                self.adaLN_modulation_global = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

                # TODO: initialize adaLN_modulation_global
                nn.init.normal_(self.adaLN_modulation_global[1].weight, mean=0.0, std=0.02)
                if self.adaLN_modulation_global[1].bias is not None:
                    nn.init.normal_(self.adaLN_modulation_global[1].bias, mean=0.0, std=0.02)

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
            if self.use_global:
                shift_msa_global, scale_msa_global, gate_msa_global = self.adaLN_modulation_global(mod).chunk(3, dim=1)

        if self.use_global:
            context_scene = context[:, 1374:, :]
            context = context[:, :1374, :]
            position_tokens = x[:, : self.num_register_tokens + 1, :]
            x = x[:, self.num_register_tokens + 1 :, :]

        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        if self.use_global:
            x = torch.cat([position_tokens, x], dim=1)
            context = torch.cat([context, context_scene], dim=1)

            h = self.norm4(x)
            h = h * (1 + scale_msa_global.unsqueeze(1)) + shift_msa_global.unsqueeze(1)
            h = self.global_attn(h)
            h = h * gate_msa_global.unsqueeze(1)
            x = x + h

            h = self.norm5(x)
            h = self.global_cross_attn(h, context)
            x = x + h

            position_tokens = x[:, : self.num_register_tokens + 1, :]
            x = x[:, self.num_register_tokens + 1 :, :]

        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        if self.use_global:
            x = torch.cat([position_tokens, x], dim=1)
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)


class ModulatedTransformerCrossOnlyBlock(nn.Module):
    """
    Modulated transformer cross-attention block with adaptive layer norm
    conditioning.

    This block contains:

    1. Multi-head Self-Attention (MSA)
    2. Multi-head Cross-Attention (MCA)
    3. Feed-Forward Network (FFN)

    Args:
    ---
        channels (int): number of input channels. This is used as the hidden dimension of this transformer block.
        ctx_channels (int): number of context channels. This is used as the dimension for cross-attention keys and values.
        num_heads (int): number of heads to use in the MSA and MCA.
        mlp_ratio (float): used to determine the hidden dimension of the FFN. Defaults to `4.0`.
        attn_mode (Literal["full", "windowed"]): the attention mode to use in the attention layers. Can be "full" or "windowed". Defaults to `"full"`.
        window_size (Optional[int]): the size of window to use. Required if `attn_mode` is "windowed" defaults to `None`.
        shift_window (Optional[Tuple[int, int, int]]): optional tuple (shift_h, shift_w, shift_d) for implementing the shifted window attention. Defaults to `None`.
        use_checkpoint (bool): uses gradient checkpointing if `True`. defaults to `False`.
        use_rope (bool): uses RoPE in attention if `True`. defaults to `False`.
        qk_rms_norm (bool): uses RMS normalization in query and key projections in self-attention if True. defaults to `False`.
        qk_rms_norm_cross (bool): uses RMS normalization in query and key projections in cross-attention if True. defaults to `False`.
        qkv_bias (bool): adds a bias to the query, key and value projections if True. defaults to `True`.
        share_mod (bool): shares modulation parameters across different parts of the block if `True`. defaults to `False`.
        use_global (bool): adds a global attention if `True`. defaults to `False`.
        global_heads (Optional[int]): number of heads to use in global attention. Required if `use_global` is `True`. defaults to `None`.
        num_register_tokens (int): number of learnable register tokens to append to the input sequence. defaults to `0`.
    """

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        use_global: bool = False,
        global_heads: Optional[int] = None,
        num_register_tokens: int = 0,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.use_global = use_global
        self.num_register_tokens = num_register_tokens

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        if use_global:
            # self.norm4 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
            self.norm4 = LayerNorm32(ctx_channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        if use_global:
            self.global_cross_attn = MultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=global_heads if global_heads is not None else num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )

            nn.init.xavier_uniform_(self.global_cross_attn.to_out.weight, gain=1.0)

            if self.global_cross_attn.to_out.bias is not None:
                nn.init.normal_(self.global_cross_attn.to_out.bias, mean=0.0, std=0.2)

            nn.init.xavier_uniform_(self.global_cross_attn.to_q.weight)
            if self.global_cross_attn.to_q.bias is not None:
                nn.init.normal_(self.global_cross_attn.to_q.bias, mean=0.0, std=0.2)
            nn.init.xavier_uniform_(self.global_cross_attn.to_kv.weight)
            if self.global_cross_attn.to_kv.bias is not None:
                nn.init.constant_(self.global_cross_attn.to_kv.bias, 0.1)

        self.mlp = FFN(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))
            if use_global:
                self.adaLN_modulation_global = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

                # TODO: initialize adaLN_modulation_global
                nn.init.normal_(self.adaLN_modulation_global[1].weight, mean=0.0, std=0.02)
                if self.adaLN_modulation_global[1].bias is not None:
                    nn.init.normal_(self.adaLN_modulation_global[1].bias, mean=0.0, std=0.02)

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
            if self.use_global:
                shift_msa_global, scale_msa_global, gate_msa_global = self.adaLN_modulation_global(mod).chunk(3, dim=1)

        if self.use_global:
            context_scene = context[:, 1374:, :]
            context = context[:, :1374, :]
            position_tokens = x[:, : self.num_register_tokens + 1, :]
            x = x[:, self.num_register_tokens + 1 :, :]

        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = x + h
        if self.use_global:
            x = torch.cat([position_tokens, x], dim=1)
            context = torch.cat([context, context_scene], dim=1)

            h = self.norm4(x)
            h = h * (1 + scale_msa_global.unsqueeze(1)) + shift_msa_global.unsqueeze(1)
            h = self.global_cross_attn(h, context)
            h = h * gate_msa_global.unsqueeze(1)
            x = x + h

            position_tokens = x[:, : self.num_register_tokens + 1, :]
            x = x[:, self.num_register_tokens + 1 :, :]

        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        if self.use_global:
            x = torch.cat([position_tokens, x], dim=1)
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, mod, context, use_reentrant=False)
        else:
            return self._forward(x, mod, context)
