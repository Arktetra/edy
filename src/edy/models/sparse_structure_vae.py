from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from edy.modules.norm import LayerNorm32
from edy.modules.spatial import pixel_shuffle_3d
from edy.modules.utils import convert_module_to_f16, convert_module_to_f32, zero_module


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = LayerNorm32(self.channels)
        self.norm2 = LayerNorm32(self.out_channels)
        self.conv1 = nn.Conv3d(self.channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        )
        self.skip_connection = nn.Conv3d(self.channels, self.out_channels, 1) if channels != self.out_channels else nn.Identity()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h

class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UpSampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return pixel_shuffle_3d(x, 2)

class SparseStructureEncoder(nn.Module):
    """
    Encoder for Sparse Structure.

    Args:
        in_channels (int): Channels of the input.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the encoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        use_fp16 (bool): Whether to use FP16.
    """
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(
            in_channels, 
            out_channels=channels[0], 
            kernel_size=3, 
            padding=1
        )

        self.blocks = nn.ModuleList([])

        for i, channel in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(channel, channel)
                for _ in range(num_res_blocks)
            ])
            if i < len(channels) - 1:
                self.blocks.append(
                    DownsampleBlock3d(channel, channels[i + 1])
                )

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[-1], channels[-1])
            for _ in range(num_res_blocks_middle)
        ])

        self.out_layer = nn.Sequential(
            LayerNorm32(channels[-1]),
            nn.SiLU(),
            nn.Conv3d(
                in_channels=channels[-1],
                out_channels=latent_channels * 2,
                kernel_size=3,
                padding=1
            )
        )       

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(
        self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_layer(x)
        h = h.type(self.dtype)

        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)

        h = h.type(x.dtype)
        h = self.out_layer(h)

        mean, logvar = h.chunk(2, dim=1)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        if return_raw:
            return z, mean, logvar
        
        return z

class SparseStructureDecoder(nn.Module):
    """
    Decoder for Sparse Structure.

     Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        use_fp16 (bool): Whether to use FP16.
    """
    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(
            in_channels=latent_channels,
            out_channels=channels[0],
            kernel_size=3,
            padding=1,
        )

        self.middle_block = nn.Sequential(*[
            ResBlock3d(channels[0], channels[0])
            for _ in range(num_res_blocks_middle)
        ])

        self.blocks = nn.ModuleList([])

        for i, channel in enumerate(channels):
            self.blocks.extend([
                ResBlock3d(channel, channel)
                for _ in range(num_res_blocks)
            ])

            if i < len(channels) - 1:
                self.blocks.append(
                    UpSampleBlock3d(channel, channels[i + 1])
                )

        self.out_layer = nn.Sequential(
            LayerNorm32(channels[-1]),
            nn.SiLU(),
            nn.Conv3d(
                in_channels=channels[-1],
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device
    
    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)

        h = h.type(self.dtype)

        h = self.middle_block(h)

        for block in self.blocks:
            h = block(h)

        h = h.type(x.dtype)
        return self.out_layer(h)
