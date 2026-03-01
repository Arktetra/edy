from pathlib import Path
from safetensors.torch import load_model
from typing import Optional, Literal, List

import huggingface_hub
import torch.nn as nn

from edy.modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from edy.modules import sparse as sp
from edy.models.slat_vae.base import SparseTransformerBase
from edy.representations.mesh import MeshExtractResult
from edy.representations.mesh import SparseFeaturesToMesh


class SparseSubdivideBlock3d(nn.Module):
    """
    A 3D subdivide block that can subdivide the sparse tensor.

    Args:
        channels: channels in the inputs and outputs.
        out_channels: if specified, the number of output channels.
        num_groups: the number of groups for the group norm.
    """
    def __init__(
        self,
        channels: int,
        resolution: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32
    ):
        super().__init__()
        self.channels = channels
        self.resolution = resolution
        self.out_resolution = resolution * 2
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels),
            sp.SparseSiLU()
        )
        
        self.sub = sp.SparseSubdivide()
        
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}"),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3, indice_key=f"res_{self.out_resolution}")),
        )
        
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1, indice_key=f"res_{self.out_resolution}")
        
    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: an [N x C x ...] Tensor of features.
        Returns:
            an [N x C x ...] Tensor of outputs.
        """
        h = self.act_layers(x)
        h = self.sub(h)
        x = self.sub(x)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        return h

class SLatDecoder(SparseTransformerBase):
    def __init__(
        self,
        resolution: int,
        model_channels: int,
        latent_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        attn_mode: Literal["full", "shift_window", "shift_sequence", "shift_order", "swin"] = "swin",
        window_size: int = 8,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        qk_rms_norm: bool = False,
        representation_config: dict = None,
    ):
        super().__init__(
            in_channels=latent_channels,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            window_size=window_size,
            pe_mode=pe_mode,
            use_fp16=use_fp16,
            use_checkpoint=use_checkpoint,
            qk_rms_norm=qk_rms_norm,
        )
        self.resolution = resolution
        self.rep_config = representation_config
        self.out_channels = self.get_feats_channels(use_color=self.rep_config.get('use_color', False))
        self.upsample = nn.ModuleList([
            SparseSubdivideBlock3d(
                channels=model_channels,
                resolution=resolution,
                out_channels=model_channels // 4
            ),
            SparseSubdivideBlock3d(
                channels=model_channels // 4,
                resolution=resolution * 2,
                out_channels=model_channels // 8
            )
        ])
        self.out_layer = sp.SparseLinear(model_channels // 8, self.out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    def get_feats_channels(self, use_color=True):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if use_color:
            # 6 channel color including normal map
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        
        start = 0
        for k, v in LAYOUTS.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        return start

    @staticmethod
    def from_pretrained():
        ckpt_path = "ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors"
        huggingface_hub.hf_hub_download(
            repo_id="haoningwu/SceneGen",
            repo_type="model",
            filename=ckpt_path,
            local_dir=Path(__file__).parents[3],
        )

        model = SLatDecoder(
            resolution=64,
            model_channels=768,
            latent_channels=8,
            num_blocks=12,
            num_heads=12,
            mlp_ratio=4,
            attn_mode="swin",
            window_size=8,
            use_fp16=True,
            representation_config={
                "use_color": True
            }
        )

        load_model(model, Path(__file__).parents[3] / ckpt_path, strict=False)

        return model

    def initialize_weights(self) -> None:
        super().initialize_weights()
        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        super().convert_to_fp16()
        self.upsample.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        super().convert_to_fp32()
        self.upsample.apply(convert_module_to_f32)  

    def forward(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        h = super().forward(x)
        for block in self.upsample:
            print(h.shape, h.feats.shape)
            h = block(h)
        h = h.type(x.dtype)
        return self.out_layer(h)


class MeshExtractor:
    def __init__(
        self,
        resolution: int,
        representation_config: dict = None,
    ):
        self.resolution = resolution
        self.rep_config = representation_config
        self.mesh_extractor = SparseFeaturesToMesh(res=self.resolution*4, use_color=self.rep_config.get('use_color', False))
        self.out_channels = self.mesh_extractor.feats_channels
    
    def to_representation(self, x: sp.SparseTensor) -> List[MeshExtractResult]:
        """
        Convert a batch of network outputs to 3D representations.

        Args:
            x: The [N x * x C] sparse tensor output by the network.

        Returns:
            list of representations
        """
        ret = []
        for i in range(x.shape[0]):
            mesh = self.mesh_extractor(x[i])
            ret.append(mesh)
        return ret
