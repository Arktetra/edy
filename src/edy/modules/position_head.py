import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from edy.modules.norm import LayerNorm32
from edy.modules.transformer.modulated import ModulatedTransformerBlock


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", s_act="linear"):
    t = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    scale = pred_pose_enc[..., 7:]

    t = base_pose_act(t, trans_act)
    quat = base_pose_act(quat, quat_act)
    scale = base_pose_act(scale, s_act)

    pred_pose_enc = torch.cat([t, quat, scale], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"Unknown act_type: {act_type}")


def inverse_log_transform(y):
    return torch.sign(y) * (torch.expm1(torch.abs(y)))


class PositionHead(nn.Module):
    def __init__(
        self,
        model_channels: int,
        trunk_depth: int = 4,
        pose_encoding_type: str = "absT_quatR_S",
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        trans_act: str = "linear",
        quat_act: str = "linear",
        s_act: str = "relu",
        use_checkpoint: bool = False,
        use_fp16: bool = False,
    ):
        super(PositionHead, self).__init__()

        if pose_encoding_type == "absT_quatR_S":
            self.target_dim = 8
        elif pose_encoding_type == "absT_eulerR_S":
            self.target_dim = 7
        else:
            raise NotImplementedError(f"Pose encoding type {pose_encoding_type} not implemented")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.s_act = s_act
        self.trunk_depth = trunk_depth
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.trunk = nn.ModuleList(
            [
                ModulatedTransformerBlock(
                    channels=model_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=self.use_checkpoint,
                    use_rope=False,
                    qk_rms_norm=True,
                    qkv_bias=True,
                    share_mod=False,
                )
                for _ in range(trunk_depth)
            ]
        )

        self.token_norm = LayerNorm32(model_channels, elementwise_affine=False, eps=1e-6)
        self.trunk_norm = LayerNorm32(model_channels, elementwise_affine=False, eps=1e-6)

        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, model_channels)

        self.adaln_norm = LayerNorm32(model_channels, elementwise_affine=False, eps=1e-6)
        self.pose_branch = nn.Sequential(
            nn.Linear(model_channels, model_channels // 2, bias=True),
            nn.GELU(),
            nn.Linear(model_channels // 2, self.target_dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0, 1e-2)
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, 1e-2)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.normal_(m.bias, 0, 1e-2)

        nn.init.zeros_(self.empty_pose_tokens)  # Translation
        nn.init.ones_(self.empty_pose_tokens[0, 3])  # Quaternion
        nn.init.ones_(self.empty_pose_tokens[0, 7])  # Scale

    def _forward(self, position_token: torch.Tensor, num_iteration: int = 4):
        position_token = self.token_norm(position_token)

        B, N, C = position_token.shape
        assert N == 1, "Position head only supports single token input"
        pred_pose_enc = None

        for _ in range(num_iteration):
            if pred_pose_enc is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, -1))
            else:
                module_input = self.embed_pose(pred_pose_enc.squeeze(1))

            position_token_block = position_token.clone()
            position_token_block = position_token_block.type(self.dtype)
            module_input = module_input.type(self.dtype)
            for block in self.trunk:
                position_token_block = block(
                    position_token_block,
                    module_input,
                )
            position_token_block = position_token_block.type(position_token.dtype)

            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(position_token_block))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            activated_pose = activate_pose(
                pred_pose_enc,
                trans_act=self.trans_act,
                quat_act=self.quat_act,
                s_act=self.s_act,
            )

        if activated_pose.ndim == 3:
            activated_pose = activated_pose.squeeze(1)
        if not self.training:
            if self.pose_encoding_type == "absT_eulerR_S":
                activated_pose[0, ...] = torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=activated_pose.device
                ).type(activated_pose.dtype)
            elif self.pose_encoding_type == "absT_quatR_S":
                activated_pose[0, ...] = torch.tensor(
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], device=activated_pose.device
                ).type(activated_pose.dtype)

        return activated_pose

    def forward(self, position_token: torch.Tensor, num_iteration: int = 4):
        if self.use_checkpoint:
            return checkpoint(
                self._forward,
                position_token,
                num_iteration=num_iteration,
                use_reentrant=False,
            )
        else:
            return self._forward(position_token, num_iteration=num_iteration)
