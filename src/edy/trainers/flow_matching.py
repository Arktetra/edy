from typing import Dict, Tuple, List
from easydict import EasyDict as edict
from pathlib import Path

import torch
import torch.nn.functional as F

from edy.feature_encoder import FeatureEncoder
from edy.models.sparse_structure_flow import SparseStructureFlowModel
from edy.trainers.base import Trainer


class FlowMatchingTrainer(Trainer):
    def __init__(
        self,
        ss_flow_model: SparseStructureFlowModel,
        unfrozen_layers: List[str],
        feature_encoder: FeatureEncoder,
        sigma_min: float = 1e-5,
        smooth_scale: float = 0.02,
        position_weight_min_ratio: float = 0.2,
        position_weight_max_ratio: float = 1.0,
    ):
        super().__init__()
        self.ss_flow_model = ss_flow_model
        self.sigma_min = sigma_min
        self.smooth_scale = smooth_scale
        self.position_weight_min_ratio = position_weight_min_ratio
        self.position_weight_max_ratio = position_weight_max_ratio
        self.unfrozen_layers = unfrozen_layers
        self.feature_encoder = feature_encoder

        self._freeze_model_params()

    def _freeze_model_params(self):
        for param in self.ss_flow_model.parameters():
            param.requires_grad = False

        for name, param in self.ss_flow_model.named_parameters():
            for layer in self.unfrozen_layers:
                if layer in name:
                    param.requires_grad = True

    def get_optimizer(self, lr):
        return torch.optim.Adam(params=self.ss_flow_model.parameters(), lr=lr)

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps in [0, 1].

        Args:
            batch_size (int): The size of the batch.

        Returns:
            torch.Tensor: timestep tensor
        """
        return torch.sigmoid(torch.randn(batch_size))

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Diffuse data for the given number of timesteps.

        Args:
            x_0 (torch.Tensor): The [N x C x ...] tensor of noiseless inputs.
            t (torch.Tensor): The [N] tensor of diffusion steps in [0, 1].
            noise (torch.Tensor): The [N x C x ...] tensor of noise.

        Returns:
            torch.Tensor: the noisy version of x_0 under timestep t.
        """
        noise = torch.randn_like(x_0)

        t = t.view(-1, *[1 for _ in range(x_0.ndim - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.

        Args:
            x_0 (torch.Tensor): The [N x C x ...] tensor of noiseless inputs.
            t (torch.Tensor): The [N] tensor of diffusion steps in [0, 1].
            noise (torch.Tensor): The [N x C x ...] tensor of noise.

        Returns:
            torch.Tensor: the target velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0  # basically the differentiation of diffusion process wrt t.

    def get_position_loss(
        self, positions: torch.Tensor, pred_positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get position loss.

        Args:
            positions (torch.Tensor): target position values.
            pred_positions (torch.Tensor): predicted position values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: translation, rotation, scale and position loss.
        """
        scene_scale = (
            torch.max(positions[0:, :3] + positions[0:, 7].unsqueeze(-1) / 2, dim=0).values
            - torch.min(positions[0:, :3] - positions[0:, 7].unsqueeze(-1) / 2, dim=0).values
        )
        min_val = torch.tensor(1e-2, device=scene_scale.device, dtype=scene_scale.dtype)
        scene_scale = torch.max(scene_scale, min_val)

        trans_loss = (
            F.smooth_l1_loss(
                pred_positions[1:, :3] / scene_scale / self.smooth_scale,
                positions[1:, :3] / scene_scale / self.smooth_scale,
            )
            * self.smooth_scale
        )
        rot_loss = (
            F.smooth_l1_loss(pred_positions[1:, 3:7] / self.smooth_scale, positions[1:, 3:7] / self.smooth_scale)
            * self.smooth_scale
        )
        scale_loss = (
            F.smooth_l1_loss(pred_positions[1:, 7:] / self.smooth_scale, positions[1:, 7:] / self.smooth_scale)
            * self.smooth_scale
        )
        pos_loss = trans_loss * self.trans_weight + rot_loss * self.rot_weight + scale_loss * self.scale_weight

        return trans_loss, rot_loss, scale_loss, pos_loss

    def training_losses(
        self, x_0: torch.Tensor, positions: torch.Tensor, cond: torch.Tensor = None
    ) -> Tuple[Dict, Dict]:
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise)

        pred, pred_positions = self.ss_flow_model(x_t, t * 1000, cond)
        target = self.get_v(x_0, noise, t)

        assert pred.shape == noise.shape == x_0.shape

        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["trans_loss"], terms["rot_loss"], terms["scale_loss"], terms["pos_loss"] = self.get_position_loss(
            positions, pred_positions
        )

        mse_norm = terms["mse"]
        pos_norm = terms["pos_loss"]
        ratio = mse_norm / (pos_norm + 1e-8)

        if not hasattr(self, "ema_pos_weight"):
            self.ema_pos_weight = ratio
        else:
            decay = 0.99
            self.ema_pos_weight = decay * self.ema_pos_weight + (1 - decay) * ratio

        pos_weight = torch.clamp(
            self.ema_pos_weight, min=self.position_weight_min_ratio, max=self.position_weight_max_ratio
        )

        terms["loss"] = terms["mse"] + terms["pos_loss"] * pos_weight

        return terms, {}

    def run(self, train_dataloader, lr, epochs=10):
        optim = self.get_optimizer(lr)
        train_losses = {
            "mse": [],
            "pos_loss": [],
            "loss": [],
        }
        total_steps = epochs * len(train_dataloader)

        self.ss_flow_model.eval()

        for epoch in epochs:
            for i, data_pack in enumerate(train_dataloader):
                scene_image = data_pack["scene_image"]
                mask_images = data_pack["mask_images"]
                masked_images = data_pack["masked_images"]
                positions = data_pack["positions"]
                ss_latents = data_pack["ss_latents"]

                cond = self.feature_encoder.get_cond(masked_images, scene_image, mask_images)

                optim.zero_grad()
                losses, _ = self.training_losses(ss_latents, positions, cond)
                optim.step()
                losses["loss"].backward()

                train_losses["mse"].detach().cpu().append(losses["mse"])
                train_losses["pos_loss"].detach().cpu().append(losses["pos_loss"])
                train_losses["loss"].detach().cpu().append(losses["loss"])

                if (i + 1) % 50 == 0:
                    print(
                        f"[{epoch * i + i + 1}/{total_steps}] mse: {train_losses['mse']:.4f} pos_loss: {train_losses['pos_loss']} loss: {train_losses['loss']}"
                    )

    def save_checkpoints(self, path: Path):
        torch.save(self.ss_flow_model.state_dict(), path)
