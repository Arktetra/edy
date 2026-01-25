from typing import Dict, Tuple, List
from easydict import EasyDict as edict
from pathlib import Path

import math
import json
import huggingface_hub
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
        trans_weight: float = 2,
        rot_weight: float = 3,
        scale_weight: float = 2,
        position_weight_min_ratio: float = 0.2,
        position_weight_max_ratio: float = 1.0,
    ):
        super().__init__(ss_flow_model)
        self.ss_flow_model = ss_flow_model
        self.sigma_min = sigma_min
        self.smooth_scale = smooth_scale
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self.scale_weight = scale_weight
        self.position_weight_min_ratio = position_weight_min_ratio
        self.position_weight_max_ratio = position_weight_max_ratio
        self.unfrozen_layers = unfrozen_layers
        self.feature_encoder = feature_encoder

        self._freeze_model_params()
        print(f"Number of trainable parameters: {self.get_trainable_params()}")

    def _freeze_model_params(self):
        for param in self.ss_flow_model.parameters():
            param.requires_grad = False

        for name, param in self.ss_flow_model.named_parameters():
            for layer in self.unfrozen_layers:
                if layer in name and "lora" in name:
                    param.requires_grad = True

    def get_trainable_params(self):
        trainable_params = 0
        for param in self.ss_flow_model.parameters():
            if param.requires_grad:
                trainable_params += param.numel()
        return trainable_params

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
                pred_positions[:, :3] / scene_scale / self.smooth_scale,
                positions[:, :3] / scene_scale / self.smooth_scale,
            )
            * self.smooth_scale
        )
        rot_loss = (
            F.smooth_l1_loss(pred_positions[:, 3:7] / self.smooth_scale, positions[:, 3:7] / self.smooth_scale)
            * self.smooth_scale
        )
        scale_loss = (
            F.smooth_l1_loss(pred_positions[:, 7:] / self.smooth_scale, positions[:, 7:] / self.smooth_scale)
            * self.smooth_scale
        )
        pos_loss = trans_loss * self.trans_weight + rot_loss * self.rot_weight + scale_loss * self.scale_weight

        return trans_loss.detach(), rot_loss.detach(), scale_loss.detach(), pos_loss

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

        mse_norm = terms["mse"].detach()
        pos_norm = terms["pos_loss"].detach()
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

    def run(self, train_dataloader, val_dataloader, lr, local_dir, steps=10000, device="cpu"):
        optim = self.get_optimizer(lr)

        try:
            huggingface_hub.hf_hub_download(
                repo_id="Darktetra/edy-models", filename="train_losses.json", local_dir=local_dir
            )

            with open(f"{local_dir}/train_losses.json", "r") as f:
                train_losses = json.load(f)

            step = len(train_losses["loss"]) - 1
        except:
            step = 0
            train_losses = {
                "mse": [],
                "pos_loss": [],
                "loss": [],
            }

        try:
            huggingface_hub.hf_hub_download(
                repo_id="Darktetra/edy-models", filename="val_losses.json", local_dir=local_dir
            )

            with open(f"{local_dir}/val_losses.json", "r") as f:
                val_losses = json.load(f)
        except:
            val_losses = {
                "step": [],
                "mse": [],
                "pos_loss": [],
                "loss": [],
            }

        model_path = Path("ckpts")
        loss_path = Path(".")
        model_path.mkdir(exist_ok=True)
        loss_path.mkdir(exist_ok=True)

        while step < steps:
            if step > 0 and step % len(train_dataloader) == 0:
                self.ss_flow_model.eval()
                with torch.no_grad():
                    loss_dict = {"mse": [], "pos_loss": [], "losses": []}

                    for i, data_pack in enumerate(val_dataloader):
                        scene_image = data_pack["scene_image"][0].to(device)
                        mask_images = data_pack["mask_images"][0].to(device)
                        masked_images = data_pack["masked_images"][0].to(device)
                        positions = data_pack["positions"][0].detach()
                        ss_latents = data_pack["ss_latents"][0].detach()

                        cond = self.feature_encoder.get_cond(masked_images, scene_image, mask_images).detach()

                        losses, _ = self.training_losses(
                            ss_latents.to("cuda:1"), positions.to("cuda:1"), cond.to("cuda:1")
                        )

                        loss_dict["mse"].append(losses["mse"].detach().cpu().item())
                        loss_dict["pos_loss"].append(["pos_loss"].detach().cpu().item())
                        loss_dict["losses"].append(losses["loss"].detach().cpu().item())

                    val_losses["step"].append(step)
                    val_losses["mse"].append(sum(loss_dict["mse"]) / len(val_dataloader))
                    val_losses["pos_loss"].append(sum(loss_dict["pos_loss"]) / len(val_dataloader))
                    val_losses["losses"].append(sum(loss_dict["losses"]) / len(val_dataloader))

                    with open(loss_path / "val_losses.json", "w") as f:
                        json.dump(val_losses, f, indent=4)

                    huggingface_hub.upload_file(
                        path_or_fileobj="/kaggle/working/val_losses.json",
                        path_in_repo="val_losses.json",
                        repo_id="Darktetra/edy-models",
                    )

            self.ss_flow_model.train()
            for i, data_pack in enumerate(train_dataloader):
                scene_image = data_pack["scene_image"][0].to(device)
                mask_images = data_pack["mask_images"][0].to(device)
                masked_images = data_pack["masked_images"][0].to(device)
                positions = data_pack["positions"][0].detach()
                ss_latents = data_pack["ss_latents"][0].detach()

                cond = self.feature_encoder.get_cond(masked_images, scene_image, mask_images).detach()

                optim.zero_grad()
                losses, _ = self.training_losses(ss_latents.to("cuda:1"), positions.to("cuda:1"), cond.to("cuda:1"))
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.ss_flow_model.parameters(), max_norm=1.0)
                optim.step()

                mse = losses["mse"].detach().cpu().item()
                pos_loss = losses["pos_loss"].detach().cpu().item()
                loss = losses["loss"].detach().cpu().item()

                train_losses["mse"].append(mse)
                train_losses["pos_loss"].append(pos_loss)
                train_losses["loss"].append(loss)

                if math.isnan(loss):
                    print(f"mse: {mse}, pos: {pos_loss}, loss: {loss}")
                    print(
                        f"trans: {losses['trans_loss'].detach().cpu().item()}, "
                        f"rot: {losses['rot_loss'].detach().cpu().item()}, "
                        f"scale: {losses['scale_loss'].detach().cpu().item()}"
                    )
                    print(f"cond max and min: {cond.max()} and {cond.min()}")
                    print(
                        f"[scene_image] shape: {scene_image.shape}, min: {scene_image.min()}, max: {scene_image.max()}\n"
                        f"[mask_images] shape: {mask_images.shape}, min: {mask_images.min()}, max: {mask_images.max()}\n"
                        f"[mask_images] shape: {mask_images.shape}, min: {mask_images.min()}, max: {mask_images.max()}\n"
                        f"[positions] shape: {positions.shape}, min: {positions.min()}, max: {positions.max()}\n"
                    )
                    raise ValueError("Encountered nan.")

                if step % 5 == 0:
                    print(f"[{step}/{steps}] mse: {mse:.4f} pos_loss: {pos_loss:.4f} loss: {loss:.4f}")

                if step % 50 == 0:
                    # self.save_checkpoints(model_path / f"ss-flow-model.pt")
                    with open(loss_path / "train_losses.json", "w") as f:
                        json.dump(train_losses, f, indent=4)

                    huggingface_hub.upload_file(
                        path_or_fileobj="/kaggle/working/train_losses.json",
                        path_in_repo="train_losses.json",
                        repo_id="Darktetra/edy-models",
                    )

                    self.ss_flow_model.push_to_hub("Darktetra/edy-models")

                if step % 1000 == 0:
                    self.save_checkpoints(model_path / f"ss-flow-model-unmerged-{step}.pt")
                    # with open(loss_path / f"training-{i + 1}.json", "w") as f:
                    #     json.dump(train_losses, f, indent=4)

                    huggingface_hub.upload_file(
                        path_or_fileobj=f"/kaggle/working/ckpts/ss-flow-model-unmerged-{step}.pt",
                        path_in_repo=f"ss-flow-model-unmerged-{step}.pt",
                        repo_id="Darktetra/edy-models",
                    )

                step += 1

    def save_checkpoints(self, path: Path):
        torch.save(self.ss_flow_model.state_dict(), path)

    def get_train_loss(self):
        pass
