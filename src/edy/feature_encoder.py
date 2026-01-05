import torch
import torch.nn.functional as F

from torchvision.transforms import v2

from vggt.models.vggt import VGGT
# from vggt.utils.load_fn import load_and_preprocess_images


class FeatureEncoder:
    def __init__(self, device):
        self.device = device
        self.dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(device)
        self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        self.transform = v2.Compose([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    @torch.no_grad()
    def encode_vggt_features(self, image):
        prediction, _ = self.vggt_model.aggregator(image[None])
        return prediction[-1][0, ...]

    @torch.no_grad()
    def encode_dinov2_features(self, image):
        image = self.transform(image).to(self.device)
        features = self.dinov2_model(image, is_training=True)["x_prenorm"]
        return F.layer_norm(features, features.shape[-1:])

    @torch.no_grad()
    def get_cond(
        self, masked_images: torch.Tensor, scene_image: torch.Tensor, mask_images: torch.Tensor
    ) -> torch.Tensor:
        cond = self.encode_dinov2_features(masked_images)

        mask_cond = self.encode_dinov2_features(mask_images)
        cond = torch.cat([cond, mask_cond], dim=1)

        if scene_image.ndim == 3:
            scene_image = scene_image.unsqueeze(0)
        assert scene_image.ndim == 4 and scene_image.shape[0] == 1, "scene_image tensor must be shaped (1, C, H, W)"
        scene_cond = self.encode_dinov2_features(scene_image)
        scene_cond = scene_cond.expand(cond.shape[0], -1, -1)
        cond = torch.cat([cond, scene_cond], dim=1)

        scene_vggt_feature = self.encode_vggt_features(scene_image)
        if scene_vggt_feature.ndim == 2:
            scene_vggt_feature = scene_vggt_feature.unsqueeze(0)
        scene_vggt_feature = torch.cat([scene_vggt_feature[..., :1024], scene_vggt_feature[..., 1024:]], dim=1)
        scene_vggt_feature = scene_vggt_feature.expand(cond.shape[0], -1, -1)
        cond = torch.cat([cond, scene_vggt_feature], dim=1)

        return cond
