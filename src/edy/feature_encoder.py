import torch
import torch.nn.functional as F

from torchvision.transforms import v2
from PIL import Image

from vggt.models.vggt import VGGT
# from vggt.utils.load_fn import load_and_preprocess_images


class FeatureEncoder:
    def __init__(self, device):
        self.device = device
        self.dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg").to(device)
        self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        self.transform = v2.Compose([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.dinov2_model.eval()
        self.vggt_model.eval()

    def compile(self):
        print("[INFO] Compiling dinov2 model.")
        self.dinov2_model = torch.compile(self.dinov2_model)
        print("[INFO] Compiling VGGT model.")
        self.vggt_model = torch.compile(self.vggt_model)

    def preprocess_vggt_image(self, input: Image.Image) -> Image.Image:
        target_size = 518
        if input.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", input.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            input = Image.alpha_composite(background, input)
        input = input.convert("RGB")
        width, height = input.size
        if width > height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14

        input = input.resize((new_width, new_height), Image.Resampling.BICUBIC)
        input = v2.ToTensor()(input)

        h_padding = target_size - input.shape[1]
        w_padding = target_size - input.shape[2]

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            input = F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)

        input = input.unsqueeze(0).unsqueeze(0)
        input = input.to(self.device)
        return input

    @torch.no_grad()
    def encode_vggt_features(self, image):
        if isinstance(image, Image.Image):
            image = self.preprocess_vggt_image(image)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            if image.ndim == 4:
                assert image.shape[0] == 1, "Image tensor should be single image (1, C, H, W)"
            image = image.to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

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
        if masked_images.ndim == 3:
            masked_images = masked_images.unsqueeze(0)
        cond = self.encode_dinov2_features(masked_images)

        if mask_images.ndim == 3:
            mask_images = mask_images.unsqueeze(0)
        mask_cond = self.encode_dinov2_features(mask_images)
        cond = torch.cat([cond, mask_cond], dim=1)

        if scene_image.ndim == 3:
            si = scene_image.unsqueeze(0)
        assert si.ndim == 4 and si.shape[0] == 1, "si tensor must be shaped (1, C, H, W)"
        scene_cond = self.encode_dinov2_features(si)
        scene_cond = scene_cond.expand(cond.shape[0], -1, -1)
        cond = torch.cat([cond, scene_cond], dim=1)

        scene_vggt_feature = self.encode_vggt_features(scene_image)
        if scene_vggt_feature.ndim == 2:
            scene_vggt_feature = scene_vggt_feature.unsqueeze(0)
        scene_vggt_feature = torch.cat([scene_vggt_feature[..., :1024], scene_vggt_feature[..., 1024:]], dim=1)
        scene_vggt_feature = scene_vggt_feature.expand(cond.shape[0], -1, -1)
        cond = torch.cat([cond, scene_vggt_feature], dim=1)

        neg_cond = torch.zeros_like(cond)

        return {"cond": cond, "neg_cond": neg_cond}
