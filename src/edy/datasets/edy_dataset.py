from pathlib import Path
from typing import List
from torch.utils.data import Dataset

from PIL import Image

import json
import numpy as np
import random
import torch


class EdyDataset(Dataset):
    def __init__(self, root: Path, image_size: int = 518):
        self.root = root
        self.scene_path = self.root / "renders"
        self.masks_path = self.root / "processed_masks"
        self.masked_path = self.root / "masked"
        self.positions_path = self.root / "transforms"
        self.ss_latents_path = self.root / "ss_latents"
        self.image_size = image_size

    def __len__(self):
        return len(list(self.ss_latents_path.glob("*")))

    def __getitem__(self, idx: int):
        scene_image_path = self.scene_path / f"{idx}" / "0.png"
        num_objects = len(list((self.ss_latents_path / f"{idx}").glob("**/*.npz")))
        object_order = list(range(num_objects))
        random.shuffle(object_order)
        mask_image_paths = [(self.masks_path / f"{idx}" / "0" / f"{i}.png") for i in object_order]
        masked_image_paths = [(self.masked_path / f"{idx}" / "0" / f"{i}.png") for i in object_order]
        ss_latents_paths = [(self.ss_latents_path / f"{idx}" / f"{i}.npz") for i in object_order]
        positions_path = self.positions_path / f"{idx}.json"

        scene_image = self.process_scene_image(scene_image_path)
        mask_images = self.process_mask_images(mask_image_paths)
        masked_images = self.process_masked_images(masked_image_paths)

        ss_latents = [np.load(ss_latents_path)["arr_0"] for ss_latents_path in ss_latents_paths]
        # print(ss_latents[0].shape)
        ss_latents = torch.tensor(np.stack(ss_latents, axis=0))

        positions = torch.empty((num_objects, 8))

        with open(positions_path) as f:
            position = json.load(f)

        for obj in object_order:
            positions[obj, :3] = torch.tensor(position[f"{obj}"][0])
            positions[obj, 3:7] = torch.tensor(position[f"{obj}"][1])
            positions[obj, 7:] = torch.tensor(position[f"{obj}"][2])

        return {
            "scene_image": scene_image,
            "mask_images": mask_images,
            "masked_images": masked_images,
            "positions": positions,
            "ss_latents": ss_latents,
        }

    def process_scene_image(self, scene_image_path: Path):
        scene_image = Image.open(scene_image_path)
        scene_image = scene_image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        scene_image = scene_image.convert("RGB")
        return torch.tensor(np.array(scene_image)).permute(2, 0, 1).float() / 255.0

    def process_mask_images(self, mask_image_paths: List[Path]):
        mask_images = [Image.open(mask_image_path) for mask_image_path in mask_image_paths]
        mask_images = [
            mask_image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            for mask_image in mask_images
        ]
        mask_images = [
            torch.tensor(np.array(mask_image)).permute(2, 0, 1).float() / 255.0 for mask_image in mask_images
        ]
        return torch.stack(mask_images, dim=0)

    def process_masked_images(self, masked_image_paths: List[Path]):
        masked_images = [Image.open(image_path) for image_path in masked_image_paths]
        bboxes = [np.array(image) for image in masked_images]
        bboxes = [[bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()] for bbox in bboxes]
        centers = [[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in bboxes]
        hsizes = [max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2 for bbox in bboxes]
        aug_size_ratio = 1.2
        aug_hsizes = [hsize * aug_size_ratio for hsize in hsizes]
        aug_center_offsets = [[0, 0] for _ in range(len(masked_images))]
        aug_centers = [
            [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
            for center, aug_center_offset in zip(centers, aug_center_offsets)
        ]
        aug_bboxes = [
            [
                int(aug_center[0] - aug_hsize),
                int(aug_center[1] - aug_hsize),
                int(aug_center[0] + aug_hsize),
                int(aug_center[1] + aug_hsize),
            ]
            for aug_center, aug_hsize in zip(aug_centers, aug_hsizes)
        ]
        masked_images = [image.crop(aug_bbox) for image, aug_bbox in zip(masked_images, aug_bboxes)]
        masked_images = [
            image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS) for image in masked_images
        ]
        masked_images = [image.convert("RGB") for image in masked_images]
        masked_images = [torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0 for image in masked_images]
        return torch.stack(masked_images, dim=0)
