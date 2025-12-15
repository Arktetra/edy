import cv2
import numpy as np

from pathlib import Path
from .metadata.edy import PROCESSED_DATA_DIR


def main():
    scene_paths = list((PROCESSED_DATA_DIR / "renders").glob("**/*.png"))
    total_scenes = len(scene_paths)

    for i, scene_path in enumerate(scene_paths):
        scene_id = scene_path.parent.stem
        view_id = scene_path.stem

        masks = list((PROCESSED_DATA_DIR / "masks" / f"{scene_id}" / f"{view_id}").glob("**/*.png"))

        scene = cv2.imread(scene_path)

        for mask in masks:
            mask_img = cv2.imread(mask)
            mask_bool = mask_img > 100
            processed_mask_img = np.where(mask_bool, 255, 0)
            masked_img = mask_bool * scene

            processed_masks_dir = PROCESSED_DATA_DIR / "processed_masks" / f"{scene_id}" / f"{view_id}"
            masked_dir = PROCESSED_DATA_DIR / "masked" / f"{scene_id}" / f"{view_id}"

            if not processed_masks_dir.exists():
                Path.mkdir(processed_masks_dir, parents=True, exist_ok=True)

            if not masked_dir.exists():
                Path.mkdir(masked_dir, parents=True, exist_ok=True)

            cv2.imwrite(processed_masks_dir / f"{mask.stem}.png", processed_mask_img)
            cv2.imwrite(masked_dir / f"{mask.stem}.png", masked_img)

        if i % 50 == 0:
            print(f"Finished Masking {i}/{total_scenes} scenes.")

    print("Finished masking all scenes.")


if __name__ == "__main__":
    main()
