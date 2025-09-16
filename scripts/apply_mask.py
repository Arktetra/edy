import numpy as np
import argparse 

from PIL import Image
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(prog="Scene Masker")
    parser.add_argument("--path", type=str, default=None, help="path to the scene and masks")

    args = parser.parse_args()

    dir = Path(args.path)

    files = [file for file in dir.glob("**/*")]

    scene = None
    masks = []

    for file in files:
        if "scene" in str(file):
            scene = file
        elif "mask" in file.stem:
            masks.append(file)

    scene_img = np.array(Image.open(scene))
    scene_mask = np.zeros(scene_img.shape[:2])

    for mask in masks:
        id = "".join(filter(str.isdigit, mask.stem))
        mask_img = np.array(Image.open(mask)) / 255.
        masked_img = Image.fromarray((mask_img[:, :, None] * scene_img).astype(np.uint8))
        masked_img.save(f"{str(dir)}/{id}.png")

        scene_mask = np.logical_or(mask_img, scene_mask)

    masked_scene = Image.fromarray((scene_mask[:, :, None] * scene_img).astype(np.uint8))
    masked_scene.save(f"{str(dir)}/masked_scene.png")


if __name__ == "__main__":
    main()
