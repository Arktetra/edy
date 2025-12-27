from dataset_toolkits.metadata.edy import OBJECTS_DATA_DIR, SS_LATENTS_DATA_DIR
from edy.models.sparse_structure_vae import SparseStructureEncoder

import torch
import utils3d
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_voxels(object_path, resolution):
    position = utils3d.io.read_ply(object_path)[0]
    coords = ((torch.tensor(position) + 0.5) * (resolution - 1)).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


if __name__ == "__main__":
    object_paths = list(OBJECTS_DATA_DIR.glob("**/*.ply"))
    num_objects = len(object_paths)
    print(f"Total number of objects to encode: {num_objects}")
    resolution = 64
    ss_encoder = SparseStructureEncoder.from_pretrained().to(DEVICE)

    ss_encoder.eval()
    with torch.no_grad():
        for i, object_path in enumerate(object_paths):
            if i % 50 == 0:
                print(f"[{i}/{num_objects}] objects completed")
            ss = get_voxels(object_path, resolution)[None].float().to(DEVICE)
            ss_latent = ss_encoder(ss)
            dest = SS_LATENTS_DATA_DIR / object_path.parent.stem
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                SS_LATENTS_DATA_DIR / object_path.parent.stem / f"{object_path.stem}.npz", ss_latent.cpu()
            )

    print("Finished Encoding all Objects.")
