import argparse
import random
import signal
import numpy as np

from pathlib import Path

from .blender_script.render import render
from .metadata.front_3d import RAW_DATA_DIR, PROCESSED_DATA_DIR

SPHERE_RAIDUS = 2
CAMERA_FOV_DEGREE = 40


def handler(signal_received, frame):
    raise KeyboardInterrupt("SIGTERM received")


def _render(file_path, num_views):
    yaws = [0]
    pitchs = [np.pi / 6]
    base_y = random.uniform(0, 2 * np.pi)
    for i in range(num_views - 1):
        p = random.uniform(np.pi / 9, np.pi / 4)

        yaws.append(base_y + 2 * i * np.pi / 3)
        pitchs.append(p)

    radius = [SPHERE_RAIDUS] * num_views
    fov = [CAMERA_FOV_DEGREE / 180 * np.pi] * num_views
    views = [{"yaw": y, "pitch": p, "radius": r, "fov": f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    print("launching blender")
    render(
        output_dir=PROCESSED_DATA_DIR,
        engine="CYCLES",
        resolution=512,
        object_path=file_path,
        views=views,
        geo_mode=False,
        save_mesh=True,
        save_mask=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="render", description="Render the model in the raw directory to obtain scenes from different views."
    )
    parser.add_argument("--num-views", type=int, default=1, help="Number of views to render")
    parser.add_argument("--save-mask", action="store_true", help="create masks for rendered images.")
    args = parser.parse_args()
    Path.mkdir(PROCESSED_DATA_DIR, parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, handler)

    input_paths = list(RAW_DATA_DIR.glob("**/*"))

    for path in input_paths:
        if not (PROCESSED_DATA_DIR / "models" / f"{path.stem}.ply").exists():
            _render(path, args.num_views)
