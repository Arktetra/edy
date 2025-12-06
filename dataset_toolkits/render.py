from concurrent.futures import ProcessPoolExecutor
import argparse
import signal
import numpy as np

from pathlib import Path

from .blender_script.render import render
from .utils import sphere_hammersley_sequence
from .metadata.front_3d import RAW_DATA_DIR, PROCESSED_DATA_DIR

SPHERE_RAIDUS = 2
CAMERA_FOV_DEGREE = 40


def handler(signal_received, frame):
    raise KeyboardInterrupt("SIGTERM received")


def _render(file_path, num_views):
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
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
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum number of workers")
    parser.add_argument("--save-mask", action="store_true", help="create masks for rendered images.")
    args = parser.parse_args()
    Path.mkdir(PROCESSED_DATA_DIR, parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, handler)

    input_paths = list(RAW_DATA_DIR.glob("**/*"))
    # _render(input_paths[0], 2)
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        try:
            executor.map(_render, input_paths, [args.num_views] * len(input_paths))
        except KeyboardInterrupt:
            print("Keyboard interrupt occurred.")
