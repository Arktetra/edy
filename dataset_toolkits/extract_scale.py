import json
import trimesh

from dataset_toolkits.metadata.edy import OBJECTS_DATA_DIR, RAW_TRANSFORMS_DATA_DIR, TRANSFORMS_DATA_DIR


def get_scale(object_path):
    model_size = trimesh.load(object_path).bounding_box.extents
    return max(model_size)


if __name__ == "__main__":
    object_dirs = list(OBJECTS_DATA_DIR.glob("*"))

    for i, object_dir in enumerate(object_dirs):
        with open(RAW_TRANSFORMS_DATA_DIR / f"{object_dir.stem}.json", "r") as file:
            transform = json.load(file)
            t_dict = {}
            for t in transform:
                t_dict.update(t)

            for object_path in object_dir.glob("**/*.ply"):
                t_dict[object_path.stem][2] = [get_scale(object_path)]

        dest_path = TRANSFORMS_DATA_DIR / f"{object_dir.stem}.json"
        TRANSFORMS_DATA_DIR.mkdir(exist_ok=True)
        with open(dest_path, "w") as f:
            json.dump(t_dict, f, indent=4)
