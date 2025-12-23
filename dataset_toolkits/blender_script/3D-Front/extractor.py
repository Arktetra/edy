# This file contains the code for extracting the 3D-Front dataset transform of each object in scene
# and extract respective object in ply format as well..

from .utils import *
from .transforms import *
from .objects import *

import argparse
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="extractor", description="Designed to extract transform prop. of all mesh/object in scene (of dataset) & export mesh individually"
    )

    cwd = Path(os.getcwd())

    parser.add_argument("--dataset-dir", type=str, default=str(cwd / "dataset"), help="Path where the dataset exists")
    parser.add_argument("--output-dir", type=str, default=str(cwd / "output"), help="Where the output will be saved")
    parser.add_argument("--ds-filter-ext", nargs='+', default=["glb"], help="dataset extension format which should only be taken from dataset folder")

    parser.add_argument("--objects", type=bool, default=False, help="Should extract objects?")
    parser.add_argument("--transforms", type=bool, default=False, help="Should extract objects transform info?")
    parser.add_argument("--at-once", type=bool, default=False, help="Should extract transform and object at single load? or seperately.")
    parser.add_argument("--all", type=bool, default=False, help="Extract all the available infos?")

    args = parser.parse_args()

    dataset_dir, output_dir = Path(args.dataset_dir), Path(args.output_dir)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Given dataset dir '{dataset_dir}' not found!!")

    if os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")
        print("*** Warn: Output path already exists override it!!! ***")
    
    os.makedirs(output_dir, exist_ok=True)

    all, at_once = args.all, args.at_once
    objects = True if all else args.objects
    transforms = True if all else args.transforms

    ext_filter = args.ds_filter_ext
    scene_paths = [item for item in os.listdir(dataset_dir) if item.split(".")[-1] in ext_filter]

    # print(args)

    # function pointer for importer and exporter
    obj_importer = bpy.ops.import_scene.gltf
    obj_exporter = bpy.ops.wm.ply_export
    exporter_ext_format = "ply" # for more flexible for sudden export format change..., warning: match the exporter and its extension

        
    if at_once:
        # transform and object extraction simulataneously in single load..
        for scene_path in scene_paths:
            # clear scene at start
            clear_scene()

            glb_path = dataset_dir / scene_path
            obj_importer(filepath=str(glb_path), merge_vertices=True, import_shading="NORMALS")

            scene_objs = bpy.data.objects["world"].children

            reset_all_obj_center()

            scene_prefix = scene_path.split(".")[0]
            trans_path = str(output_dir / f"{scene_prefix}.json")
            extract_scene_objs_transform(scene_objs, trans_path)

            move_objects(scene_objs)
            export_obj_dir = Path(output_dir) / scene_prefix
            export_scene_objs(scene_objs, export_obj_dir, obj_exporter, exporter_ext_format)

    # extract seperately otherwise
    else:
        if transforms:
            # extract_transforms()
            extract_all_transforms(dataset_dir, scene_paths)
            pass
        elif objects:
            # extract_objects()
            export_all_objects(dataset_dir, scene_paths, output_dir, obj_importer, obj_exporter, exporter_ext_format)
            pass
