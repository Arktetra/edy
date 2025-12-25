from .utils import *
from pathlib import Path
from ..render import normalize_scene

# other internal usages based..
transform_precision = None

def extract_scene_objs_transform(scene_objs, trans_path):
    scene_transforms = []
    for obj_ind, obj in enumerate(scene_objs):
        transforms = get_mat_json(obj.matrix_world, obj_ind, transform_precision)
        # or use any of below matrix..
        # transforms = get_mat_json(obj.matrix_basis, obj_ind)
        # transforms = get_mat_json(obj.matrix_local, obj_ind)

        scene_transforms.append(transforms)

    # transform file name should be same as the scene path prefix. (i.e before extension)
    json_wrapper(trans_path, scene_transforms)


def extract_all_transforms(dir: Path, scene_paths, out_dir: Path, importer):
    out_dir = out_dir / "transforms"

    for scene_path in scene_paths:
        # clear scene at start
        clear_scene()

        glb_path = dir / scene_path
        importer(filepath=str(glb_path), merge_vertices=True, import_shading="NORMALS")
        scene_setup()

        """ get hold of all the object in the scene.. inside world parent container..
            warning: this is specific to 3d front dataset only (test), where all the objects have world as parent
            -> world
                - obj1
                - obj2
                ...
        """
        scene_objs = bpy.data.objects["world"].children

        reset_all_obj_center()

        scene_prefix = scene_path.split(".")[0]
        trans_path = str(out_dir / f"{scene_prefix}.json")
        extract_scene_objs_transform(scene_objs, trans_path)