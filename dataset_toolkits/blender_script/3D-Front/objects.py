from .utils import *
from pathlib import Path
import os
from ..render import normalize_scene

def export_scene_objs(scene_objs, out_dir: Path, exporter, exp_ext):

    for oind, obj in enumerate(scene_objs):
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)

        obj_path = str(out_dir / f"{oind}.{exp_ext}")
        exporter(filepath=obj_path, export_selected_objects=True, export_triangulated_mesh=True)


def export_all_objects(dir: Path, scene_paths, out_dir, importer, exporter, exp_ext):
    """
    provided the dir & list of filename containing scene, this will export all the object individually in every scene into out_dir
    """

    out_dir = out_dir / "objects"
    for scene_path in scene_paths:
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
        move_objects(scene_objs)

        scene_prefix = scene_path.split(".")[0]
        export_obj_dir = Path(out_dir) / scene_prefix
        os.makedirs(export_obj_dir, exist_ok=True)
        export_scene_objs(scene_objs, export_obj_dir, exporter, exp_ext)
