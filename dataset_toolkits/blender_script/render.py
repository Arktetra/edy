from pathlib import Path
import math
from typing import Callable, Dict, List, Tuple
import bpy
from mathutils import Vector
import numpy as np
import json
import sys

from ..types import View, SceneProps


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXPORT_FUNCTIONS: Dict[str, Callable] = {
    "ply": bpy.ops.wm.ply_export,
    "obj": bpy.ops.wm.obj_export,
    "glb": bpy.ops.export_scene.gltf,
    "gltf": bpy.ops.export_scene.gltf,
    "fbx": bpy.ops.export_scene.fbx
}

EXT = {"PNG": "png", "JPEG": "jpg", "OPEN_EXR": "exr", "TIFF": "tiff", "BMP": "bmp", "HDR": "hdr", "TARGA": "tga"}

OUTPUT_RES_PER = 100 # render res size in percentage.. i.e 100%
IMAGE_FILE_FORMAT = "PNG"
IMAGE_COLOR_MODE = "RGBA"
RENDER_SAMPLE_COUNT_NORMAL = 128 # this is the total sample taken for single image render (in cycles)
RENDER_SAMPLE_COUNT_FALLBACK = 1

def init_render(engine="CYCLES", resolution=512, geo_mode=False) -> None:
    """
    Initialize the blender settings for engine, images, cycle properties for render setup 
    
    Returns:
        None
    """
    # temporary references.. for easy access
    active_context = bpy.context
    active_render = active_context.scene.render

    active_render.engine = engine
    active_render.resolution_x = resolution
    active_render.resolution_y = resolution
    active_render.resolution_percentage = OUTPUT_RES_PER
    active_render.image_settings.file_format = IMAGE_FILE_FORMAT
    active_render.image_settings.color_mode = IMAGE_COLOR_MODE
    active_render.film_transparent = True

    engine_cycle = active_context.scene.cycles

    engine_cycle.device = "GPU" # this might cause the problem if the device doesn't have GPU listed/available to choose
    engine_cycle.samples = RENDER_SAMPLE_COUNT_NORMAL if not geo_mode else RENDER_SAMPLE_COUNT_FALLBACK
    engine_cycle.pixel_filter_type = "BOX"
    engine_cycle.filter_width = 1
    # this are the light bounces setting in cycles with hard coded values..
    engine_cycle.diffuse_bounces = 1
    engine_cycle.glossy_bounces = 1
    engine_cycle.transparent_max_bounces = 3 if not geo_mode else 0
    engine_cycle.transmission_bounces = 3 if not geo_mode else 1
    engine_cycle.use_denoising = True

    active_context.preferences.addons["cycles"].preferences.get_devices()

    comp_device = "NONE"
    if sys.platform == 'darwin':
        comp_device = "METAL"
    elif sys.platform.startswith('linux') or sys.platform == 'win32':
        comp_device = "CUDA"
    else:
        print(f"Unknown platform so fallback to NONE for {sys.platform}")

    active_context.preferences.addons["cycles"].preferences.compute_device_type = comp_device


def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def init_camera():
    active_context = bpy.context

    cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    active_context.collection.objects.link(cam)
    active_context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32

    # create constraint to look at for camera
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    # this is the empty target that camera will focus with above constrain
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0) # assuming that the scene is fixed at the center/origin point.
    active_context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty

    return cam


def init_lighting():
    # Clear existing lights, might not be necessary as all object are cleared in init_scene(), but just to be sure.. clearing the lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    active_collection = bpy.context.collection
    area_light_energy = 10000
    area_light_scale = (100, 100, 100)
    area_light_location = (0, 0, 10)

    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    active_collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)

    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    active_collection.objects.link(top_light)
    top_light.data.energy = area_light_energy
    top_light.location = area_light_location
    top_light.rotation_euler = (0, 0, 0)
    top_light.scale = area_light_scale

    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    active_collection.objects.link(bottom_light)
    bottom_light.data.energy = area_light_energy
    bottom_light.location = area_light_location * -1 # place exactly opposite to the top light
    bottom_light.scale = area_light_scale
    bottom_light.rotation_euler = (np.pi, 0, 0) # creating the bottom light facing to origin

    return {"default_light": default_light, "top_light": top_light, "bottom_light": bottom_light}


def load_object(object_path: Path) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.suffix[1:]
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    object_path = str(object_path)
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading="NORMALS")
    else:
        import_function(filepath=object_path)


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def delete_custom_normals():
    for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()


def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    new_mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    bpy.context.scene.view_layers["ViewLayer"].material_override = new_mat


def override_material_emission():
    new_mat = bpy.data.materials.new(name="EmissionOverride")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    emission = new_mat.node_tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    output = new_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
    new_mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    bpy.context.scene.view_layers["ViewLayer"].material_override = new_mat


def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)


def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")


def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")


def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    return scale, offset


def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

# emission node creating helper function
def create_emi_mat(name, color=255, strength=1):
    emi_mat = bpy.data.materials.new(name=name)
    emi_mat.use_nodes = True

    # Clear existing nodes for a clean slate
    if emi_mat.node_tree:
        emi_mat.node_tree.links.clear()
        emi_mat.node_tree.nodes.clear()

    nodes = emi_mat.node_tree.nodes
    links = emi_mat.node_tree.links

    # Create output and emission nodes
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    emission_node = nodes.new(type="ShaderNodeEmission")

    # Set emission color and strength, if not default to white, and 1
    c = color / 255
    colRGBA = (c, c, c, 1)
    emission_node.inputs["Color"].default_value = colRGBA
    emission_node.inputs["Strength"].default_value = strength

    # Link emission output to material output surface
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])

    return emi_mat


def render(
    output_dir: Path,
    engine: str,
    resolution: int,
    object_path: Path,
    views: List[View],
    geo_mode: bool = True,
    split_normal: bool = True,
    save_mesh: bool = True,
    save_mask: bool = True,
):
    """Renders the given object by rotating a camera around it.

    Args:
        output_dir (Path): the path where the outputs will be saved at.
        engine (str): blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...
        resolution (int): the resolution of the rendered images.
        object_path (Path): the path to the 3D model file to be rendered.
        views (List[View]): the views from which the object will be rendered.
        geo_mode (bool, optional): geometry mode for rendering. Defaults to True.
        split_normal (bool, optional): split the normals of the mesh. Defaults to True.
        save_mesh (bool, optional): save the mesh as a `.ply` file. Defaults to True.
        save_masks (bool, optional): save masks for rendered images.
    """
    # Initialize context
    init_render(engine, resolution, geo_mode)
    init_scene()
    load_object(object_path)

    if split_normal:
        split_mesh_normal()

    scale, offset = normalize_scene()

    cam = init_camera()
    scene_lights = init_lighting()

    if geo_mode:
        override_material()

    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": [],
    }

    # setting the background environment to black..
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

    # keeping the track of the original materials for loaded object/scene
    original_materials = {}
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            original_materials[obj.name] = []
            for mat in obj.data.materials:
                original_materials[obj.name].append(mat)

    # this will be the override materials for mask rendering, i.e two emission (white & black)
    mask_materials = {
        "emission": create_emi_mat("emission"),
        "default": create_emi_mat("def_emi", 0, 0)
    }

    # selecting the objects that are to be rendered (i.e inside world objects) & its count
    objects = bpy.context.scene.collection.all_objects["world"].children
    objects_count = len(objects)

    for i, view in enumerate(views):
        cam.location = (
            view["radius"] * np.cos(view["yaw"]) * np.cos(view["pitch"]),
            view["radius"] * np.sin(view["yaw"]) * np.cos(view["pitch"]),
            np.abs(view["radius"] * np.sin(view["pitch"])),
        )
        cam.data.lens = 16 / np.tan(view["fov"] / 2)

        # enabling all the lights to render the original scene
        for light_key in scene_lights:
            scene_lights[light_key].hide_render = False

        bpy.context.scene.render.filepath = str(output_dir / "renders" / object_path.stem / f"{i}.png")

        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()

        metadata = {
            "file_path": f"{i}.png",
            "camera_angle_x": view["fov"],
            "transform_matrix": get_transform_matrix(cam),
        }

        if save_mask:
            # disabling all the lights to render the mask case
            for light_key in scene_lights:
                scene_lights[light_key].hide_render = True

            # first clear all the material attached to the object to prepare for mask
            for obj in objects:
                obj.data.materials.clear()

            # for each object from the single view, set emission mat to active object and default to rest
            for ind in range(objects_count):
                # first setting all object materials to default..
                for obj in objects:
                    obj.active_material = mask_materials["default"]
                    obj.visible_diffuse = True  # reset all to true..

                # now set the material for mask case... (emission with ray visibility diffuse set to false)
                active_obj = objects[ind]
                active_obj.active_material = mask_materials["emission"]
                active_obj.visible_diffuse = False  # this is to ensure that light bounces doesn't occur..

                bpy.context.scene.render.filepath = str(output_dir / "masks" / object_path.stem / f"{i}/{ind}.png")
                bpy.ops.render.render(write_still=True)

            # now clear all the mask mat. attached to the object & set the original materials of objects
            for obj_name, mats in original_materials.items():
                obj = bpy.data.objects.get(obj_name)
                if obj and obj.type == "MESH":
                    obj.data.materials.clear()
                    # Re-assign original materials
                    for mat in mats:
                        obj.data.materials.append(mat)

            # this will only be useful if material override method.. is used before mask usage and required disabling for the original render.
            # bpy.context.scene.view_layers["ViewLayer"].material_override = None

        to_export["frames"].append(metadata)

        with open(output_dir / "renders" / object_path.stem / "transforms.json", "w") as f:
            json.dump(to_export, f, indent=4)

    if save_mesh:
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()

        # if all objects are required (with light and stuffs) to export then uncomment below.. with use_selection=False in export below..
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)

        export_function = EXPORT_FUNCTIONS["ply"]

        # filename is same as the object name
        ply_filename = object_path.name.split(".")[0]
        export_function(filepath=output_dir / "models" / f"{ply_filename}.ply", use_selection=True)
