# script specific to blender...

import bpy
import json
from ..render import normalize_scene
from mathutils import Vector
from typing import Tuple
import math


def clear_scene():
    # choose this options
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    # or choose this options
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.delete()

    # following are the additional delete incase some remained like mat, img, tex...
    # remove any collections
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection, do_unlink=True)

    # remove all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # clear all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # clear all the meshes
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)

    # clear all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def reset_all_obj_center(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN'):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.origin_set(type=type, center=center)

def object_bbox(obj) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the object.

        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
        #note: stripped version of scene_bbox
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    
    for coord in obj.bound_box:
        coord = Vector(coord)
        coord = obj.matrix_world @ coord
        bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
        bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        
    return Vector(bbox_min), Vector(bbox_max)


def reset_each_obj_center_bb(scene_objs):

    for obj in scene_objs:
        bmin, bmax = object_bbox(obj)
        # find the center location of the bounding box..
        center = (bmax + bmin) / 2
        # adjust the 3d cursor position to center of bb
        bpy.context.scene.cursor.location = center
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        # finally set the object original to the 3D cursor origin (i.e center of bounding box...)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')


def move_objects(objs, loc = (0, 0, 0)):
    """
        Provided list of objects and location (optional, default to (0, 0, 0)), it moves object to location,
        note: all the object will use same location..
    """
    for obj in objs:
        obj.location = loc


def get_mat_json(obj_mat, ind, pre_call = None):
    """
    Takes the world matrix of object and decompose it & construct the dict for json ready transform
    """
    loc, rot, scale = obj_mat.decompose()

    lo, ro, sc = [], [], []

    # if no pre_call is defined just forward the input using lambda.., no change occurs..
    # can directly have default in args if required...
    if pre_call == None:
        pre_call = lambda x: x

    for l in loc:
        lo.append(pre_call(l))

    for r in rot:
        ro.append(pre_call(r))

    for s in scale:
        sc.append(pre_call(s))

    return { f"{ind}": [lo, ro, sc] }

def json_wrapper(filepath: str, payload, indent=4):
    """
    simple helper function that dumps the json in filepath from given dict or arry of dict in payload
    """
    json_str = json.dumps(payload, indent=indent)

    with open(filepath, "w") as nf:
        nf.write(json_str)


def scene_setup():
    """
    for now normalizing the scene applying the all the transform
    """
    normalize_scene()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.select_all(action="DESELECT")

