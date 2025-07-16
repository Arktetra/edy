import sys
import numpy as np
import json

from PIL import Image
from typing import NamedTuple
from .colmap_loader import (
    read_extrinsics_text,
    read_extrinsics_binary,
    read_intrinsics_text,
    read_intrinsics_binary,
    read_points3D_text,
    read_points3D_binary,
    qvec2rotmat
)
from utils.graphics import focal_to_fov, fov_to_focal

from pathlib import Path
from plyfile import PlyData, PlyElement
from scene.point_cloud import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovX: float
    FovY: float
    image_path: str
    image_name: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud | None
    train_cameras: list
    test_cameras: list
    ply_path: Path

def store_ply(path: Path, xyz, rgb):
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('blue', 'u2'), ('blue', 'u1')
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetch_ply(path: Path):
    ply_data = PlyData.read(path)
    vertices = ply_data["vertex"]
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['blue'], vertices['green']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def read_colmap_cameras(
    cam_extrinsics, cam_intrinsics, images_dir
):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extrinsic = cam_extrinsics[key]
        intrinsic = cam_intrinsics[extrinsic.camera_id]

        height = intrinsic.height
        width = intrinsic.width

        uid = intrinsic.id
        R = np.transpose(qvec2rotmat(extrinsic.qvec))
        T = np.array(extrinsic.tvec)

        if intrinsic.model == "SIMPLE_PINHOLE":
            focal_length_x = intrinsic.params[0]
            FovY = focal_to_fov(focal_length_x, height)
            FovX = focal_to_fov(focal_length_x, width)
        elif intrinsic.model=="PINHOLE":
            focal_length_x = intrinsic.params[0]
            focal_length_y = intrinsic.params[1]
            FovY = focal_to_fov(focal_length_y, height)
            FovX = focal_to_fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras)are supported!"

        # n_remove = len(extrinsic.name.split['.'][-1]) + 1

        image_path = images_dir / extrinsic.name
        image_name = extrinsic.name

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovX=FovX,
            FovY=FovY,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            is_test=image_name
        )

        cam_infos.append(cam_info)

    return cam_infos



def read_colmap_scene_info(path: Path, images: str, eval, train_test_exp):
    """
    Read COLMAP scene info from a image set processed by COLMAP.

    args:
    ---
    path (Path): path to the COLMAP processed image set.
    images (str): image directory in the path.
    """
    try:
        cameras_extrinsic_file = path / "sparse/0/images.bin"
        cameras_intrinsic_file = path / "sparse/0/cameras.bin"
    except:
        cameras_extrinsic_file = path / "sparse/0/images.txt"
        cameras_intrinsic_file = path / "sparse/0/cameras.txt"


    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    images_dir = "images" if images == None else images
    
    cam_infos_unsorted = read_colmap_cameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_dir=path / images_dir,
    )

    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    ply_path = path / "sparse/0/points3D.ply"
    bin_path = path / "sparse/0/points3d.bin"
    txt_path = path / "sparse/0/points3d.txt"

    if not ply_path.exists():
        print("Converting point3D.bin to point3D.ply. This will only happen the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        store_ply(ply_path, xyz, rgb)
    try:
        pcd = fetch_ply(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        ply_path=ply_path
    )

    return scene_info
