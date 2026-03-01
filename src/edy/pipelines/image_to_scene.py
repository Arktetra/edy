import io
import numpy as np
import torch
import trimesh
import utils3d

from contextlib import redirect_stdout
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, List

from edy.models.slat_vae.decoder_mesh import SLatDecoder
from edy.models.structured_latent_flow import SLatFlowModel
from edy.pipelines.base import Pipeline
from edy import samplers
from edy.models.sparse_structure_flow import SparseStructureFlowModel
from edy.models.sparse_structure_vae import SparseStructureDecoder
from edy.renderers.octree_renderer import OctreeRenderer
from edy.representations import Octree
from edy.modules import sparse as sp
# from edy.utils import postprocessing_utils


def transform_from_local(local_pos, local_euler, ref_pos, ref_euler):
    # Convert euler angles to rotation matrices
    if len(ref_euler) == 4:
        ref_euler = R.from_quat(ref_euler, scalar_first=True).as_euler("xyz", degrees=True)
    if len(local_euler) == 4:
        local_euler = R.from_quat(local_euler, scalar_first=True).as_euler("xyz", degrees=True)

    rot_ref = R.from_euler("xyz", ref_euler, degrees=True).as_matrix()
    local_rot = R.from_euler("xyz", local_euler, degrees=True).as_matrix()

    # Compute global position
    global_pos = np.dot(rot_ref, local_pos) + np.array(ref_pos)

    # Compute global rotation
    global_rot = np.dot(rot_ref, local_rot)
    global_euler = R.from_matrix(global_rot).as_quat(scalar_first=True)

    return tuple(global_pos), tuple(global_euler)


def decode_slat(decoder: SLatDecoder, slat: sp.SparseTensor) -> dict:
    """
    Decode the structured latents into mesh.

    Args:
        decoder (SLatMeshDecoder): the structured latents mesh decoder.
        slat (sp.SparseTensor): the structured latents.

    Returns:
        (dict): the decoded structured latents.
    """
    return decoder(slat)


# def convert_to_glb(
#     scene,
#     positions,
#     resorted_indices: List[int] = None,
#     simplify: float = 0.95,
#     texture_size: int = 1024
# ):
#     pos_query = (0.0, 0.0, 0.0)
#     quat_query = R.from_euler('xyz', (90, 0, 0), degrees=True).as_quat(scalar_first=True)

#     local_pos, local_quat, local_scale = [], [], []

#     for i in range(1, len(positions)):
#         local_pos.append(np.array(positions[i][0:3]))
#         local_quat.append(np.array(positions[i][3:7]))
#         local_scale.append(float(positions[i][7]))

#     for i in range(len(local_pos)):
#         pos = transform_from_local(local_pos[i], local_quat[i], pos_query, quat_query)
#         local_pos[i] = pos[0]
#         local_quat[i] = pos[1]

#     positions = np.array([pos_query] + local_pos)
#     quats = np.array([quat_query] + local_quat)
#     scales = np.array([1.0] + local_scale)

#     trimeshes = []
#     for i in range(len(outputs)):
#         with redirect_stdout(io.StringIO()):  # Redirect stdout to a string buffer
#             glb = postprocessing_utils.to_glb(
#                 outputs[i]['gaussian'][0],
#                 outputs[i]['mesh'][0],
#                 # Optional parameters
#                 simplify=simplify,          # Ratio of triangles to remove in the simplification process
#                 texture_size=texture_size,      # Size of the texture used for the GLB
#             )
#         trimeshes.append(glb)

#     # Compose the output meshes into a single scene
#     scene = trimesh.Scene()
#     # Add each mesh to the scene with the appropriate transformation
#     if resorted_indices is None:
#         resorted_indices = range(len(trimeshes))

#     current_transform = np.eye(4)
#     for i in resorted_indices:
#         rmat = R.from_quat(quats[i], scalar_first=True).as_matrix() * scales[i]
#         transform = np.eye(4)
#         transform[:3, :3] = rmat
#         transform[:3, 3] = positions[i]
#         scene.add_geometry(trimeshes[i], transform=transform)
#         if i == resorted_indices[0]:
#             current_transform = transform

#     # Move the query asset to the origin with no rotation
#     R_matrix = current_transform[:3, :3]
#     t = current_transform[:3, 3]
#     T = np.eye(4)
#     T[:3, :3] = R_matrix.T
#     T[:3, 3] = -np.dot(R_matrix.T, t)
#     scene.apply_transform(T)

#     # Normalize the scene to fit within (-1, -1, -1) and (1, 1, 1) with a margin.
#     bounds = scene.bounds
#     scene_min, scene_max = bounds
#     scene_center = (scene_min + scene_max) / 2.0
#     extents = scene_max - scene_min
#     max_extent = extents.max()

#     # Define a margin (e.g., 2% margin from each side)
#     margin = 0.02
#     target_half_size = 1 - margin
#     scale_factor = target_half_size * 2 / max_extent
#     normalize_transform = trimesh.transformations.compose_matrix(
#         translate=-scene_center,
#         scale=[scale_factor, scale_factor, scale_factor]
#     )

#     scene.apply_transform(normalize_transform)

#     positions = [positions[i] for i in resorted_indices]
#     positions = np.array(positions)
#     trimeshes = [trimeshes[i] for i in resorted_indices]

#     outputs = {
#         'scene': scene,
#         'positions': positions,
#         'assets': trimeshes,
#     }

#     return outputs


class ImageToScenePipeline(Pipeline):
    """
    Pipeline for inferring Edy image-to-3D models.

    Args:

    """

    def __init__(
        self,
        ss_flow_model: Optional[SparseStructureFlowModel] = None,
        slat_flow_model: Optional[SLatFlowModel] = None,
        sparse_structure_sampler: Optional[samplers.Sampler] = None,
        slat_sampler: Optional[samplers.Sampler] = None,
        slat_normalization: Optional[dict] = None,
        # feature_encoder: Optional[FeatureEncoder] = None,
        device: str = "cpu",
    ):
        super().__init__()
        if sparse_structure_sampler is not None:
            self.sparse_structure_sampler = sparse_structure_sampler
        else:
            self.sparse_structure_sampler = samplers.FlowEulerGuidanceIntervalSamplerVGGT(sigma_min=1e-5)
        if slat_sampler is not None:
            self.slat_sampler = slat_sampler
        else:
            self.slat_sampler = samplers.FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        # if feature_encoder is None:
        #     self.feature_encoder = FeatureEncoder(device=device)
        # else:
        #     self.feature_encoder = feature_encoder
        # self.ss_flow_model = SparseStructureFlowModel.from_pretrained(
        #     transformer_block_type="CrossOnly", use_checkpoint=False, device=device
        # )
        # if ss_flow_model is None:
        #     self.ss_flow_model = SparseStructureFlowModel.from_pretrained(
        #         transformer_block_type="CrossOnly", use_checkpoint=False, device=device
        #     )
        # else:
        #     self.ss_flow_model = ss_flow_model
        # if slat_flow_model is None:
        #     self.slat_flow_model = SLatFlowModel.from_pretrained().to(device)
        # else:
        #     self.slat_flow_model = slat_flow_model.to(device)
        assert ss_flow_model is not None or slat_flow_model is not None, (
            "Either a sparse structure or structured latent flow model must be given."
        )
        self.ss_flow_model = ss_flow_model
        self.slat_flow_model = slat_flow_model
        self.ss_decoder = SparseStructureDecoder.from_pretrained(device=device).to(device)
        self.device = device

        self.slat_normalization = {
            "mean": [
                -2.1687545776367188,
                -0.004347046371549368,
                -0.13352349400520325,
                -0.08418072760105133,
                -0.5271206498146057,
                0.7238689064979553,
                -1.1414450407028198,
                1.2039363384246826,
            ],
            "std": [
                2.377650737762451,
                2.386378288269043,
                2.124418020248413,
                2.1748552322387695,
                2.663944721221924,
                2.371192216873169,
                2.6217446327209473,
                2.684523105621338,
            ],
        }

    @staticmethod
    def from_pretrained(path) -> "ImageToScenePipeline":
        return ImageToScenePipeline()

    @torch.no_grad()
    def visualize(
        self, z_s: torch.Tensor, positions: torch.Tensor, batch_size: int, bg_color: Tuple[float] = (0, 0, 0)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Visualize the sparse structure latents as rendered voxels.

        Args:
        ---
            z_s (torch.Tensor): the input sparse structure latents.


        Returns:
        ---
            Tuple[torch.Tensor, torch.Tensor]: a tuple of the rendered asset images and scene image.
        """
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = bg_color
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = "voxel"

        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []

        for yaw, pitch in zip(yaws, pitch):
            orig = (
                torch.tensor([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)]).float().cuda()
                * 2
            )
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(
                orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda()
            )
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        x_0 = self.ss_decoder(z_s)
        for i in range(batch_size):
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],
                device=self.device,
                primitive="voxel",
                sh_degree=0,
                primitive_config={"solid": True},
            )
            coords_vis = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords_vis.float() / resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device=self.device
            )

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[
                    :,
                    512 * (j // tile[1]) : 512 * (j // tile[1] + 1),
                    512 * (j % tile[1]) : 512 * (j % tile[1] + 1),
                ] = res["color"]
            images.append(image)

        scene_positions = []
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 1024
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = "voxel"
        representation = Octree(
            depth=10,
            aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0],
            device=self.device,
            primitive="voxel",
            sh_degree=0,
            primitive_config={"solid": True},
        )

        for i in range(x_0.shape[0]):
            coords_i = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            org_position = coords_i.float() / resolution
            translation = positions[i, 0:3].float()
            rotation = positions[i, 3:7].float()
            scale = positions[i, 7].float()

            centered_position = org_position - 0.5
            scaled_position = centered_position * scale

            quat_angles = rotation.cpu().numpy()
            rot_matrix_np = R.from_quat(quat_angles, scalar_first=True).as_matrix()

            rot_x_np = R.from_euler("xyz", np.array([-90, 0, 0]), degrees=True).as_matrix()
            rot_np = rot_x_np
            rot_matrix_np = np.dot(rot_matrix_np, rot_np)

            rot_matrix = torch.tensor(rot_matrix_np, device=rotation.device, dtype=rotation.dtype)
            rotated_position = torch.matmul(scaled_position, rot_matrix.T)

            final_position = rotated_position + 0.5 + translation
            scene_positions.append(final_position)

        scene_positions = torch.cat(scene_positions, dim=0)
        x_max, x_min = scene_positions[:, 0].max(), scene_positions[:, 0].min()
        y_max, y_min = scene_positions[:, 1].max(), scene_positions[:, 1].min()
        z_max, z_min = scene_positions[:, 2].max(), scene_positions[:, 2].min()

        edge_length = max(x_max - y_min, y_max - y_min, z_max - z_min)
        center = torch.tensor([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2], device=self.device)

        scene_positions = (scene_positions - center) / edge_length + 0.5

        representation.position = scene_positions
        representation.depth = torch.full(
            (representation.position.shape[0], 1), int(np.log2(512)), dtype=torch.uint8, device=self.device
        )

        axis_length = 0.3
        num_points_per_axis = 50

        axes_positions = []
        axes_colors = []

        origin = torch.tensor([0.5, 0.5, 0.5], device=self.device)

        for i, (axis_dir, color) in enumerate(
            zip(
                [
                    torch.tensor([1.0, 0.0, 0.0], device=self.device),  # X-axis
                    torch.tensor([0.0, 1.0, 0.0], device=self.device),  # Y-axis
                    torch.tensor([0.0, 0.0, 1.0], device=self.device),
                ],  # Z-axis
                [
                    torch.tensor([1.0, 0.0, 0.0], device=self.device),  # Red
                    torch.tensor([0.0, 1.0, 0.0], device=self.device),  # Green
                    torch.tensor([0.0, 0.0, 1.0], device=self.device),
                ],  # Blue
            )
        ):
            line_points = torch.linspace(0, axis_length, num_points_per_axis, device=self.device)
            for t in line_points:
                pos = origin + axis_dir * t
                axes_positions.append(pos)
                axes_colors.append(color)

        # Convert lists to tensors
        axes_positions = torch.stack(axes_positions)
        axes_colors = torch.stack(axes_colors)

        # Add axes points to the representation
        representation.position = torch.cat([representation.position, axes_positions], dim=0)
        representation.depth = torch.cat(
            [
                representation.depth,
                torch.full((axes_positions.shape[0], 1), int(np.log2(512)), dtype=torch.uint8, device=self.device),
            ],
            dim=0,
        )

        # Create color map for rendering
        all_colors = torch.zeros((representation.position.shape[0], 3), device=self.device)
        all_colors[: scene_positions.shape[0]] = scene_positions  # Original voxel positions as colors
        all_colors[scene_positions.shape[0] :] = axes_colors  # Axis colors

        # Store this color map for later use in rendering
        scene_positions = all_colors  # This will be used as colors_overwrite in rendering

        image = torch.zeros(3, 2048, 2048).cuda()
        tile = [2, 2]

        for j, (ext, intr) in enumerate(zip(exts, ints)):
            res = renderer.render(representation, ext, intr, colors_overwrite=scene_positions)
            image[
                :,
                1024 * (j // tile[1]) : 1024 * (j // tile[1] + 1),
                1024 * (j % tile[1]) : 1024 * (j % tile[1] + 1),
            ] = res["color"]
        scene_images = [image]
        scene_images = torch.stack(scene_images)
        images = torch.stack(images)

        return (images, scene_images)

    @torch.no_grad()
    def sample_sparse_structure(
        self, cond: dict, sampler_params: dict = {}, get_voxel_vis: bool = True, positions_type: str = "last"
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditions.

        Args:
        ----
            cond (dict): the conditioning information.
            sampler_params (dict): additional parameters for the sampler.
            get_voxel_vis (bool): whether to generate voxel visualization.
            positions_type (str): the type of positions to use ("avg" or "last").
        """
        res = self.ss_flow_model.resolution

        if cond["cond"].ndim == 3:
            batch_size = cond["cond"].shape[0]
        elif cond["cond"].ndim == 4:
            batch_size = cond["cond"].shape[1]

        noise = torch.randn(batch_size, self.ss_flow_model.in_channels, res, res, res).to(self.device)
        ret = self.sparse_structure_sampler.sample(self.ss_flow_model, noise, **cond, **sampler_params, verbose=True)
        z_s = ret.samples

        if positions_type == "avg":
            positions = ret.pred_pos_t
            if cond["cond"].ndim == 3:
                positions = ret.positions
            elif cond["cond"].ndim == 4:
                positions = positions[-cond["cond"].shape[0] :]
                positions = torch.stack(positions, dim=0)
                positions = torch.mean(positions, dim=0, keepdim=True).squeeze(0)
        elif positions_type == "last":
            positions = ret.positions
        else:
            raise ValueError(f"Unsupported positions type: {positions_type}")

        coords = torch.argwhere(self.ss_decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

        images, scene_images = None, None
        if get_voxel_vis:
            images, scene_images = self.visualize(z_s, positions, batch_size)

        return coords, positions, images, scene_images, ret

    def sample_slat(self, cond: dict, coords: torch.Tensor, sampler_params: dict = {}) -> sp.SparseTensor:
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], self.slat_flow_model.in_channels).to(
                self.device, dtype=cond["cond"].dtype
            ),
            coords=coords,
        )

        sampler_params = {**sampler_params}
        slat = self.slat_sampler.sample(self.slat_flow_model, noise, **cond, **sampler_params, verbose=True).samples

        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean

        return slat
