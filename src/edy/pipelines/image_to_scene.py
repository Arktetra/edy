import numpy as np
import torch
import utils3d

from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple

from edy.pipelines.base import Pipeline
from edy import samplers
from edy.feature_encoder import FeatureEncoder
from edy.models.sparse_structure_flow import SparseStructureFlowModel
from edy.models.sparse_structure_vae import SparseStructureDecoder
from edy.renderers.octree_renderer import OctreeRenderer
from edy.representations import Octree


class ImageToScenePipeline(Pipeline):
    """
    Pipeline for inferring Edy image-to-3D models.

    Args:

    """

    def __init__(
        self,
        ss_flow_model: Optional[SparseStructureFlowModel] = None,
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
        self.slat_sampler = slat_sampler
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
        if ss_flow_model is None:
            self.ss_flow_model = SparseStructureFlowModel.from_pretrained(
                transformer_block_type="CrossOnly", use_checkpoint=False, device=device
            )
        else:
            self.ss_flow_model = ss_flow_model
        self.ss_decoder = SparseStructureDecoder.from_pretrained(device=device).to(device)
        self.device = device

    @staticmethod
    def from_pretrained(path) -> "ImageToScenePipeline":
        return ImageToScenePipeline()

    @torch.no_grad()
    def visualize(self, z_s: torch.Tensor, positions: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
        renderer.rendering_options.bg_color = (0, 0, 0)
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
                torch.tensor([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)])
                .float()
                .cuda()
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
                device="cuda",
                primitive="voxel",
                sh_degree=0,
                primitive_config={"solid": True},
            )
            coords_vis = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords_vis.float() / resolution
            representation.depth = torch.full(
                (representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device="cuda"
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
            device="cuda",
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
        center = torch.tensor([(x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2], device="cuda")

        scene_positions = (scene_positions - center) / edge_length + 0.5

        representation.position = scene_positions
        representation.depth = torch.full(
            (representation.position.shape[0], 1), int(np.log2(512)), dtype=torch.uint8, device="cuda"
        )

        axis_length = 0.3
        num_points_per_axis = 50

        axes_positions = []
        axes_colors = []

        origin = torch.tensor([0.5, 0.5, 0.5], device="cuda")

        for i, (axis_dir, color) in enumerate(
            zip(
                [
                    torch.tensor([1.0, 0.0, 0.0], device="cuda"),  # X-axis
                    torch.tensor([0.0, 1.0, 0.0], device="cuda"),  # Y-axis
                    torch.tensor([0.0, 0.0, 1.0], device="cuda"),
                ],  # Z-axis
                [
                    torch.tensor([1.0, 0.0, 0.0], device="cuda"),  # Red
                    torch.tensor([0.0, 1.0, 0.0], device="cuda"),  # Green
                    torch.tensor([0.0, 0.0, 1.0], device="cuda"),
                ],  # Blue
            )
        ):
            line_points = torch.linspace(0, axis_length, num_points_per_axis, device="cuda")
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
                torch.full((axes_positions.shape[0], 1), int(np.log2(512)), dtype=torch.uint8, device="cuda"),
            ],
            dim=0,
        )

        # Create color map for rendering
        all_colors = torch.zeros((representation.position.shape[0], 3), device="cuda")
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
