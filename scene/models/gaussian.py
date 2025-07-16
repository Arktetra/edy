import torch

from utils.activations import inverse_sigmoid
from utils.graphics import scaling_rotation

class GaussianModel:
    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.optimizer_type = optimizer_type
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self.max_radii2d = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def setup_functions(self):
        """
        Creates functions for use.
        """
        def scaling_rotation_to_covariance(
            scaling: torch.Tensor,
            scaling_modifier: float,
            rotation: torch.Tensor
        ) -> torch.Tensor:
            """
            Creates covariance matrix from scaling and rotation.

            Args:
            -----
                scaling (torch.Tensor): Nx3 tensor representing scalings.
                scaling_modifier (float): scaling modifier.
                rotation (torch.Tensor): Nx4 tensor representing quternions.

            Results:
            -----
                (torch.Tensor): Nx4x4 tensor representing covariance.
            """
            L = scaling_rotation(scaling * scaling_modifier, rotation)
            # May require stipping of symmetrics. Why?
            return L @ L.transpose(1, 2) 

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = scaling_rotation_to_covariance
        
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

