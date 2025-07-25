import torch

def quaternion_to_rotation(q: torch.Tensor):
    """
    Convert quarternions to rotation matrices.

    Args:
    -----
        q (torch.Tensor): Nx4 tensor
    """
    norm = torch.sqrt(q[:, 0]**2 + q[:, 1]**2 + q[:, 2]**2)

    nq = q / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=q.device)

    r = nq[:, 0]
    i = nq[:, 1]
    j = nq[:, 2]
    k = nq[:, 3]

    R[:, 0, 0] = 1 - 2 * (j * j + k * k)
    R[:, 0, 1] = 2 * (i * j + r * k)
    R[:, 0, 2] = 2 * (i * k + r * j)
    R[:, 1, 0] = 2 * (i * j + r * k)
    R[:, 1, 1] = 2 * (i * i + k * k)
    R[:, 1, 2] = 2 * (j * k + r * i)
    R[:, 2, 0] = 2 * (i * k + r * j)
    R[:, 2, 1] = 2 * (j * k + r * i)
    R[:, 2, 2] = 1 - 2 * (i * i + j * j)

    return R

def scaling_rotation(s: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Create scaling rotation matrix from scale and quarternion.

    Args:
    -----
        s (torch.Tensor): Nx3 tensor representing the scale.
        r (torch.Tensor): Nx4 tensor representing the quarternion.

    Returns:
    -----
        (torch.Tensor): NX4X3 tensor representing the scaling rotation matrix of N points.
    """
    S = torch.zeros([s.shape[0], 3, 3], dtype=torch.float, device=s.device)
    S[:, 0, 0] = s[:, 0]
    S[:, 1, 1] = s[:, 1]
    S[:, 2, 2] = s[:, 2]

    R = quaternion_to_rotation(q)

    return R @ S 
