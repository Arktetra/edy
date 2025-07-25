from scipy.spatial import KDTree
import torch

def dist_square_nearest_k(points: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    Finds the k-nearest neighbors of each point. Then, returns the mean of the square of the distance between the found neighbor points for each point.

    Args:
    -----
        points (torch.Tensor): [Nx3] tensor representing N 3d points.
        k (int): number of neighbors to consider.

    Returns:
    -----
        (torch.Tensor): [N] tensor representing the mean of the square of the distance between a point and its neighbors.
    """

    points_np = points.detach().cpu().float().numpy()
    dists, _ = KDTree(points_np).query(points_np, k=k+1)
    mean_dists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(mean_dists, dtype=points.dtype, device=points.device)


