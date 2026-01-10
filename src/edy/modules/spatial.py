import torch

from jaxtyping import Float


def pixel_shuffle_3d(x: Float[torch.Tensor, "B C D H W"], scale_factor: int) -> Float[torch.Tensor, "B C H W D"]:
    """
    Perform 3D version of a pixel shuffle (subpixel upsampling) operation.

    Args:
    -----
        x (torch.Tensor): input tensor of shape (B, C, D, H, W), where
            B = batch size,
            C = number of channels,
            D, H, W = spatial dimensions
        scale_factor (int): factor by which each spatial dimension is increased.
    """
    B, C, H, W, D = x.shape
    C_ = C // scale_factor**3
    x = x.reshape(B, C_, scale_factor, scale_factor, scale_factor, H, W, D)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, C_, H * scale_factor, W * scale_factor, D * scale_factor)
    return x


def patchify(
    x: Float[torch.Tensor, "B C D H W"],
    patch_size: int,  # patch size
) -> Float[torch.Tensor, "B C*(p**3) D//p H//p W//p"]:
    """
    Split a tensor into non-overlapping patches, by subdividing each spatial dimension into blocks of size `p`, and then re-encoding those patches into the channel dimension.

    Args:
    -----
        x (torch.Tensor): input tensor of shape (B, C, D, H, W), where
            B = batch size,
            C = number of channels,
            D, H, W = spatial dimensions
        patch_size (int): a positive integer called patch size that exactly divides D, H, and W.

    Returns:
    -----
        torch.Tensor: a tensor of shape (B, C*(p**3), D//p, H//p, W//p)
    """
    DIM = x.dim() - 2
    for d in range(2, DIM + 2):
        assert x.shape[d] % patch_size == 0, (
            f"Dimension {d} of input tensor must be divisible by patch size, got {x.shape[d]} and {patch_size}"
        )

    # [B, C, D, H, W] -> [B, C, D//p, p, H//p, p, W//p, p]
    x = x.reshape(*x.shape[:2], *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, DIM + 2)], []))

    # [B, C, D//p, p, H//p, p, W//p, p] -> [B, C, p, p, p, D//p, H//p, W//p]
    x = x.permute(0, 1, *([2 * i + 3 for i in range(DIM)] + [2 * i + 2 for i in range(DIM)]))

    # [B, C, p, p, p, D//p, H//p, W//p] -> [B, C * (p ** 3), D//p, H//p, W//p]
    x = x.reshape(x.shape[0], x.shape[1] * (patch_size**DIM), *(x.shape[-DIM:]))

    return x


def unpatchify(
    x: Float[torch.Tensor, "B C D H W"],
    patch_size: int,
) -> Float[torch.Tensor, "B C//(p**3) D*p H*p W*p"]:
    """
    Reverse the patchification process by rearranging the patch dimensions back into spatial dimensions.

    Args:
    -----
        x (torch.Tensor): input tensor of shape (B, C, D, H, W), where
            B = batch size,
            C = number of channels (must be divisible by p^3),
            D, H, W = spatial dimensions
        patch_size (int): a positive integer called patch size.

    Returns:
    -----
        torch.Tensor: reconstructed tensor of shape (B, C//(p**3), D*p, H*p, W*p).
    """
    DIM = x.dim() - 2
    assert x.shape[1] % (patch_size**DIM) == 0, (
        f"Second dimension of input tensor must be divisible by patch size to unpatchify, got {x.shape[1]} and {patch_size**DIM}"
    )

    x = x.reshape(x.shape[0], x.shape[1] // (patch_size**DIM), *([patch_size] * DIM), *(x.shape[-DIM:]))
    x = x.permute(0, 1, *(sum([[2 + DIM + i, 2 + i] for i in range(DIM)], [])))
    x = x.reshape(x.shape[0], x.shape[1], *[x.shape[2 + 2 * i] * patch_size for i in range(DIM)])
    return x
