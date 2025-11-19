import torch

from jaxtyping import Float

def pixel_shuffle_3d(
    x: Float[torch.Tensor, "B C D H W"],
    scale_factor: int
) -> Float[torch.Tensor, "B C H W D"]:
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
    raise NotImplementedError("Implement Me!")

def patchify(
    x: Float[torch.Tensor, "B C D H W"],
    p: int  # patch size
) -> Float[torch.Tensor, "B C*(p**3) D//p H//p W//p"]:
    """
    Split a tensor into non-overlapping patches, by subdividing each spatial dimension into blocks of size `p`, and then re-encoding those patches into the channel dimension.

    Args:
    -----
        x (torch.Tensor): input tensor of shape (B, C, D, H, W), where
            B = batch size,
            C = number of channels,
            D, H, W = spatial dimensions
        p (int): a positive integer called patch size that exactly divides D, H, and W.

    Returns:
    -----
        torch.Tensor: a tensor of shape (B, C*(p**3), D//p, H//p, W//p)
    """
    raise NotImplementedError("Implement Me!")

def unpatchify(
    x: Float[torch.Tensor, "B C D H W"],
    p: int,
) -> Float[torch.Tensor, "B C//(p**3) D*p H*p W*p"]:
    """
    Reverse the patchification process by rearranging the patch dimensions back into spatial dimensions.

    Args:
    -----
        x (torch.Tensor): input tensor of shape (B, C, D, H, W), where
            B = batch size,
            C = number of channels (must be divisible by p^3),
            D, H, W = spatial dimensions
        p (int): a positive integer called patch size.

    Returns:
    -----
        torch.Tensor: reconstructed tensor of shape (B, C//(p**3), D*p, H*p, W*p).
    """
    raise NotImplementedError("Implement Me!")
