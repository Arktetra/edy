from typing import List, Optional, Tuple, Union
import torch


def check_tensor(
    tensor: torch.Tensor,
    shape: Optional[Union[torch.Size, List[int], Tuple[int]]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
):
    """
    check if the tensor satisfies the criterias.

    Args:
        tensor (torch.Tensor):
        shape (Optional[Union[torch.Size, List[int], Tuple[int]]]):
        dtype (Optional[torch.dtype]):
        device (Optional[str]):
        throw (bool):
    """
    if shape is not None:
        assert tensor.shape == shape, \
            f"tensor has shape {tensor.shape}, should be {shape}"
    if dtype is not None:
        assert tensor.dtype == dtype, \
            f"tensor has dtype {tensor.dtype}, should be {dtype}"
    if device is not None:
        assert tensor.device == device, \
            f"tensor has device {tensor.device}, should be {device}"

