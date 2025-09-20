import torch

from enum import Enum
from typing import Tuple, List

from edy.modules.sparse.tensor import SparseTensor

class SerializeMode(Enum):
    Z_ORDER = 0
    Z_ORDER_TRANSPOSED = 1
    HILBERT = 2
    HILBERT_TRANSPOSED = 3

SerializeModes = [
    SerializeMode.Z_ORDER,
    SerializeMode.Z_ORDER_TRANSPOSED,
    SerializeMode.HILBERT,
    SerializeMode.HILBERT_TRANSPOSED,
]

def calc_serialization(
    tensor: SparseTensor,
    window_size: int,
    serialize_mode: SerializeMode = SerializeMode.Z_ORDER,
    shift_sequence: int = 0,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    raise NotImplementedError("Implement Me!")

def sparse_serialized_scaled_dot_product_self_attention(
    qkv: SparseTensor,
        window_size: int,
        serialize_mode: SerializeMode = SerializeMode.Z_ORDER,
    shift_sequence: int = 0,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    raise NotImplementedError("Implement Me!")
