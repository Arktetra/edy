from edy.modules.sparse.conv.conv import SparseConv3d
from edy.modules.sparse.linear import SparseLinear
from edy.modules.sparse.spatial import SparseDownSample, SparseUpSample, SparseSubdivide
from edy.modules.sparse.tensor import SparseTensor
from edy.modules.sparse.norm import SparseGroupNorm32
from edy.modules.sparse.activations import SparseSiLU

__all__ = [
    "SparseTensor",
    "SparseConv3d",
    "SparseLinear",
    "SparseDownSample",
    "SparseUpSample",
    "SparseGroupNorm32",
    "SparseSiLU",
    "SparseSubdivide",
]

DEBUG = False  # 'auto', 'implicit_gemm', 'native'

# def __from_env():
#     import os

#     global SPCONV_ALGO
#     env_spconv_algo = os.environ.get('SPCONV_ALGO')
#     if env_spconv_algo is not None and env_spconv_algo in ['auto', 'implicit_gemm', 'native']:
#         SPCONV_ALGO = env_spconv_algo
#     print(f"[SPARSE][CONV] spconv algo: {SPCONV_ALGO}")


# __from_env()


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
