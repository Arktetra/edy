from edy.modules.sparse.conv import SparseConv3d, SparseInverseConv3d
from edy.modules.sparse.linear import SparseLinear
from edy.modules.sparse.spatial import SparseDownSample, SparseUpSample
from edy.modules.sparse.tensor import SparseTensor

__all__ = [
    "SparseTensor",
    "SparseConv3d",
    "SparseInverseConv3d",
    "SparseLinear",
    "SparseDownSample",
    "SparseUpSample"
]

DEBUG = False
SPCONV_ALGO = 'auto'    # 'auto', 'implicit_gemm', 'native'

def __from_env():
    import os
        
    global SPCONV_ALGO
    env_spconv_algo = os.environ.get('SPCONV_ALGO')
    if env_spconv_algo is not None and env_spconv_algo in ['auto', 'implicit_gemm', 'native']:
        SPCONV_ALGO = env_spconv_algo
    print(f"[SPARSE][CONV] spconv algo: {SPCONV_ALGO}")
        

__from_env()

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug
