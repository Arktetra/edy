from typing import Optional
import math
import torch
import torch.nn as nn

from flex_gemm.ops.spconv import sparse_submanifold_conv3d

from edy.modules.sparse.tensor import SparseTensor


class SparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        indice_key: str = None,
    ):
        super(SparseConv3d, self).__init__()
        self.sparse_conv3d_init(in_channels, out_channels, kernel_size, stride, dilation, padding, bias)

    def sparse_conv3d_init(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
    ):
        assert stride == 1 and (padding is None), (
            "Currently flex_gemm implementation only support submanifold sparse convolutoin (stride=1, padding=None)"
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
        self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
        self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

        self.weight = nn.Parameter(self.weight.permute(0, 2, 3, 4, 1).contiguous())

    def forward(self, x: SparseTensor) -> SparseTensor:
        Co, Kd, Kh, Kw, Ci = self.weight.shape
        neighbor_cache_key = f"SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}"
        neighbor_cache = x.get_spatial_cache(neighbor_cache_key)

        # It seems that during the computation of out-block, the neighbor-cache with only 13072
        # entries is used to compute the convolution for input with 47532 entries, which clearly
        # requires a recomputation of neighbor-cache.

        out, neighbor_cache_ = sparse_submanifold_conv3d(
            x.feats,
            x.coords,
            torch.Size([*x.shape, *x.spatial_shape]),
            self.weight,
            self.bias,
            # neighbor_cache,       # there is bug due to passing a neighbor cache here look line 67.
            dilation=self.dilation,
        )

        if neighbor_cache is None:
            x.register_spatial_cache(neighbor_cache_key, neighbor_cache_)

        out = x.replace(out)
        return out


# class SparseConv3d(nn.Module):
#     def __init__(
#         self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None
#     ):
#         super(SparseConv3d, self).__init__()

#         algo = None

#         if SPCONV_ALGO == "native":
#             algo = spconv.ConvAlgo.Native
#         elif SPCONV_ALGO == "implicit_gemm":
#             algo = spconv.ConvAlgo.MaskImplicitGemm
#         if stride == 1 and (padding is None):
#             self.conv = spconv.SubMConv3d(
#                 in_channels, out_channels, kernel_size, dilation=dilation, bias=bias, indice_key=indice_key, algo=algo
#             )
#         else:
#             self.conv = spconv.SparseConv3d(
#                 in_channels,
#                 out_channels,
#                 kernel_size,
#                 stride=stride,
#                 dilation=dilation,
#                 padding=padding,
#                 bias=bias,
#                 indice_key=indice_key,
#                 algo=algo,
#             )

#         self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)
#         self.padding = padding

#     def forward(self, x: SparseTensor) -> SparseTensor:
#         spatial_changed = any(s != 1 for s in self.stride) or (self.padding is not None)
#         new_data = self.conv(x.data)
#         new_shape = [x.shape[0], self.conv.out_channels]
#         new_layout = None if spatial_changed else x.layout

#         if spatial_changed and (x.shape[0] != 1):
#             fwd = new_data.indices[:, 0].argsort()
#             bwd = torch.zeros_like(fwd).scatter_(0, fwd, torch.arange(fwd.shape[0], device=fwd.device))
#             sorted_feats = new_data.features[fwd]
#             sorted_coords = new_data.indices[fwd]
#             unsorted_data = new_data
#             new_data = spconv.SparseConvTensor(
#                 sorted_feats, sorted_coords, unsorted_data.spatial_shape, unsorted_data.batch_size
#             )  # type: ignore

#         out = SparseTensor(
#             new_data,
#             shape=torch.Size(new_shape),
#             layout=new_layout,
#             scale=tuple([s * stride for s, stride in zip(x._scale, self.stride)]),
#             spatial_cache=x._spatial_cache,
#         )

#         if spatial_changed and (x.shape[0] != 1):
#             out.register_spatial_cache(f"conv_{self.stride}_unsorted_data", unsorted_data)
#             out.register_spatial_cache(f"conv_{self.stride}_sort_bwd", bwd)

#         return out


# class SparseInverseConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
#         super(SparseInverseConv3d, self).__init__()

#         self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, bias=bias, indice_key=indice_key)

#         self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride, stride, stride)

#     def forward(self, x: SparseTensor) -> SparseTensor:
#         spatial_changed = any(s != 1 for s in self.stride)

#         if spatial_changed:
#             data = x.get_spatial_cache(f"conv_{self.stride}_unsorted_data")
#             bwd = x.get_spatial_cache(f"conv_{self.stride}_sort_bwd")
#             data = data.replace_feature(x.feats[bwd])
#         else:
#             data = x.data

#         new_data = self.conv(data)
#         new_shape = [x.shape[0], self.conv.out_channels]
#         new_layout = None if spatial_changed else x.layout
#         out = SparseTensor(
#             new_data,
#             shape=torch.Size(new_shape),
#             layout=new_layout,
#             scale=tuple([s // stride for s, stride in zip(x._scale, self.stride)]),
#             spatial_cache=x._spatial_cache,
#         )

#         return out
