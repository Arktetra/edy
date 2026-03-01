import torch

from torch.nn.attention.varlen import varlen_attn


def flash_attn_varlen_func(q, k, v, cu_seq_q, cu_seq_k, max_q, max_k, causal=None):
    head_size_og = q.size(2)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    out_padded = varlen_attn(q, k, v, cu_seq_q, cu_seq_k, max_q, max_k, is_causal=causal)
    return out_padded[..., :head_size_og]


def flash_attn_varlen_kvpacked_func(q, kv, cu_seq_q, cu_seq_k, max_q, max_k, causal=None):
    k, v = kv[:, 0].detach(), kv[:, 1].detach()
    head_size_og = q.size(2)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    out_padded = varlen_attn(q, k, v, cu_seq_q, cu_seq_k, max_q, max_k, is_causal=causal)
    return out_padded[..., :head_size_og]


def flash_attn_varlen_qkvpacked_func(qkv, cu_seq, max_len, causal=None):
    q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
    head_size_og = q.size(2)
    if head_size_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
        v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
    out_padded = varlen_attn(q, k, v, cu_seq, cu_seq, max_len, max_len, is_causal=causal)
    return out_padded[..., :head_size_og]
