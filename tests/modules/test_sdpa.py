import torch

from edy.modules.attention.full_attn import scaled_dot_product_attention


class TestFullAttention:
    def test_full_attention_shapes(self):
        N = 2  # batch size
        L = 8  # sequence length
        d = 32  # model dimension
        h = 4  # number of heads
        d_head = d // h

        # Create test q, k, v tensors
        q = torch.randn(N, L, h, d_head)
        k = torch.randn(N, L, h, d_head)
        v = torch.randn(N, L, h, d_head)

        output = scaled_dot_product_attention(q, k, v)

        # Assertions
        assert output.shape == (N, L, h, d_head)
