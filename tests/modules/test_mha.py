import torch

from edy.modules.attention.mha import MultiHeadAttention


class TestMultiHeadAttention:
    def test_self_attention_shape(self):
        B, L, C, H = 2, 8, 16, 4
        x = torch.randn(B, L, C)

        mha = MultiHeadAttention(channels=C, num_heads=H, type="self", use_rope=True, qk_rms_norm=True)

        # Forward pass
        out = mha(x)
        print("Self-attention output shape:", out.shape)
        # Check output shape
        assert out.shape == torch.Size([B, L, C])
        # Optional: check dtype and finiteness
        assert torch.isfinite(out).all()
        assert out.dtype == x.dtype

    def test_cross_attention_shape(self):
        B, L, C, H = 2, 8, 16, 4
        ctx_L = 6
        x = torch.randn(B, L, C)
        context = torch.randn(B, ctx_L, C)

        mha_cross = MultiHeadAttention(channels=C, num_heads=H, type="cross", use_rope=True, qk_rms_norm=True)
        out_cross = mha_cross(x, context=context)
        assert out_cross.shape == torch.Size([B, L, C])
        print(out_cross.shape)

    def test_global_attention_shape(self):
        B, L, C, H = 2, 8, 16, 4
        x = torch.randn(B, L, C)

        mha_global = MultiHeadAttention(channels=C, num_heads=H, type="global", qk_rms_norm=True)
        out_global = mha_global(x)
        assert out_global.shape == torch.Size([B, L, C])
        print(out_global.shape)
