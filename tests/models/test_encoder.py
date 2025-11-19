import torch

from edy.models.sparse_structure_flow import BatchEncoder

class TestShapeEncoder:
    def test_batch_encoder(self):
        batch_encoder = BatchEncoder(
            max_batch_size=32,
            embed_dim=1024,
            init_scale=5.0
        )
        batch_indices = torch.randint(low=0, high=7, size=(8,))
        batch_encodings = batch_encoder(batch_indices)
        assert batch_encodings.shape == torch.Size((8, 1, 1024))

