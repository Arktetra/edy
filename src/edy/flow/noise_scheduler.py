from abc import ABC, abstractmethod

import torch


class Alpha(ABC):
    def __init__(self):
        assert torch.allclose(
            self(torch.zeros(1, 1, 1, 1, 1)), torch.zeros(1, 1, 1, 1, 1)
        )
        assert torch.allclose(
            self(torch.ones(1, 1, 1, 1, 1)), torch.ones(1, 1, 1, 1, 1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def dt(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1)

class Beta(ABC):
    def __init__(self):
        assert torch.allclose(
            self(torch.zeros(1, 1, 1, 1, 1)), torch.ones(1, 1, 1, 1, 1)
        )
        assert torch.allclose(
            self(torch.ones(1, 1, 1, 1, 1)), torch.zeros(1, 1, 1, 1, 1)
        )   

        @abstractmethod
        def __call__(self, t: torch.Tensor) -> torch.Tensor:
            pass

        @abstractmethod
        def dt(self, t: torch.Tensor) -> torch.Tensor:
            t = t.unsqueeze(1)
            dt = vmap(jacrev(self))(t)
            return dt.view(-1, 1, 1, 1)
