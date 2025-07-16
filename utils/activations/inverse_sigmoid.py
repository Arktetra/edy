import torch

def inverse_sigmoid(x: torch.Tensor):
    return torch.log(x / (1 - x))
