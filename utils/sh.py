import torch

C0 = 0.28209479177387814

def rgb_to_sh(rgb: torch.Tensor):
    return (rgb - 0.5) / C0

def sh_to_rgb(sh: torch.Tensor):
    return sh * C0 + 0.5
