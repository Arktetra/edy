import math

def focal_to_fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def fov_to_focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
