from typing import TypedDict, List

class View(TypedDict):
    yaw: List[float]
    pitch: List[float]
    radius: List[float]
    fov: List[float]