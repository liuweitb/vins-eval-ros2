from .camera_base import CameraBase
from .pinhole_camera import PinholeCamera
from .mei_camera import MeiCamera
from .equidistant_camera import EquidistantCamera
from .yaml_utils import load_camera_yaml


class CameraFactory:
    @staticmethod
    def generate_camera_from_yaml(path: str) -> CameraBase:
        data = load_camera_yaml(path)
        model = data.get("model_type", "PINHOLE").upper()
        if model in ("PINHOLE", "PINHOLE_FULL"):
            return PinholeCamera.from_yaml(path)
        elif model == "MEI":
            return MeiCamera.from_yaml(path)
        elif model in ("EQUIDISTANT", "KANNALA_BRANDT"):
            return EquidistantCamera.from_yaml(path)
        else:
            raise ValueError(f"Unknown camera model: {model}")
