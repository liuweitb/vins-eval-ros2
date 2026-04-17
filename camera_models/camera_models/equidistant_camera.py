import numpy as np
import cv2
from .camera_base import CameraBase
from .yaml_utils import load_camera_yaml


class EquidistantCamera(CameraBase):
    """Kannala-Brandt fisheye / equidistant camera model."""

    def __init__(self, width: int, height: int,
                 fx: float, fy: float, cx: float, cy: float,
                 k1: float = 0.0, k2: float = 0.0,
                 k3: float = 0.0, k4: float = 0.0):
        super().__init__(width, height)
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self._dist = np.array([k1, k2, k3, k4], dtype=np.float64)

    @classmethod
    def from_yaml(cls, path: str) -> "EquidistantCamera":
        data = load_camera_yaml(path)
        ip = data.get("intrinsic_parameters", data.get("projection_parameters", {}))
        dp = data.get("distortion_parameters", {})
        return cls(
            width=data["image_width"],
            height=data["image_height"],
            fx=ip["fx"], fy=ip["fy"], cx=ip["cx"], cy=ip["cy"],
            k1=dp.get("k1", 0.0), k2=dp.get("k2", 0.0),
            k3=dp.get("k3", 0.0), k4=dp.get("k4", 0.0),
        )

    def lift_projective(self, p: np.ndarray) -> np.ndarray:
        mx = (p[0] - self.cx) / self.fx
        my = (p[1] - self.cy) / self.fy
        # Iterative undistortion
        r = np.sqrt(mx**2 + my**2)
        theta = r
        for _ in range(10):
            th2 = theta**2
            th4 = th2**2
            th6 = th2**3
            th8 = th4**2
            f = theta*(1 + self.k1*th2 + self.k2*th4 + self.k3*th6 + self.k4*th8) - r
            fp = 1 + 3*self.k1*th2 + 5*self.k2*th4 + 7*self.k3*th6 + 9*self.k4*th8
            theta -= f / fp
        z = np.cos(theta)
        s = np.sin(theta)
        if r < 1e-10:
            return np.array([0.0, 0.0, 1.0])
        return np.array([s*mx/r, s*my/r, z])

    def space_to_plane(self, P: np.ndarray) -> np.ndarray:
        r = np.sqrt(P[0]**2 + P[1]**2)
        theta = np.arctan2(r, P[2])
        th2 = theta**2
        th4 = th2**2
        th6 = th2**3
        th8 = th4**2
        rho = theta*(1 + self.k1*th2 + self.k2*th4 + self.k3*th6 + self.k4*th8)
        if r < 1e-10:
            return np.array([self.cx, self.cy])
        return np.array([
            self.fx * rho * P[0] / r + self.cx,
            self.fy * rho * P[1] / r + self.cy,
        ])

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.fisheye.undistortImage(img, self._K, self._dist, Knew=self._K)
