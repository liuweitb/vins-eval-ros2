from abc import ABC, abstractmethod
import numpy as np


class CameraBase(ABC):
    def __init__(self, width: int, height: int):
        self.image_width = width
        self.image_height = height

    @abstractmethod
    def lift_projective(self, p: np.ndarray) -> np.ndarray:
        """Lift 2D image point to unit 3D bearing vector."""

    @abstractmethod
    def space_to_plane(self, P: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D image coordinates."""

    @abstractmethod
    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        """Return undistorted image."""

    def estimate_intrinsics(self, board_size, image_points, object_points):
        raise NotImplementedError
