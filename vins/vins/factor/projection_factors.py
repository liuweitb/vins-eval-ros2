"""Visual reprojection residuals mirroring the C++ projection factor headers."""
import numpy as np
from ..utility import quat_to_rot, quat_mul, quat_inv


FOCAL_LENGTH = 460.0  # normalised-coordinates focal length used in C++ code


def project_mono(pts_i: np.ndarray, pts_j: np.ndarray,
                 Pi: np.ndarray, Qi: np.ndarray,
                 Pj: np.ndarray, Qj: np.ndarray,
                 tic: np.ndarray, qic: np.ndarray,
                 inv_depth: float) -> np.ndarray:
    """Two-frame, one-camera projection residual (2-vector)."""
    Ri = quat_to_rot(Qi)
    Rj = quat_to_rot(Qj)
    Ric = quat_to_rot(qic)

    pts_camera_i = pts_i / inv_depth
    pts_imu_i = Ric @ pts_camera_i + tic
    pts_w = Ri @ pts_imu_i + Pi
    pts_imu_j = Rj.T @ (pts_w - Pj)
    pts_camera_j = Ric.T @ (pts_imu_j - tic)

    dep_j = pts_camera_j[2]
    res = FOCAL_LENGTH * (pts_camera_j[:2] / dep_j - pts_j[:2])
    return res


def project_stereo_two_frame(pts_i: np.ndarray, pts_j: np.ndarray,
                              Pi: np.ndarray, Qi: np.ndarray,
                              Pj: np.ndarray, Qj: np.ndarray,
                              tic0: np.ndarray, qic0: np.ndarray,
                              tic1: np.ndarray, qic1: np.ndarray,
                              inv_depth: float) -> np.ndarray:
    """Two-frame, two-camera (right camera in frame j) projection residual (2-vector)."""
    Ri = quat_to_rot(Qi)
    Rj = quat_to_rot(Qj)
    Ric0 = quat_to_rot(qic0)
    Ric1 = quat_to_rot(qic1)

    pts_camera_i = pts_i / inv_depth
    pts_imu_i = Ric0 @ pts_camera_i + tic0
    pts_w = Ri @ pts_imu_i + Pi
    pts_imu_j = Rj.T @ (pts_w - Pj)
    pts_camera_j = Ric1.T @ (pts_imu_j - tic1)

    dep_j = pts_camera_j[2]
    res = FOCAL_LENGTH * (pts_camera_j[:2] / dep_j - pts_j[:2])
    return res


def project_stereo_one_frame(pts_i: np.ndarray, pts_j: np.ndarray,
                              tic0: np.ndarray, qic0: np.ndarray,
                              tic1: np.ndarray, qic1: np.ndarray,
                              inv_depth: float) -> np.ndarray:
    """Single-frame stereo projection residual (2-vector)."""
    Ric0 = quat_to_rot(qic0)
    Ric1 = quat_to_rot(qic1)

    pts_camera_0 = pts_i / inv_depth
    pts_w = Ric0 @ pts_camera_0 + tic0
    pts_camera_1 = Ric1.T @ (pts_w - tic1)

    dep = pts_camera_1[2]
    res = FOCAL_LENGTH * (pts_camera_1[:2] / dep - pts_j[:2])
    return res
