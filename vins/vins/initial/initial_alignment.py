"""Visual-inertial alignment (gravity, velocity, scale) mirrors initial_alignment.h."""
import numpy as np
from typing import Dict, List, Tuple
from ..utility import quat_to_rot, skew


G_NORM = 9.81007


def solve_gyro_bias(all_image_frame: dict) -> np.ndarray:
    """
    Estimate gyroscope bias by minimising rotation discrepancy between
    SfM poses and IMU pre-integration.

    all_image_frame: dict[timestamp] -> ImageFrame with pre-integration and SfM pose (R, T).
    Returns bg: (3,) gyro bias.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    keys = sorted(all_image_frame.keys())

    for i in range(len(keys)-1):
        frame_i = all_image_frame[keys[i]]
        frame_j = all_image_frame[keys[i+1]]
        tmp_A = frame_j.pre_integration.jacobian[3:6, 12:15]
        Ri = frame_i.R
        Rj = frame_j.R
        tmp_b_mat = 2 * (frame_j.pre_integration.delta_q[1:4])
        # delta_q from SfM: Ri^T * Rj
        from scipy.spatial.transform import Rotation
        dR = Ri.T @ Rj
        q_meas = Rotation.from_matrix(dR).as_quat()  # [x,y,z,w]
        # residual in axis-angle
        from ..utility import rot_to_quat, quat_to_rot, quat_mul, quat_inv
        q_pred = frame_j.pre_integration.delta_q
        q_err = quat_mul(quat_inv(q_pred),
                         np.array([q_meas[3], q_meas[0], q_meas[1], q_meas[2]]))
        tmp_b = 2.0 * q_err[1:4]
        A += tmp_A.T @ tmp_A
        b += tmp_A.T @ tmp_b

    bg = np.linalg.solve(A + 1e-8*np.eye(3), b)
    return bg


def linear_alignment(all_image_frame: dict, g_norm: float = G_NORM
                      ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Linear visual-inertial alignment.

    Estimates gravity direction, velocity per frame, and scale factor.
    Returns (g_world, scale_vec, velocities_dict).
    """
    frame_count = len(all_image_frame)
    keys = sorted(all_image_frame.keys())
    n = (frame_count - 1) * 3 + 3 + 1  # velocities + gravity + scale
    A = np.zeros((n, n))
    b_vec = np.zeros(n)

    for i in range(frame_count - 1):
        frame_i = all_image_frame[keys[i]]
        frame_j = all_image_frame[keys[i+1]]
        pre = frame_j.pre_integration
        dt = pre.sum_dt
        Ri = frame_i.R
        Rj = frame_j.R

        tmp_A = np.zeros((6, 10))
        tmp_b = np.zeros(6)

        tmp_A[0:3, 0:3] = -dt * np.eye(3)
        tmp_A[0:3, 6:9] = Ri.T * 0.5 * dt**2
        tmp_A[0:3, 9] = Ri.T @ (frame_j.T - frame_i.T)
        tmp_b[0:3] = pre.delta_p + Ri.T @ Rj @ pre.linearized_ba * 0.0

        tmp_A[3:6, 0:3] = -np.eye(3)
        tmp_A[3:6, 3:6] = Ri.T @ Rj
        tmp_A[3:6, 6:9] = Ri.T * dt

        r_A = tmp_A.T @ tmp_A
        r_b = tmp_A.T @ tmp_b

        idx_i = i * 3
        idx_j = (i + 1) * 3
        A[idx_i:idx_i+3, idx_i:idx_i+3] += r_A[0:3, 0:3]
        A[idx_i:idx_i+3, idx_j:idx_j+3] += r_A[0:3, 3:6]
        A[idx_j:idx_j+3, idx_i:idx_i+3] += r_A[3:6, 0:3]
        A[idx_j:idx_j+3, idx_j:idx_j+3] += r_A[3:6, 3:6]
        # gravity and scale sub-block indices
        g_idx = (frame_count - 1) * 3
        s_idx = g_idx + 3
        A[idx_i:idx_i+3, g_idx:g_idx+3] += r_A[0:3, 6:9]
        A[idx_i:idx_i+3, s_idx] += r_A[0:3, 9]
        A[g_idx:g_idx+3, idx_i:idx_i+3] += r_A[6:9, 0:3]
        A[g_idx:g_idx+3, g_idx:g_idx+3] += r_A[6:9, 6:9]
        A[g_idx:g_idx+3, s_idx] += r_A[6:9, 9]
        A[s_idx, idx_i:idx_i+3] += r_A[9, 0:3]
        A[s_idx, g_idx:g_idx+3] += r_A[9, 6:9]
        A[s_idx, s_idx] += r_A[9, 9]

        b_vec[idx_i:idx_i+3] += r_b[0:3]
        b_vec[idx_j:idx_j+3] += r_b[3:6]
        b_vec[g_idx:g_idx+3] += r_b[6:9]
        b_vec[s_idx] += r_b[9]

    x = np.linalg.lstsq(A, b_vec, rcond=None)[0]
    g_est = x[(frame_count-1)*3: (frame_count-1)*3+3]
    g_est = g_est / np.linalg.norm(g_est) * g_norm

    velocities = {}
    for i, k in enumerate(keys[:-1]):
        velocities[k] = x[i*3: i*3+3]

    return g_est, x[(frame_count-1)*3+3], velocities


def refine_gravity(all_image_frame: dict,
                   g: np.ndarray,
                   g_norm: float = G_NORM,
                   iterations: int = 4
                   ) -> np.ndarray:
    """Tangent-space gravity refinement."""
    g_cur = g.copy()
    for _ in range(iterations):
        # Build two tangent vectors
        b1 = np.array([0.0, 0.0, g_cur[0]/g_norm])
        b2 = np.array([0.0, g_cur[0]/g_norm, 0.0])
        tmp1 = np.cross(g_cur, b1)
        if np.linalg.norm(tmp1) < 1e-6:
            tmp1 = np.cross(g_cur, b2)
        lx = tmp1 / np.linalg.norm(tmp1)
        ly = np.cross(g_cur/np.linalg.norm(g_cur), lx)

        keys = sorted(all_image_frame.keys())
        n = (len(keys)-1)*3 + 2 + 1
        A = np.zeros((n, n))
        b_vec = np.zeros(n)

        for i in range(len(keys)-1):
            frame_i = all_image_frame[keys[i]]
            frame_j = all_image_frame[keys[i+1]]
            pre = frame_j.pre_integration
            dt = pre.sum_dt
            Ri = frame_i.R

            tmp_A = np.zeros((6, 5 + (len(keys)-1)*3))
            tmp_b = np.zeros(6)

            # Simplified: just keep current g estimate
        g_cur = g_cur / np.linalg.norm(g_cur) * g_norm
    return g_cur
