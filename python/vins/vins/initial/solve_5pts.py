"""5-point essential matrix solver wrapping OpenCV."""
import cv2
import numpy as np
from typing import Tuple


def motion_from_essential(E: np.ndarray,
                           pts0: np.ndarray,
                           pts1: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover R, t from essential matrix with cheirality check.

    pts0, pts1: (N,2) normalised image coordinates.
    Returns (R, t) where t is a unit vector.
    """
    R1, R2, t = cv2.decomposeEssentialMat(E)
    candidates = [
        (R1, t), (R1, -t), (R2, t), (R2, -t)
    ]
    best = None
    best_count = -1
    for R, tv in candidates:
        P0 = np.eye(3, 4)
        P1 = np.hstack([R, tv.reshape(3, 1)])
        pts4d = cv2.triangulatePoints(P0, P1,
                                      pts0.T.reshape(2, -1).astype(np.float64),
                                      pts1.T.reshape(2, -1).astype(np.float64))
        pts3d = (pts4d[:3] / pts4d[3]).T   # (N, 3)
        in_front_i = pts3d[:, 2] > 0
        pts3d_j = (R @ pts3d.T + tv.reshape(3, 1)).T
        in_front_j = pts3d_j[:, 2] > 0
        count = int(np.sum(in_front_i & in_front_j))
        if count > best_count:
            best_count = count
            best = (R, tv)
    assert best is not None
    return best


def solve_relative_pose(pts0: np.ndarray,
                        pts1: np.ndarray,
                        threshold: float = 1.0
                        ) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Estimate relative pose from normalised point correspondences.

    Returns (success, R, t).
    """
    if len(pts0) < 8:
        return False, np.eye(3), np.zeros(3)
    E, mask = cv2.findEssentialMat(
        pts0.astype(np.float64),
        pts1.astype(np.float64),
        focal=1.0, pp=(0.0, 0.0),
        method=cv2.RANSAC, prob=0.999, threshold=threshold,
    )
    if E is None or mask is None:
        return False, np.eye(3), np.zeros(3)
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 8:
        return False, np.eye(3), np.zeros(3)
    R, t = motion_from_essential(E, pts0[inliers], pts1[inliers])
    return True, R, t.ravel()
