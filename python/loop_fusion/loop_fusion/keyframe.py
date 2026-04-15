"""Keyframe with BRIEF descriptors for loop-closure detection (mirrors keyframe.h)."""
import cv2
import numpy as np
from typing import List, Optional, Tuple


class BriefExtractor:
    """Wrapper around OpenCV's ORB (provides BRIEF-like descriptors)."""

    def __init__(self, n_features: int = 500):
        self._detector = cv2.ORB_create(nfeatures=n_features, fastThreshold=10)

    def compute(self, img: np.ndarray,
                keypoints: Optional[List[cv2.KeyPoint]] = None
                ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        if keypoints is None:
            kps = self._detector.detect(img)
        else:
            kps = keypoints
        kps, descs = self._detector.compute(img, kps)
        return kps, descs


class KeyFrame:
    _frame_id_counter = 0

    def __init__(self,
                 t: float,
                 index: int,
                 P: np.ndarray,
                 R: np.ndarray,
                 img: np.ndarray,
                 pts_3d: np.ndarray,   # (N, 3) world-frame landmarks
                 pts_2d: np.ndarray,   # (N, 2) pixel observations
                 sequence: int = 0):
        KeyFrame._frame_id_counter += 1
        self.frame_id = KeyFrame._frame_id_counter
        self.t = t
        self.index = index
        self.T_w_i = P.copy()    # position in world
        self.R_w_i = R.copy()    # rotation body->world
        self.img = img.copy()
        self.pts_3d = pts_3d.copy()
        self.pts_2d = pts_2d.copy()
        self.sequence = sequence

        self.keypoints: List[cv2.KeyPoint] = []
        self.descriptors: Optional[np.ndarray] = None

        self.has_loop = False
        self.loop_index = -1
        self.relative_t = np.zeros(3)
        self.relative_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.relative_yaw = 0.0

        extractor = BriefExtractor()
        self.keypoints, self.descriptors = extractor.compute(img)

    def find_connection(self, other: "KeyFrame",
                        min_inliers: int = 15
                        ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Try to find a geometric loop connection between this and another keyframe.

        Returns (success, relative_t, relative_R).
        """
        if self.descriptors is None or other.descriptors is None:
            return False, np.zeros(3), np.eye(3)

        # Descriptor matching
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(self.descriptors, other.descriptors)
        if len(matches) < min_inliers:
            return False, np.zeros(3), np.eye(3)

        pts_self = np.array([self.keypoints[m.queryIdx].pt for m in matches],
                             dtype=np.float64)
        pts_other = np.array([other.keypoints[m.trainIdx].pt for m in matches],
                              dtype=np.float64)

        # F-matrix RANSAC
        _, mask = cv2.findFundamentalMat(pts_self, pts_other,
                                          cv2.FM_RANSAC, 3.0, 0.99)
        if mask is None or mask.sum() < min_inliers:
            return False, np.zeros(3), np.eye(3)

        inlier_mask = mask.ravel().astype(bool)
        if inlier_mask.sum() < min_inliers:
            return False, np.zeros(3), np.eye(3)

        # PnP: use 3D points from this frame matched to 2D in other
        obj_pts = []
        img_pts = []
        for i, m in enumerate(matches):
            if not inlier_mask[i]:
                continue
            kp_idx = m.queryIdx
            if kp_idx < len(self.pts_3d):
                obj_pts.append(self.pts_3d[kp_idx])
                img_pts.append(pts_other[i])

        if len(obj_pts) < 6:
            return False, np.zeros(3), np.eye(3)

        obj_pts = np.array(obj_pts, dtype=np.float64)
        img_pts = np.array(img_pts, dtype=np.float64)
        K = np.eye(3)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, K, None)
        if not ok or inliers is None or len(inliers) < 6:
            return False, np.zeros(3), np.eye(3)

        R_mat, _ = cv2.Rodrigues(rvec)
        return True, tvec.ravel(), R_mat
