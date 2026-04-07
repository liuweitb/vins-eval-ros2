"""KLT optical-flow feature tracker (mirrors feature_tracker.cpp)."""
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class FeatureTracker:
    def __init__(self, max_cnt: int = 150, min_dist: int = 30,
                 f_threshold: float = 1.0, flow_back: bool = True,
                 equalize: bool = True):
        self.max_cnt = max_cnt
        self.min_dist = min_dist
        self.f_threshold = f_threshold
        self.flow_back = flow_back
        self.equalize = equalize

        self._id_counter = 0
        self.prev_img: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None   # (N, 2)
        self.prev_ids: List[int] = []
        self.prev_track_cnt: List[int] = []

        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def _new_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _equalize_image(self, img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _detect_new_features(self, img: np.ndarray,
                              existing: np.ndarray | None) -> np.ndarray:
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        if existing is not None and len(existing) > 0:
            for pt in existing.astype(np.int32):
                cv2.circle(mask, tuple(pt), self.min_dist, 0, -1)
        n_new = self.max_cnt - (len(existing) if existing is not None else 0)
        if n_new <= 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = cv2.goodFeaturesToTrack(
            img, n_new, qualityLevel=0.01, minDistance=self.min_dist, mask=mask
        )
        return pts.reshape(-1, 2) if pts is not None else np.empty((0, 2), dtype=np.float32)

    def _reject_with_F(self, pts0: np.ndarray, pts1: np.ndarray,
                       status: np.ndarray) -> np.ndarray:
        if np.sum(status) < 8:
            return status
        idx = np.where(status)[0]
        p0 = pts0[idx]
        p1 = pts1[idx]
        _, mask = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC,
                                         self.f_threshold, 0.99)
        if mask is None:
            return status
        new_status = status.copy()
        for k, i in enumerate(idx):
            if not mask[k]:
                new_status[i] = 0
        return new_status

    def track_image(self, img: np.ndarray, t: float,
                    img_right: np.ndarray | None = None
                    ) -> Dict[int, List[np.ndarray]]:
        """
        Track features and return observations.

        Returns dict: feature_id -> list of [x, y, 1, u, v, 1, vx, vy] per camera.
        (x,y) = normalised camera coords; (u,v) = pixel coords; (vx,vy) = velocity.
        """
        if self.equalize:
            img = self._equalize_image(img)

        cur_pts: np.ndarray
        cur_ids: List[int]
        cur_cnt: List[int]
        status: np.ndarray

        if self.prev_img is not None and self.prev_pts is not None and len(self.prev_pts) > 0:
            pts_prev = self.prev_pts.reshape(-1, 1, 2).astype(np.float32)
            pts_cur, st, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_img, img, pts_prev, None, **self._lk_params)
            st = st.ravel()

            if self.flow_back:
                pts_back, st_back, _ = cv2.calcOpticalFlowPyrLK(
                    img, self.prev_img, pts_cur, None, **self._lk_params)
                diff = np.linalg.norm(pts_prev.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1)
                st[diff > 0.5] = 0

            st = self._reject_with_F(
                self.prev_pts, pts_cur.reshape(-1, 2), st)

            cur_pts = pts_cur.reshape(-1, 2)
            cur_ids = [self.prev_ids[i] for i in range(len(st)) if st[i]]
            cur_cnt = [self.prev_track_cnt[i]+1 for i in range(len(st)) if st[i]]
            cur_pts = cur_pts[st.astype(bool)]
        else:
            cur_pts = np.empty((0, 2), dtype=np.float32)
            cur_ids = []
            cur_cnt = []

        new_pts = self._detect_new_features(img, cur_pts if len(cur_pts) > 0 else None)
        if len(new_pts) > 0:
            new_ids = [self._new_id() for _ in range(len(new_pts))]
            new_cnt = [1] * len(new_pts)
            cur_pts = np.vstack([cur_pts, new_pts]) if len(cur_pts) > 0 else new_pts
            cur_ids = cur_ids + new_ids
            cur_cnt = cur_cnt + new_cnt

        # Pixel velocity
        dt = 1.0 / 30.0
        velocity = np.zeros((len(cur_pts), 2))
        if self.prev_pts is not None and len(cur_pts) > 0:
            id_to_prev = {pid: pp for pid, pp in zip(self.prev_ids, self.prev_pts)}
            for k, fid in enumerate(cur_ids):
                if fid in id_to_prev:
                    velocity[k] = (cur_pts[k] - id_to_prev[fid]) / dt

        self.prev_img = img
        self.prev_pts = cur_pts
        self.prev_ids = cur_ids
        self.prev_track_cnt = cur_cnt

        return self._build_observation(cur_pts, cur_ids, velocity, img_right)

    def _build_observation(self, pts: np.ndarray, ids: List[int],
                           velocity: np.ndarray,
                           img_right: np.ndarray | None
                           ) -> Dict[int, List[np.ndarray]]:
        """Convert pixel observations to normalised camera coordinates."""
        result: Dict[int, List[np.ndarray]] = {}
        for k, fid in enumerate(ids):
            u, v = float(pts[k, 0]), float(pts[k, 1])
            vx, vy = float(velocity[k, 0]), float(velocity[k, 1])
            # Observation vector matches C++ format: [x, y, z=1, u, v, 1, vx, vy]
            obs = np.array([u, v, 1.0, u, v, 1.0, vx, vy])
            result[fid] = [obs]
        return result

    def set_camera(self, camera) -> None:
        """Inject camera model for undistortion if needed."""
        self._camera = camera
