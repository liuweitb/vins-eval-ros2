"""Global SfM initialisation (mirrors initial_sfm.h/cpp)."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional
from ..utility import quat_to_rot, rot_to_quat, normalize_quat


class SFMFeature:
    def __init__(self):
        self.state = False
        self.id: int = -1
        self.observation: List[Tuple[int, np.ndarray]] = []  # (frame_idx, normalised_pt)
        self.position = np.zeros(3)


class GlobalSFM:
    def construct(self,
                  frame_num: int,
                  q: List[np.ndarray],    # [w,x,y,z] per frame (output)
                  T: List[np.ndarray],    # 3-vec per frame (output)
                  l: int,                 # pivot frame index
                  relative_R: np.ndarray,
                  relative_T: np.ndarray,
                  sfm_f: List[SFMFeature]
                  ) -> bool:
        """
        Run incremental SfM to recover camera poses and landmark positions.

        Frames are indexed 0..frame_num-1.  Frame l and frame frame_num-1 are
        the initial pair whose relative pose is given.
        """
        feature_num = len(sfm_f)

        # Initialise pivot and last frame
        q_arr = [None] * frame_num
        T_arr = [None] * frame_num

        q_arr[l] = np.array([1.0, 0.0, 0.0, 0.0])
        T_arr[l] = np.zeros(3)

        q_arr[frame_num - 1] = normalize_quat(rot_to_quat(relative_R))
        T_arr[frame_num - 1] = relative_T.copy()

        Rl = np.eye(3)
        Tl = np.zeros(3)
        R_initial = relative_R.copy()
        P_initial = relative_T.copy()

        # Triangulate features between l and last frame
        self._triangulate_two_frames(l, Rl, Tl, frame_num-1, R_initial, P_initial, sfm_f)

        # Forward from l+1 to frame_num-2
        for i in range(l+1, frame_num-1):
            R_ini = quat_to_rot(q_arr[i-1]) if q_arr[i-1] is not None else np.eye(3)
            P_ini = T_arr[i-1] if T_arr[i-1] is not None else np.zeros(3)
            ok, R, P = self._solve_pnp(i, sfm_f, R_ini, P_ini)
            if not ok:
                return False
            q_arr[i] = normalize_quat(rot_to_quat(R))
            T_arr[i] = P
            self._triangulate(l, Rl, Tl, i, R, P, sfm_f)
            self._triangulate(frame_num-1, R_initial, P_initial, i, R, P, sfm_f)

        # Backward from l-1 down to 0
        for i in range(l-1, -1, -1):
            R_ini = quat_to_rot(q_arr[i+1]) if q_arr[i+1] is not None else np.eye(3)
            P_ini = T_arr[i+1] if T_arr[i+1] is not None else np.zeros(3)
            ok, R, P = self._solve_pnp(i, sfm_f, R_ini, P_ini)
            if not ok:
                return False
            q_arr[i] = normalize_quat(rot_to_quat(R))
            T_arr[i] = P
            self._triangulate(l, Rl, Tl, i, R, P, sfm_f)

        # Fill remaining triangulations
        for i in range(frame_num-1):
            for j in range(i+1, frame_num):
                if q_arr[i] is not None and q_arr[j] is not None:
                    self._triangulate(i, quat_to_rot(q_arr[i]), T_arr[i],
                                      j, quat_to_rot(q_arr[j]), T_arr[j], sfm_f)

        for i in range(frame_num):
            q[i] = q_arr[i]
            T[i] = T_arr[i]
        return True

    def _triangulate_two_frames(self, i: int, Ri: np.ndarray, Ti: np.ndarray,
                                  j: int, Rj: np.ndarray, Tj: np.ndarray,
                                  sfm_f: List[SFMFeature]) -> None:
        for feat in sfm_f:
            if feat.state:
                continue
            obs_i = obs_j = None
            for fi, pt in feat.observation:
                if fi == i:
                    obs_i = pt
                if fi == j:
                    obs_j = pt
            if obs_i is not None and obs_j is not None:
                p3d = self._triangulate_point(Ri, Ti, Rj, Tj, obs_i, obs_j)
                if p3d is not None:
                    feat.position = p3d
                    feat.state = True

    def _triangulate(self, i: int, Ri: np.ndarray, Ti: np.ndarray,
                     j: int, Rj: np.ndarray, Tj: np.ndarray,
                     sfm_f: List[SFMFeature]) -> None:
        self._triangulate_two_frames(i, Ri, Ti, j, Rj, Tj, sfm_f)

    @staticmethod
    def _triangulate_point(R0: np.ndarray, t0: np.ndarray,
                            R1: np.ndarray, t1: np.ndarray,
                            pt0: np.ndarray, pt1: np.ndarray
                            ) -> Optional[np.ndarray]:
        P0 = np.hstack([R0, t0.reshape(3, 1)])
        P1 = np.hstack([R1, t1.reshape(3, 1)])
        A = np.array([
            pt0[0]*P0[2] - P0[0],
            pt0[1]*P0[2] - P0[1],
            pt1[0]*P1[2] - P1[0],
            pt1[1]*P1[2] - P1[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-10:
            return None
        return (X[:3] / X[3])

    def _solve_pnp(self, frame: int, sfm_f: List[SFMFeature],
                   R_init: np.ndarray, P_init: np.ndarray
                   ) -> Tuple[bool, np.ndarray, np.ndarray]:
        pts3d = []
        pts2d = []
        for feat in sfm_f:
            if not feat.state:
                continue
            for fi, pt in feat.observation:
                if fi == frame:
                    pts3d.append(feat.position)
                    pts2d.append(pt[:2])
        if len(pts3d) < 4:
            return False, R_init, P_init
        pts3d = np.array(pts3d, dtype=np.float64)
        pts2d = np.array(pts2d, dtype=np.float64)
        K = np.eye(3)
        rvec, _ = cv2.Rodrigues(R_init)
        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            pts3d, pts2d, K, None,
            rvec.astype(np.float64), P_init.reshape(3, 1).astype(np.float64),
            useExtrinsicGuess=True,
        )
        if not ok:
            return False, R_init, P_init
        R, _ = cv2.Rodrigues(rvec)
        return True, R, tvec.ravel()

    def _bundle_adjustment(self, frame_num: int,
                            q_arr: list, T_arr: list,
                            sfm_f: List[SFMFeature]) -> bool:
        """Simplified bundle adjustment using scipy least_squares."""
        # Build parameter vector: [q0_axis_angle(3), T0(3)] * frame_num + [X(3)] * n_pts
        frames = [i for i in range(frame_num) if q_arr[i] is not None]
        pts_idx = [k for k, f in enumerate(sfm_f) if f.state]

        if len(pts_idx) == 0:
            return True

        def pack(q_list, T_list, feats):
            params = []
            for i in frames:
                from scipy.spatial.transform import Rotation
                R = quat_to_rot(q_list[i])
                aa = Rotation.from_matrix(R).as_rotvec()
                params.extend(aa.tolist())
                params.extend(T_list[i].tolist())
            for k in pts_idx:
                params.extend(feats[k].position.tolist())
            return np.array(params)

        def unpack(params):
            from scipy.spatial.transform import Rotation
            n_frames = len(frames)
            q_out = {}
            T_out = {}
            for idx, i in enumerate(frames):
                aa = params[idx*6: idx*6+3]
                t = params[idx*6+3: idx*6+6]
                R = Rotation.from_rotvec(aa).as_matrix()
                q_out[i] = normalize_quat(rot_to_quat(R))
                T_out[i] = t.copy()
            pts_out = {}
            base = n_frames * 6
            for jj, k in enumerate(pts_idx):
                pts_out[k] = params[base + jj*3: base + jj*3+3].copy()
            return q_out, T_out, pts_out

        def residuals(params):
            q_d, T_d, pts_d = unpack(params)
            res = []
            for k in pts_idx:
                feat = sfm_f[k]
                X = pts_d[k]
                for fi, pt in feat.observation:
                    if fi not in T_d:
                        continue
                    R = quat_to_rot(q_d[fi])
                    t = T_d[fi]
                    Xc = R @ X + t
                    if abs(Xc[2]) < 1e-6:
                        continue
                    xp = Xc[:2] / Xc[2]
                    res.extend((xp - pt[:2]).tolist())
            return np.array(res) if res else np.zeros(1)

        x0 = pack(q_arr, T_arr, sfm_f)
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=200,
                                    ftol=1e-4, xtol=1e-4)
        except Exception:
            return False

        q_d, T_d, pts_d = unpack(result.x)
        for i, qi in q_d.items():
            q_arr[i] = qi
            T_arr[i] = T_d[i]
        for k, X in pts_d.items():
            sfm_f[k].position = X
        return True
