"""GPS/global pose fusion for VINS-Fusion.

This mirrors the original C++ global optimization module: VIO odometry gives
relative pose constraints, GPS gives absolute position constraints, and scipy
least_squares refines the global pose chain.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading

import numpy as np
from pyproj import Transformer
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def _rot_from_wxyz(q: np.ndarray) -> Rotation:
    q = _normalize_quat_wxyz(q)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def _wxyz_from_rot(rot: Rotation) -> np.ndarray:
    q_xyzw = rot.as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _pose_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rot_from_wxyz(quat_wxyz).as_matrix()
    T[:3, 3] = position
    return T


@dataclass
class Pose:
    position: np.ndarray
    orientation: np.ndarray

    @classmethod
    def from_arrays(cls, position: np.ndarray, orientation: np.ndarray) -> "Pose":
        return cls(
            position=np.asarray(position, dtype=float).reshape(3),
            orientation=_normalize_quat_wxyz(np.asarray(orientation, dtype=float).reshape(4)),
        )


@dataclass
class GPSMeasurement:
    position: np.ndarray
    accuracy: float


class LocalCartesianProjector:
    """Convert WGS84 latitude/longitude/altitude to local ENU coordinates."""

    def __init__(self) -> None:
        self._lla_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
        self._origin_ecef: np.ndarray | None = None
        self._ecef_to_enu: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self._origin_ecef is not None

    def forward(self, latitude: float, longitude: float, altitude: float) -> np.ndarray:
        lon = float(longitude)
        lat = float(latitude)
        alt = float(altitude)
        ecef = np.array(self._lla_to_ecef.transform(lon, lat, alt), dtype=float)

        if self._origin_ecef is None:
            self._origin_ecef = ecef
            self._ecef_to_enu = self._enu_rotation(lat, lon)

        return self._ecef_to_enu @ (ecef - self._origin_ecef)

    @staticmethod
    def _enu_rotation(latitude_deg: float, longitude_deg: float) -> np.ndarray:
        lat = np.deg2rad(latitude_deg)
        lon = np.deg2rad(longitude_deg)
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)
        return np.array(
            [
                [-sin_lon, cos_lon, 0.0],
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
            ],
            dtype=float,
        )


class GlobalOptimization:
    """Fuse VIO pose chains with GPS position measurements."""

    def __init__(
        self,
        vio_t_var: float = 0.1,
        vio_q_var: float = 0.01,
        max_iterations: int = 20,
    ) -> None:
        self.vio_t_var = float(vio_t_var)
        self.vio_q_var = float(vio_q_var)
        self.max_iterations = int(max_iterations)

        self.projector = LocalCartesianProjector()
        self.local_pose_map: OrderedDict[float, Pose] = OrderedDict()
        self.global_pose_map: OrderedDict[float, Pose] = OrderedDict()
        self.gps_position_map: dict[float, GPSMeasurement] = {}

        self.wgps_t_wvio = np.eye(4)
        self.last_pose = Pose.from_arrays(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        self._lock = threading.RLock()

    def add_gps(
        self,
        stamp: float,
        latitude: float,
        longitude: float,
        altitude: float,
        position_accuracy: float = 1.0,
        optimize: bool = True,
    ) -> None:
        xyz = self.gps_to_xyz(latitude, longitude, altitude)
        accuracy = float(position_accuracy)
        if not np.isfinite(accuracy) or accuracy <= 0.0:
            accuracy = 1.0

        with self._lock:
            self.gps_position_map[float(stamp)] = GPSMeasurement(xyz, accuracy)

        if optimize:
            self.optimize()

    def add_vio_pose(self, stamp: float, position: np.ndarray, orientation: np.ndarray) -> Pose:
        local_pose = Pose.from_arrays(position, orientation)
        with self._lock:
            t = float(stamp)
            self.local_pose_map[t] = local_pose
            self.local_pose_map = OrderedDict(sorted(self.local_pose_map.items()))

            global_position = self.wgps_t_wvio[:3, :3] @ local_pose.position + self.wgps_t_wvio[:3, 3]
            global_rotation = Rotation.from_matrix(self.wgps_t_wvio[:3, :3]) * _rot_from_wxyz(
                local_pose.orientation
            )
            global_pose = Pose.from_arrays(global_position, _wxyz_from_rot(global_rotation))

            self.global_pose_map[t] = global_pose
            self.global_pose_map = OrderedDict(sorted(self.global_pose_map.items()))
            self.last_pose = global_pose
            return global_pose

    def get_global_pose(self) -> Pose:
        with self._lock:
            return Pose.from_arrays(self.last_pose.position, self.last_pose.orientation)

    def get_global_path(self) -> list[tuple[float, Pose]]:
        with self._lock:
            return [
                (t, Pose.from_arrays(pose.position, pose.orientation))
                for t, pose in self.global_pose_map.items()
            ]

    def gps_to_xyz(self, latitude: float, longitude: float, altitude: float) -> np.ndarray:
        return self.projector.forward(latitude, longitude, altitude)

    def optimize(self) -> bool:
        with self._lock:
            times = list(self.local_pose_map.keys())
            if len(times) < 2 or not self.gps_position_map:
                return False

            local_poses = [self.local_pose_map[t] for t in times]
            global_poses = [self.global_pose_map[t] for t in times]
            x0 = self._pack(global_poses)

        def residuals(x: np.ndarray) -> np.ndarray:
            poses = self._unpack(x)
            res: list[np.ndarray] = []

            for i in range(len(poses) - 1):
                local_i = local_poses[i]
                local_j = local_poses[i + 1]
                meas_t, meas_r = self._relative_pose(local_i, local_j)

                global_i = poses[i]
                global_j = poses[i + 1]
                pred_t, pred_r = self._relative_pose(global_i, global_j)

                res.append((pred_t - meas_t) / self.vio_t_var)
                rot_err = (meas_r.inv() * pred_r).as_rotvec()
                res.append(rot_err / self.vio_q_var)

            for i, t in enumerate(times):
                gps = self.gps_position_map.get(t)
                if gps is not None:
                    res.append((poses[i].position - gps.position) / gps.accuracy)

            if not res:
                return np.zeros(0)
            return np.concatenate(res)

        result = least_squares(
            residuals,
            x0,
            loss="huber",
            f_scale=1.0,
            max_nfev=self.max_iterations,
        )

        optimized = self._unpack(result.x)
        with self._lock:
            for t, pose in zip(times, optimized):
                self.global_pose_map[t] = pose

            self.last_pose = optimized[-1]
            last_t = times[-1]
            local_T = _pose_matrix(
                self.local_pose_map[last_t].position,
                self.local_pose_map[last_t].orientation,
            )
            global_T = _pose_matrix(optimized[-1].position, optimized[-1].orientation)
            self.wgps_t_wvio = global_T @ np.linalg.inv(local_T)

        return bool(result.success)

    @staticmethod
    def _relative_pose(pose_i: Pose, pose_j: Pose) -> tuple[np.ndarray, Rotation]:
        r_i = _rot_from_wxyz(pose_i.orientation)
        r_j = _rot_from_wxyz(pose_j.orientation)
        rel_t = r_i.inv().apply(pose_j.position - pose_i.position)
        rel_r = r_i.inv() * r_j
        return rel_t, rel_r

    @staticmethod
    def _pack(poses: list[Pose]) -> np.ndarray:
        chunks = []
        for pose in poses:
            chunks.append(pose.position)
            chunks.append(_rot_from_wxyz(pose.orientation).as_rotvec())
        return np.concatenate(chunks)

    @staticmethod
    def _unpack(x: np.ndarray) -> list[Pose]:
        poses = []
        for i in range(0, len(x), 6):
            position = x[i : i + 3]
            orientation = _wxyz_from_rot(Rotation.from_rotvec(x[i + 3 : i + 6]))
            poses.append(Pose.from_arrays(position, orientation))
        return poses


__all__ = ["GPSMeasurement", "GlobalOptimization", "LocalCartesianProjector", "Pose"]
