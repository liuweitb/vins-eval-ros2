import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import List


class _OpenCVYamlLoader(yaml.SafeLoader):
    pass


def _opencv_matrix_constructor(loader: yaml.Loader, node: yaml.Node) -> dict:
    return loader.construct_mapping(node, deep=True)


_OpenCVYamlLoader.add_constructor("tag:yaml.org,2002:opencv-matrix", _opencv_matrix_constructor)


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        text = f.read()

    lines = text.splitlines()
    if lines and lines[0].startswith("%YAML:"):
        lines = lines[1:]

    data = yaml.load("\n".join(lines), Loader=_OpenCVYamlLoader)
    return data or {}


@dataclass
class Parameters:
    # IMU
    use_imu: bool = True
    imu_topic: str = "/imu0"
    acc_n: float = 0.1
    gyr_n: float = 0.01
    acc_w: float = 0.001
    gyr_w: float = 0.0001
    g_norm: float = 9.81007

    # Camera
    num_of_cam: int = 1
    image0_topic: str = "/cam0/image_raw"
    image1_topic: str = "/cam1/image_raw"
    cam0_calib: str = ""
    cam1_calib: str = ""
    image_width: int = 752
    image_height: int = 480

    # Extrinsics (body_T_cam): list of 4x4 np.ndarray
    body_T_cam: List[np.ndarray] = field(default_factory=list)
    estimate_extrinsic: int = 0  # 0=fixed, 1=optimize, 2=init only

    # Time offset
    estimate_td: bool = False
    td: float = 0.0
    rolling_shutter: bool = False

    # Feature tracking
    max_cnt: int = 150
    min_dist: int = 30
    f_threshold: float = 1.0
    flow_back: bool = True

    # Sliding window
    window_size: int = 10
    min_parallax: float = 10.0

    # Optimization
    max_solver_time: float = 0.04
    max_num_iterations: int = 8
    optimize_every_n_frames: int = 5

    # Misc
    multiple_thread: bool = True
    equalize: bool = True
    fisheye: bool = False
    use_vicon_as_odometry: bool = False
    vicon_topic: str = "/vicon/firefly_sbx/firefly_sbx"

    @classmethod
    def from_yaml(cls, path: str) -> "Parameters":
        data = _load_yaml(path)

        p = cls()
        p.use_imu = bool(data.get("imu", 1))
        p.imu_topic = data.get("imu_topic", p.imu_topic)
        p.acc_n = float(data.get("acc_n", p.acc_n))
        p.gyr_n = float(data.get("gyr_n", p.gyr_n))
        p.acc_w = float(data.get("acc_w", p.acc_w))
        p.gyr_w = float(data.get("gyr_w", p.gyr_w))
        p.g_norm = float(data.get("g_norm", p.g_norm))

        p.num_of_cam = int(data.get("num_of_cam", p.num_of_cam))
        p.image0_topic = data.get("image0_topic", p.image0_topic)
        p.image1_topic = data.get("image1_topic", p.image1_topic)
        p.cam0_calib = data.get("cam0_calib", "")
        p.cam1_calib = data.get("cam1_calib", "")
        p.image_width = int(data.get("image_width", p.image_width))
        p.image_height = int(data.get("image_height", p.image_height))

        p.estimate_extrinsic = int(data.get("estimate_extrinsic", 0))
        p.estimate_td = bool(data.get("estimate_td", 0))
        p.td = float(data.get("td", 0.0))
        p.rolling_shutter = bool(data.get("ROLLING_SHUTTER", 0))

        p.max_cnt = int(data.get("max_cnt", p.max_cnt))
        p.min_dist = int(data.get("min_dist", p.min_dist))
        p.f_threshold = float(data.get("F_threshold", p.f_threshold))
        p.flow_back = bool(data.get("flow_back", 1))

        p.window_size = int(data.get("window_size", p.window_size))
        p.min_parallax = float(data.get("keyframe_parallax", p.min_parallax))

        p.max_solver_time = float(data.get("max_solver_time", p.max_solver_time))
        p.max_num_iterations = int(data.get("max_num_iterations", p.max_num_iterations))
        p.optimize_every_n_frames = int(data.get("optimize_every_n_frames", p.optimize_every_n_frames))
        p.multiple_thread = bool(data.get("multiple_thread", 1))
        p.equalize = bool(data.get("equalize", 1))
        p.fisheye = bool(data.get("fisheye", 0))
        p.use_vicon_as_odometry = bool(data.get("use_vicon_as_odometry", 0))
        p.vicon_topic = data.get("vicon_topic", p.vicon_topic)

        # Extrinsics
        for i in range(p.num_of_cam):
            key = f"body_T_cam{i}"
            if key in data:
                rows = data[key]["data"]
                T = np.array(rows, dtype=np.float64).reshape(4, 4)
            else:
                T = np.eye(4)
            p.body_T_cam.append(T)

        return p
