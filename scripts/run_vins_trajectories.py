#!/usr/bin/env python3
"""Run VINS over ROS2 EuRoC bags and record estimated trajectories."""

from __future__ import annotations

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


HEADER = ["timestamp_ns", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z"]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0.0 else -q


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def rot_to_quat(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        quat = np.array([
            0.25 / scale,
            (rot[2, 1] - rot[1, 2]) * scale,
            (rot[0, 2] - rot[2, 0]) * scale,
            (rot[1, 0] - rot[0, 1]) * scale,
        ])
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        quat = np.array([
            (rot[2, 1] - rot[1, 2]) / scale,
            0.25 * scale,
            (rot[0, 1] + rot[1, 0]) / scale,
            (rot[0, 2] + rot[2, 0]) / scale,
        ])
    elif rot[1, 1] > rot[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        quat = np.array([
            (rot[0, 2] - rot[2, 0]) / scale,
            (rot[0, 1] + rot[1, 0]) / scale,
            0.25 * scale,
            (rot[1, 2] + rot[2, 1]) / scale,
        ])
    else:
        scale = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        quat = np.array([
            (rot[1, 0] - rot[0, 1]) / scale,
            (rot[0, 2] + rot[2, 0]) / scale,
            (rot[1, 2] + rot[2, 1]) / scale,
            0.25 * scale,
        ])
    return normalize_quat(quat)


def pose_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = quat_to_rot(quaternion)
    transform[:3, 3] = position
    return transform


def parse_body_t_cam(config_path: Path, camera: str) -> np.ndarray:
    text = config_path.read_text()
    match = re.search(
        rf"{re.escape('body_T_' + camera)}:\s*!!opencv-matrix.*?data:\s*\[([^\]]+)\]",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"could not find body_T_{camera} in {config_path}")
    values = [float(value) for value in re.split(r"[,\s]+", match.group(1).strip()) if value]
    if len(values) != 16:
        raise ValueError(f"body_T_{camera} has {len(values)} values, expected 16")
    return np.array(values, dtype=float).reshape(4, 4)


class OdometryRecorder(Node):
    def __init__(self) -> None:
        super().__init__("vins_trajectory_recorder")
        self.rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.create_subscription(Odometry, "/vins_estimator/odometry", self._callback, 100)

    def _callback(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        quaternion = np.array([
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
        ])
        self.rows.append((stamp, position, normalize_quat(quaternion)))


def bash_command(command: str, cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        ["bash", "-lc", f"source install/setup.bash && {command}"],
        cwd=cwd,
        preexec_fn=os.setsid,
    )


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(os.getpgid(process.pid), signal.SIGINT)
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)


def write_trajectory(path: Path, rows: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for stamp, position, quaternion in rows:
            writer.writerow([
                stamp,
                f"{position[0]:.9f}",
                f"{position[1]:.9f}",
                f"{position[2]:.9f}",
                f"{quaternion[0]:.9f}",
                f"{quaternion[1]:.9f}",
                f"{quaternion[2]:.9f}",
                f"{quaternion[3]:.9f}",
            ])


def write_sequence_outputs(
    output_dir: Path,
    sequence: str,
    body_rows: list[tuple[int, np.ndarray, np.ndarray]],
    body_t_cam: dict[str, np.ndarray],
) -> None:
    for camera, transform in body_t_cam.items():
        camera_rows = []
        for stamp, body_position, body_quaternion in body_rows:
            world_t_body = pose_matrix(body_position, body_quaternion)
            world_t_camera = world_t_body @ transform
            camera_rows.append((
                stamp,
                world_t_camera[:3, 3].copy(),
                rot_to_quat(world_t_camera[:3, :3]),
            ))
        base = output_dir / sequence / camera
        write_trajectory(base / "keyframe_trajectory.csv", body_rows)
        write_trajectory(base / "camera_trajectory.csv", camera_rows)


def discover_bags(data_dir: Path) -> list[Path]:
    return sorted(path.parent for path in data_dir.rglob("metadata.yaml") if path.parent.name.endswith("_ros2"))


def run_sequence(repo: Path, bag: Path, config: Path, rate: float) -> list[tuple[int, np.ndarray, np.ndarray]]:
    rclpy.init()
    recorder = OdometryRecorder()
    executor_thread = threading.Thread(target=rclpy.spin, args=(recorder,), daemon=True)
    executor_thread.start()

    vins = bash_command(f"ros2 run vins vins_node {config}", repo)
    time.sleep(2.0)
    player = bash_command(
        "ros2 bag play "
        f"{bag} --rate {rate} --topics /imu0 /cam0/image_raw /cam1/image_raw",
        repo,
    )
    player.wait()
    time.sleep(2.0)
    terminate(vins)

    recorder.destroy_node()
    rclpy.shutdown()
    executor_thread.join(timeout=3.0)
    return recorder.rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/vins_fusion"))
    parser.add_argument("--config", type=Path, default=Path("config/euroc/euroc_stereo_imu_config.yaml"))
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--sequence", action="append", help="Sequence name, e.g. V1_01_easy. Can be repeated.")
    args = parser.parse_args()

    repo = Path.cwd()
    body_t_cam = {
        "cam0": parse_body_t_cam(args.config, "cam0"),
        "cam1": parse_body_t_cam(args.config, "cam1"),
    }
    bags = discover_bags(args.data_dir)
    if args.sequence:
        wanted = set(args.sequence)
        bags = [bag for bag in bags if bag.name.removesuffix("_ros2") in wanted]

    manifest_rows = []
    for bag in bags:
        sequence = bag.name.removesuffix("_ros2")
        print(f"running {sequence} from {bag}")
        rows = run_sequence(repo, bag, args.config, args.rate)
        if not rows:
            print(f"warning: no odometry recorded for {sequence}", file=sys.stderr)
            continue
        write_sequence_outputs(args.output_dir, sequence, rows, body_t_cam)
        manifest_rows.append((sequence, len(rows)))
        print(f"wrote {len(rows)} poses for {sequence}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "poses"])
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
