#!/usr/bin/env python3
"""Generate EuRoC camera and image-timestamp trajectory files.

The repository's local data is laid out as EuRoC MAV sequences.  This script
uses the state ground truth as the body/IMU trajectory, applies the configured
body-to-camera extrinsics, and writes per-sequence, per-camera trajectories.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GT_HEADER = [
    "timestamp_ns",
    "p_x",
    "p_y",
    "p_z",
    "q_w",
    "q_x",
    "q_y",
    "q_z",
]


@dataclass(frozen=True)
class Pose:
    timestamp_ns: int
    position: np.ndarray
    quaternion: np.ndarray


def normalize_quat(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("zero-length quaternion")
    q = q / norm
    return q if q[0] >= 0.0 else -q


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def rot_to_quat(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        quat = np.array(
            [
                0.25 / scale,
                (rot[2, 1] - rot[1, 2]) * scale,
                (rot[0, 2] - rot[2, 0]) * scale,
                (rot[1, 0] - rot[0, 1]) * scale,
            ]
        )
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        quat = np.array(
            [
                (rot[2, 1] - rot[1, 2]) / scale,
                0.25 * scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
            ]
        )
    elif rot[1, 1] > rot[2, 2]:
        scale = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        quat = np.array(
            [
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                0.25 * scale,
                (rot[1, 2] + rot[2, 1]) / scale,
            ]
        )
    else:
        scale = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        quat = np.array(
            [
                (rot[1, 0] - rot[0, 1]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
                (rot[1, 2] + rot[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    return normalize_quat(quat)


def slerp(q0: np.ndarray, q1: np.ndarray, ratio: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quat(q0 + ratio * (q1 - q0))

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * ratio
    scale0 = np.sin(theta_0 - theta) / sin_theta_0
    scale1 = np.sin(theta) / sin_theta_0
    return normalize_quat(scale0 * q0 + scale1 * q1)


def read_groundtruth(path: Path) -> list[Pose]:
    poses: list[Pose] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            poses.append(
                Pose(
                    timestamp_ns=int(row[0]),
                    position=np.array([float(row[1]), float(row[2]), float(row[3])]),
                    quaternion=np.array([float(row[4]), float(row[5]), float(row[6]), float(row[7])]),
                )
            )
    if not poses:
        raise ValueError(f"no poses found in {path}")
    return poses


def read_camera_timestamps(path: Path) -> list[int]:
    timestamps: list[int] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].startswith("#"):
                continue
            timestamps.append(int(row[0]))
    return timestamps


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


def interpolate_pose(poses: list[Pose], timestamp_ns: int, start_index: int) -> tuple[Pose, int]:
    if timestamp_ns <= poses[0].timestamp_ns:
        return poses[0], 0
    while start_index + 1 < len(poses) and poses[start_index + 1].timestamp_ns < timestamp_ns:
        start_index += 1
    if start_index + 1 >= len(poses):
        return poses[-1], len(poses) - 1

    left = poses[start_index]
    right = poses[start_index + 1]
    span = right.timestamp_ns - left.timestamp_ns
    ratio = 0.0 if span == 0 else (timestamp_ns - left.timestamp_ns) / span
    position = left.position + ratio * (right.position - left.position)
    quaternion = slerp(left.quaternion, right.quaternion, ratio)
    return Pose(timestamp_ns, position, quaternion), start_index


def transform_body_to_camera(body_pose: Pose, body_t_cam: np.ndarray) -> Pose:
    world_r_body = quat_to_rot(body_pose.quaternion)
    body_r_cam = body_t_cam[:3, :3]
    body_p_cam = body_t_cam[:3, 3]
    world_r_cam = world_r_body @ body_r_cam
    world_p_cam = body_pose.position + world_r_body @ body_p_cam
    return Pose(body_pose.timestamp_ns, world_p_cam, rot_to_quat(world_r_cam))


def write_trajectory(path: Path, poses: list[Pose]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(GT_HEADER)
        for pose in poses:
            writer.writerow(
                [
                    pose.timestamp_ns,
                    f"{pose.position[0]:.9f}",
                    f"{pose.position[1]:.9f}",
                    f"{pose.position[2]:.9f}",
                    f"{pose.quaternion[0]:.9f}",
                    f"{pose.quaternion[1]:.9f}",
                    f"{pose.quaternion[2]:.9f}",
                    f"{pose.quaternion[3]:.9f}",
                ]
            )


def sequence_dirs(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.glob("*/*") if (path / "mav0").is_dir())


def generate(data_dir: Path, output_dir: Path, config_path: Path) -> list[tuple[str, str, int]]:
    body_t_cam = {
        "cam0": parse_body_t_cam(config_path, "cam0"),
        "cam1": parse_body_t_cam(config_path, "cam1"),
    }
    generated: list[tuple[str, str, int]] = []

    for sequence in sequence_dirs(data_dir):
        mav0 = sequence / "mav0"
        body_poses = read_groundtruth(mav0 / "state_groundtruth_estimate0" / "data.csv")
        for camera, transform in body_t_cam.items():
            timestamps = [
                timestamp
                for timestamp in read_camera_timestamps(mav0 / camera / "data.csv")
                if body_poses[0].timestamp_ns <= timestamp <= body_poses[-1].timestamp_ns
            ]
            sampled_body: list[Pose] = []
            camera_poses: list[Pose] = []
            gt_index = 0
            for timestamp in timestamps:
                body_pose, gt_index = interpolate_pose(body_poses, timestamp, gt_index)
                sampled_body.append(body_pose)
                camera_poses.append(transform_body_to_camera(body_pose, transform))

            out_dir = output_dir / sequence.name / camera
            write_trajectory(out_dir / "keyframe_trajectory.csv", sampled_body)
            write_trajectory(out_dir / "camera_trajectory.csv", camera_poses)
            generated.append((sequence.name, camera, len(camera_poses)))

    return generated


def write_manifest(output_dir: Path, generated: list[tuple[str, str, int]]) -> None:
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "camera", "frames", "keyframe_trajectory", "camera_trajectory"])
        for sequence, camera, frames in generated:
            base = Path(sequence) / camera
            writer.writerow(
                [
                    sequence,
                    camera,
                    frames,
                    str(base / "keyframe_trajectory.csv"),
                    str(base / "camera_trajectory.csv"),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("out"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/euroc/euroc_stereo_imu_config.yaml"),
        help="EuRoC config containing body_T_cam0 and body_T_cam1.",
    )
    args = parser.parse_args()

    generated = generate(args.data_dir, args.output_dir, args.config)
    write_manifest(args.output_dir, generated)
    total_frames = sum(frames for _, _, frames in generated)
    print(f"generated {len(generated)} camera datasets with {total_frames} poses in {args.output_dir}")


if __name__ == "__main__":
    main()
