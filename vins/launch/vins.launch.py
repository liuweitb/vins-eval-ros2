"""Launch the Python VINS-Fusion estimator."""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_config_path() -> str:
    repo_config = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "euroc"
        / "euroc_stereo_imu_config.yaml"
    )
    if repo_config.exists():
        return str(repo_config)
    return ""


def generate_launch_description() -> LaunchDescription:
    config_path = LaunchConfiguration("config_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value=_default_config_path(),
                description="Absolute path to the VINS-Fusion YAML config file.",
            ),
            Node(
                package="vins",
                executable="vins_node",
                name="vins_estimator",
                output="screen",
                arguments=[config_path],
            ),
        ]
    )
