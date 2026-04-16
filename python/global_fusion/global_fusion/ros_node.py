"""ROS2 node for GPS/global fusion."""
from __future__ import annotations

from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import NavSatFix

from .global_opt import GlobalOptimization, Pose


class GlobalFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("global_fusion")

        self.declare_parameter("gps_topic", "/gps")
        self.declare_parameter("vio_topic", "/vins_estimator/odometry")
        self.declare_parameter("global_odom_topic", "/global_fusion/odometry")
        self.declare_parameter("global_path_topic", "/global_fusion/path")
        self.declare_parameter("sync_tolerance", 0.01)
        self.declare_parameter("vio_translation_sigma", 0.1)
        self.declare_parameter("vio_rotation_sigma", 0.01)
        self.declare_parameter("max_iterations", 20)

        self.sync_tolerance = float(self.get_parameter("sync_tolerance").value)
        self.optimizer = GlobalOptimization(
            vio_t_var=float(self.get_parameter("vio_translation_sigma").value),
            vio_q_var=float(self.get_parameter("vio_rotation_sigma").value),
            max_iterations=int(self.get_parameter("max_iterations").value),
        )
        self._gps_queue: deque[NavSatFix] = deque()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self.create_subscription(
            NavSatFix,
            str(self.get_parameter("gps_topic").value),
            self._gps_callback,
            sensor_qos,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("vio_topic").value),
            self._vio_callback,
            sensor_qos,
        )

        self._pub_odom = self.create_publisher(
            Odometry,
            str(self.get_parameter("global_odom_topic").value),
            100,
        )
        self._pub_path = self.create_publisher(
            Path,
            str(self.get_parameter("global_path_topic").value),
            100,
        )
        self._path_msg = Path()
        self._path_msg.header.frame_id = "world"

        self.get_logger().info("Global fusion node started.")

    def _gps_callback(self, msg: NavSatFix) -> None:
        self._gps_queue.append(msg)
        while len(self._gps_queue) > 200:
            self._gps_queue.popleft()

    def _vio_callback(self, msg: Odometry) -> None:
        t = self._stamp_to_sec(msg.header.stamp)
        position = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=float,
        )
        orientation = np.array(
            [
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
            ],
            dtype=float,
        )

        global_pose = self.optimizer.add_vio_pose(t, position, orientation)
        self._consume_matching_gps(t)
        global_pose = self.optimizer.get_global_pose()

        self._publish_odometry(msg, global_pose)
        self._publish_path()

    def _consume_matching_gps(self, vio_t: float) -> None:
        while self._gps_queue:
            gps = self._gps_queue[0]
            gps_t = self._stamp_to_sec(gps.header.stamp)
            dt = gps_t - vio_t

            if abs(dt) <= self.sync_tolerance:
                self._gps_queue.popleft()
                accuracy = gps.position_covariance[0]
                if not np.isfinite(accuracy) or accuracy <= 0.0:
                    accuracy = 1.0
                self.optimizer.add_gps(
                    vio_t,
                    gps.latitude,
                    gps.longitude,
                    gps.altitude,
                    accuracy,
                )
                return

            if dt < -self.sync_tolerance:
                self._gps_queue.popleft()
                continue

            return

    def _publish_odometry(self, vio_msg: Odometry, pose: Pose) -> None:
        msg = Odometry()
        msg.header = vio_msg.header
        msg.header.frame_id = "world"
        msg.child_frame_id = "world"
        msg.pose.pose.position.x = float(pose.position[0])
        msg.pose.pose.position.y = float(pose.position[1])
        msg.pose.pose.position.z = float(pose.position[2])
        msg.pose.pose.orientation.w = float(pose.orientation[0])
        msg.pose.pose.orientation.x = float(pose.orientation[1])
        msg.pose.pose.orientation.y = float(pose.orientation[2])
        msg.pose.pose.orientation.z = float(pose.orientation[3])
        self._pub_odom.publish(msg)

    def _publish_path(self) -> None:
        path = self.optimizer.get_global_path()
        now = self.get_clock().now().to_msg()
        self._path_msg.header.stamp = now
        self._path_msg.poses.clear()

        for t, pose in path:
            ps = PoseStamped()
            ps.header.stamp = self._sec_to_stamp(t)
            ps.header.frame_id = "world"
            ps.pose.position.x = float(pose.position[0])
            ps.pose.position.y = float(pose.position[1])
            ps.pose.position.z = float(pose.position[2])
            ps.pose.orientation.w = float(pose.orientation[0])
            ps.pose.orientation.x = float(pose.orientation[1])
            ps.pose.orientation.y = float(pose.orientation[2])
            ps.pose.orientation.z = float(pose.orientation[3])
            self._path_msg.poses.append(ps)

        self._pub_path.publish(self._path_msg)

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def _sec_to_stamp(t: float):
        return Time(seconds=float(t)).to_msg()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
