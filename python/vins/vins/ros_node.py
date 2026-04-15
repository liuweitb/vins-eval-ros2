"""Main ROS2 node"""
import sys
import threading
import numpy as np
from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
import cv2
from cv_bridge import CvBridge

from .estimator.parameters import Parameters
from .estimator.estimator import Estimator
from .feature_tracker.feature_tracker import FeatureTracker


class VinsNode(Node):
    def __init__(self, config_path: str):
        super().__init__("vins_estimator")

        self.params = Parameters.from_yaml(config_path)
        self.estimator = Estimator(self.params)
        self.tracker = FeatureTracker(
            max_cnt=self.params.max_cnt,
            min_dist=self.params.min_dist,
            f_threshold=self.params.f_threshold,
            flow_back=self.params.flow_back,
            equalize=self.params.equalize,
        )
        self.bridge = CvBridge()

        # Buffers
        self._imu_buf: deque[tuple] = deque()
        self._img0_buf: deque[tuple] = deque()
        self._img1_buf: deque[tuple] = deque()
        self._buf_mutex = threading.Lock()

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        # Subscriptions
        self.create_subscription(Imu, self.params.imu_topic,
                                  self._imu_callback, sensor_qos)
        self.create_subscription(Image, self.params.image0_topic,
                                  self._image0_callback, sensor_qos)
        if self.params.num_of_cam == 2:
            self.create_subscription(Image, self.params.image1_topic,
                                      self._image1_callback, sensor_qos)

        # Publishers
        self._pub_odom = self.create_publisher(Odometry, "/vins_estimator/odometry", 10)
        self._pub_path = self.create_publisher(Path, "/vins_estimator/path", 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._path_msg = Path()
        self._path_msg.header.frame_id = "world"

        # Processing thread
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

        self.get_logger().info(f"VINS estimator started. Config: {config_path}")

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #
    def _imu_callback(self, msg: Imu):
        t = self._stamp_to_sec(msg.header.stamp)
        acc = np.array([msg.linear_acceleration.x,
                         msg.linear_acceleration.y,
                         msg.linear_acceleration.z])
        gyr = np.array([msg.angular_velocity.x,
                         msg.angular_velocity.y,
                         msg.angular_velocity.z])
        with self._buf_mutex:
            self._imu_buf.append((t, acc, gyr))

    def _image0_callback(self, msg: Image):
        t = self._stamp_to_sec(msg.header.stamp)
        img = self.bridge.imgmsg_to_cv2(msg, "mono8")
        with self._buf_mutex:
            self._img0_buf.append((t, img))

    def _image1_callback(self, msg: Image):
        t = self._stamp_to_sec(msg.header.stamp)
        img = self.bridge.imgmsg_to_cv2(msg, "mono8")
        with self._buf_mutex:
            self._img1_buf.append((t, img))

    # ------------------------------------------------------------------ #
    #  Processing loop                                                     #
    # ------------------------------------------------------------------ #
    def _process_loop(self):
        import time
        while rclpy.ok():
            self._process_measurements()
            time.sleep(0.002)

    def _process_measurements(self):
        with self._buf_mutex:
            if not self._img0_buf:
                return
            img_t, img0 = self._img0_buf[0]
            img1 = None
            if self.params.num_of_cam == 2:
                if not self._img1_buf:
                    return
                img1_t, img1 = self._img1_buf[0]
                # Synchronise stereo
                if abs(img_t - img1_t) > 0.003:
                    if img_t < img1_t:
                        self._img0_buf.popleft()
                    else:
                        self._img1_buf.popleft()
                    return
                self._img1_buf.popleft()
            self._img0_buf.popleft()

            # Collect IMU measurements up to image time
            imu_msgs = []
            while self._imu_buf and self._imu_buf[0][0] <= img_t:
                imu_msgs.append(self._imu_buf.popleft())

        # Feed IMU
        prev_t = None
        for imu_t, acc, gyr in imu_msgs:
            if prev_t is not None:
                dt = imu_t - prev_t
                if 0 < dt < 0.1:
                    self.estimator.process_imu(dt, acc, gyr)
            prev_t = imu_t

        # Track features
        features = self.tracker.track_image(img0, img_t, img1)

        # Feed to estimator
        result = self.estimator.process_image(features, img_t)
        if result is not None:
            self._publish_odometry(result)

    # ------------------------------------------------------------------ #
    #  Publishers                                                          #
    # ------------------------------------------------------------------ #
    def _publish_odometry(self, odom: dict):
        now = self.get_clock().now().to_msg()

        msg = Odometry()
        msg.header.stamp = now
        msg.header.frame_id = "world"
        msg.child_frame_id = "body"

        P = odom["position"]
        Q = odom["orientation"]   # [w,x,y,z]
        V = odom["velocity"]

        msg.pose.pose.position.x = float(P[0])
        msg.pose.pose.position.y = float(P[1])
        msg.pose.pose.position.z = float(P[2])
        msg.pose.pose.orientation.w = float(Q[0])
        msg.pose.pose.orientation.x = float(Q[1])
        msg.pose.pose.orientation.y = float(Q[2])
        msg.pose.pose.orientation.z = float(Q[3])
        msg.twist.twist.linear.x = float(V[0])
        msg.twist.twist.linear.y = float(V[1])
        msg.twist.twist.linear.z = float(V[2])
        self._pub_odom.publish(msg)

        # Path
        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = msg.pose.pose
        self._path_msg.header.stamp = now
        self._path_msg.poses.append(pose_stamped)
        self._pub_path.publish(self._path_msg)

        # TF
        tf = TransformStamped()
        tf.header = msg.header
        tf.child_frame_id = "body"
        tf.transform.translation.x = float(P[0])
        tf.transform.translation.y = float(P[1])
        tf.transform.translation.z = float(P[2])
        tf.transform.rotation = msg.pose.pose.orientation
        self._tf_broadcaster.sendTransform(tf)

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9


def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 2:
        print("Usage: vins_node <config.yaml>")
        sys.exit(1)
    node = VinsNode(sys.argv[1])
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
