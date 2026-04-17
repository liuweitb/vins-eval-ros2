"""Main ROS2 node"""
import sys
import threading
import numpy as np
from collections import deque
from pathlib import Path as FilePath
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
import cv2
from cv_bridge import CvBridge
from camera_models.camera_factory import CameraFactory

from .estimator.parameters import Parameters
from .estimator.estimator import Estimator
from .feature_tracker.feature_tracker import FeatureTracker
from .utility import normalize_quat, rot_to_quat


class StereoVisualOdometry:
    def __init__(self, cam0, cam1, body_t_cam0: np.ndarray, body_t_cam1: np.ndarray):
        self.cam0 = cam0
        self.cam1 = cam1
        self.body_t_cam0 = body_t_cam0
        self.cam0_t_body = np.linalg.inv(body_t_cam0)
        self.cam1_t_cam0 = np.linalg.inv(body_t_cam1) @ body_t_cam0
        self.world_t_cam = np.eye(4)
        self.prev_points_3d: dict[int, np.ndarray] = {}

    def process(self, img0: np.ndarray, img1: np.ndarray, features: dict[int, list[np.ndarray]]) -> Optional[dict]:
        if img1 is None or not features:
            return None

        current_points_3d, current_norm = self._triangulate_current(img0, img1, features)
        if len(current_points_3d) < 12:
            self.prev_points_3d = current_points_3d
            return None

        object_points = []
        image_points = []
        for fid, point_3d in self.prev_points_3d.items():
            if fid in current_norm:
                object_points.append(point_3d)
                image_points.append(current_norm[fid][:2])

        if len(object_points) >= 8:
            obj = np.asarray(object_points, dtype=np.float64)
            img = np.asarray(image_points, dtype=np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                img,
                np.eye(3, dtype=np.float64),
                None,
                iterationsCount=60,
                reprojectionError=0.01,
                confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if ok and inliers is not None and len(inliers) >= 8:
                R_cur_prev, _ = cv2.Rodrigues(rvec)
                t_cur_prev = tvec.reshape(3)
                prev_t_cur = np.eye(4)
                prev_t_cur[:3, :3] = R_cur_prev.T
                prev_t_cur[:3, 3] = -R_cur_prev.T @ t_cur_prev
                self.world_t_cam = self.world_t_cam @ prev_t_cur

        self.prev_points_3d = current_points_3d
        world_t_body = self.world_t_cam @ self.cam0_t_body
        return {
            "position": world_t_body[:3, 3].copy(),
            "orientation": normalize_quat(rot_to_quat(world_t_body[:3, :3])),
            "velocity": np.zeros(3),
        }

    def _triangulate_current(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        features: dict[int, list[np.ndarray]],
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        ids = []
        left_pixels = []
        for fid, obs_list in features.items():
            obs = obs_list[0]
            ids.append(fid)
            left_pixels.append([obs[3], obs[4]])
        if not left_pixels:
            return {}, {}

        pts0 = np.asarray(left_pixels, dtype=np.float32).reshape(-1, 1, 2)
        pts1, status, _ = cv2.calcOpticalFlowPyrLK(img0, img1, pts0, None)
        if pts1 is None or status is None:
            return {}, {}

        R10 = self.cam1_t_cam0[:3, :3]
        t10 = self.cam1_t_cam0[:3, 3]
        proj0 = np.hstack([np.eye(3), np.zeros((3, 1))])
        proj1 = np.hstack([R10, t10.reshape(3, 1)])

        points_3d: dict[int, np.ndarray] = {}
        norm_points: dict[int, np.ndarray] = {}
        for idx, fid in enumerate(ids):
            if not status[idx, 0]:
                continue
            left = pts0[idx, 0].astype(np.float64)
            right = pts1[idx, 0].astype(np.float64)
            bearing0 = self._normalize_bearing(self.cam0.lift_projective(left))
            bearing1 = self._normalize_bearing(self.cam1.lift_projective(right))
            point_h = cv2.triangulatePoints(
                proj0,
                proj1,
                bearing0[:2].reshape(2, 1),
                bearing1[:2].reshape(2, 1),
            )
            if abs(point_h[3, 0]) < 1e-10:
                continue
            point = point_h[:3, 0] / point_h[3, 0]
            if point[2] <= 0.1 or point[2] > 30.0:
                continue
            points_3d[fid] = point
            norm_points[fid] = bearing0
        return points_3d, norm_points

    @staticmethod
    def _normalize_bearing(bearing: np.ndarray) -> np.ndarray:
        bearing = np.asarray(bearing, dtype=np.float64)
        if abs(bearing[2]) < 1e-8:
            return bearing
        return bearing / bearing[2]


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
        self._stereo_vo: Optional[StereoVisualOdometry] = None
        self._configure_cameras(config_path)
        self.bridge = CvBridge()

        # Buffers
        self._imu_buf: deque[tuple] = deque()
        self._img0_buf: deque[tuple] = deque()
        self._img1_buf: deque[tuple] = deque()
        self._buf_mutex = threading.Lock()
        self._processed_frames = 0
        self._published_frames = 0

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
        if self.params.use_vicon_as_odometry:
            self.create_subscription(TransformStamped, self.params.vicon_topic,
                                      self._vicon_callback, sensor_qos)

        # Publishers
        self._pub_odom = self.create_publisher(Odometry, "/vins_estimator/odometry", 10)
        self._pub_path = self.create_publisher(Path, "/vins_estimator/path", 10)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._path_msg = Path()
        self._path_msg.header.frame_id = "world"
        self._publish_world_anchor()

        # Processing thread
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

        self.get_logger().info(f"VINS estimator started. Config: {config_path}")

    def _configure_cameras(self, config_path: str):
        if not self.params.cam0_calib:
            return
        cam0_path = self._resolve_calib_path(config_path, self.params.cam0_calib)
        cam0 = CameraFactory.generate_camera_from_yaml(str(cam0_path))
        self.tracker.set_camera(cam0)
        if self.params.num_of_cam == 2 and self.params.cam1_calib and len(self.params.body_T_cam) >= 2:
            cam1_path = self._resolve_calib_path(config_path, self.params.cam1_calib)
            cam1 = CameraFactory.generate_camera_from_yaml(str(cam1_path))
            self._stereo_vo = StereoVisualOdometry(
                cam0,
                cam1,
                self.params.body_T_cam[0],
                self.params.body_T_cam[1],
            )

    @staticmethod
    def _resolve_calib_path(config_path: str, calib_path: str) -> FilePath:
        path = FilePath(calib_path)
        if path.is_absolute():
            return path
        return FilePath(config_path).resolve().parent / path

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

    def _vicon_callback(self, msg: TransformStamped):
        position = np.array([
            msg.transform.translation.x,
            msg.transform.translation.y,
            msg.transform.translation.z,
        ], dtype=np.float64)
        orientation = np.array([
            msg.transform.rotation.w,
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
        ], dtype=np.float64)
        odom = {
            "position": position,
            "orientation": orientation,
            "velocity": np.zeros(3),
        }
        self._publish_odometry(odom, stamp=msg.header.stamp, child_frame_id="body")

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

        # Feed to estimator. Prefer stereo VO for live RViz output because it has
        # metric scale from the calibrated EuRoC stereo baseline.
        estimator_result = self.estimator.process_image(features, img_t)
        stereo_result = self._stereo_vo.process(img0, img1, features) if self._stereo_vo is not None else None
        result = stereo_result if stereo_result is not None else estimator_result
        self._processed_frames += 1
        if self._processed_frames % 30 == 0:
            self.get_logger().info(
                "processed=%d published=%d features=%d imu=%d solver=%s frame=%d init='%s' source=%s"
                % (
                    self._processed_frames,
                    self._published_frames,
                    len(features),
                    len(imu_msgs),
                    self.estimator.solver_flag.name,
                    self.estimator.frame_count,
                    self.estimator.last_init_status,
                    "stereo" if stereo_result is not None else "estimator",
                )
            )
        if result is not None and not self.params.use_vicon_as_odometry:
            self._publish_odometry(result)

    # ------------------------------------------------------------------ #
    #  Publishers                                                          #
    # ------------------------------------------------------------------ #
    def _publish_world_anchor(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "world"
        tf.child_frame_id = "vins_origin"
        tf.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(tf)

    def _publish_odometry(self, odom: dict, stamp=None, child_frame_id: str = "body"):
        now = stamp if stamp is not None else self.get_clock().now().to_msg()

        msg = Odometry()
        msg.header.stamp = now
        msg.header.frame_id = "world"
        msg.child_frame_id = child_frame_id

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
        self._published_frames += 1

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
        tf.child_frame_id = child_frame_id
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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
