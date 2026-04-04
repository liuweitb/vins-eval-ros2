# VINS-Fusion Implementation in ROS2

This is a VINS-Fusion implementation using ROS2 Humble

Based on the original [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) by HKUST Aerial Robotics Group.

## Prerequisites

- Docker
- (optional) A ROS2 bag in ROS2 format

## Build Docker Image

We implemented a docker image for unified development environment.

```bash
cd docker
make build
```

## Run Container

Before executing into the docker container, display access to the master should be enabled using:

```bash
xhost local:docker
```

Run the image and execute the container:

```bash
./exec.sh
```

Inside the container, the workspace is mounted at `/workspace`.

## Build Packages

```bash
colcon build --symlink-install
source install/setup.bash
```

## Run

### EuRoC — stereo + IMU

```bash
ros2 run vins vins_node /workspace/config/euroc/euroc_stereo_imu_config.yaml
# optional loop closure
ros2 run loop_fusion loop_fusion_node /workspace/config/euroc/euroc_stereo_imu_config.yaml
```

### EuRoC — mono + IMU

```bash
ros2 run vins vins_node /workspace/config/euroc/euroc_mono_imu_config.yaml
```

### EuRoC — stereo only

```bash
ros2 run vins vins_node /workspace/config/euroc/euroc_stereo_config.yaml
```

### Visualisation

```bash
ros2 launch vins vins_rviz.launch.xml
```

### KITTI Odometry (stereo)

```bash
ros2 run vins kitti_odom_test /workspace/config/kitti_odom/kitti_config00-02.yaml /path/to/kitti/sequences/00/
```

### KITTI GPS Fusion (stereo + GPS)

```bash
ros2 run vins kitti_gps_test /workspace/config/kitti_raw/kitti_10_03_config.yaml /path/to/kitti/2011_10_03_drive_0027_sync/
ros2 run global_fusion global_fusion_node
```

## Play a ROS2 Bag

```bash
ros2 bag play /workspace/data/<bag_name>
```

To convert a ROS1 bag:

Since the bag file provided by ETH is in ros1 format, one should convert them using ```rosbag-convert```.

```bash
pip install rosbags
rosbags-convert data/V1_02_medium.bag --dst data/V1_02_medium_ros2
```

## Packages

| Package | Description |
| --- | --- |
| `camera_models` | Camera calibration library (pinhole, mei, equidistant, catadioptric) |
| `vins` | Core VIO estimator |
| `loop_fusion` | Visual loop closure using DBoW2 + BRIEF |
| `global_fusion` | GPS/global pose fusion using GeographicLib |
