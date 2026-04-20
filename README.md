# Python VINS-Fusion for ROS2

This repository contains a Python implementation of VINS-Fusion running on ROS2. The evaluation script is included in the ```scrip/``` folder. Further detailed are described in our [presentation](https://youtu.be/Gr1ZjeuqnIY).

![vins fusion demo](./imgs/vinsfusion_demo%20-%2001.gif)
*Demo of vins fusion, visualized in RViz2 using EuRoC MAV dataset (Vicon Room1 Medium).*

![Project Poster](./imgs/Poster.pptx.jpg)

## Requirements

- ROS2 with `rclpy`, `cv_bridge`, `tf2_ros`, `sensor_msgs`, `nav_msgs`, and `geometry_msgs`
- Python 3.10+
- `uv`

Source ROS2:

```bash
source /opt/ros/<ros2-distro>/setup.bash
```

## Install Python Dependencies

From the repository root:

```bash
uv sync
```

## Build The ROS2 Python Packages

```bash
colcon build --symlink-install
```

Source the overlay:

```bash
source install/setup.bash
```

## Run VINS

Run the estimator directly:

```bash
ros2 run vins vins_node config/euroc/euroc_stereo_imu_config.yaml
```

Or launch it:

```bash
ros2 launch vins vins.launch.py \
  config_path:=config/euroc/euroc_stereo_imu_config.yaml
```

Other useful configs:

```bash
ros2 run vins vins_node config/euroc/euroc_mono_imu_config.yaml
ros2 run vins vins_node config/euroc/euroc_stereo_config.yaml
ros2 run vins vins_node config/realsense_d435i/realsense_stereo_imu_config.yaml
```

## Run Loop Fusion

Start VINS first, then in another terminal:

```bash
cd /media/weiliu/SSD/code/vins_fusion_ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run vins_loop_fusion loop_fusion_node config/euroc/euroc_stereo_imu_config.yaml
```

## Convert ROS1 Bags

If your dataset is still in ROS1 bag format:

```bash
uv pip install rosbags
rosbags-convert /path/to/input.bag --dst /path/to/output_ros2
```

Then play the converted ROS2 bag:

```bash
ros2 bag play /path/to/output_ros2
```

For normal ROS2 usage, prefer the `colcon build` and `ros2 run` workflow above.
