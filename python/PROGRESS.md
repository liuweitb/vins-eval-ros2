# VINS-Fusion Python Translation — Progress

## What's Done

### Root workspace
- `pyproject.toml` — uv workspace with 4 members + `[tool.uv.sources]` for intra-workspace deps
- `uv.lock` — generated via `uv lock`

---

### `camera_models/`  ✅ Complete
| File | Description |
|------|-------------|
| `camera_models/camera_base.py` | Abstract base class |
| `camera_models/pinhole_camera.py` | Pinhole + radial/tangential distortion |
| `camera_models/mei_camera.py` | MEI omnidirectional model |
| `camera_models/equidistant_camera.py` | Kannala-Brandt fisheye model |
| `camera_models/camera_factory.py` | YAML-driven factory (`CameraFactory.generate_camera_from_yaml`) |
| `camera_models/__init__.py` | |
| `pyproject.toml`, `package.xml` | |

---

### `vins/`  ✅ Complete
| File | Description |
|------|-------------|
| `vins/utility/utility.py` | Quaternion ops, SO(3) exp/log, skew, rotation helpers |
| `vins/factor/integration_base.py` | IMU pre-integration: mid-point rule, 15-state Jacobian, covariance |
| `vins/factor/projection_factors.py` | Mono / stereo-two-frame / stereo-one-frame reprojection residuals |
| `vins/factor/marginalization_factor.py` | Schur complement marginalization prior |
| `vins/feature_tracker/feature_tracker.py` | KLT optical flow, forward-backward check, F-matrix RANSAC, CLAHE |
| `vins/initial/solve_5pts.py` | 5-point essential matrix via `cv2.findEssentialMat` |
| `vins/initial/initial_sfm.py` | Incremental SfM + bundle adjustment (`scipy.optimize.least_squares`) |
| `vins/initial/initial_alignment.py` | Gyro bias estimation, linear visual-inertial alignment |
| `vins/estimator/parameters.py` | YAML config loader (`Parameters.from_yaml`) |
| `vins/estimator/feature_manager.py` | Sliding-window feature tracking, DLT triangulation |
| `vins/estimator/estimator.py` | Full sliding-window VIO: init phase + nonlinear phase, `scipy` backend |
| `vins/ros_node.py` | rclpy node: IMU/image callbacks, stereo sync, TF + odometry publisher |
| `pyproject.toml`, `package.xml` | |

---

### `loop_fusion/`  ✅ Complete
| File | Description |
|------|-------------|
| `loop_fusion/keyframe.py` | ORB descriptors, BFMatcher, F-matrix + PnP geometric verification |
| `loop_fusion/pose_graph.py` | Loop detection, 4-DOF pose graph optimisation (`scipy` L-BFGS-B) |
| `loop_fusion/ros_node.py` | rclpy node: subscribes to VINS odometry + images, publishes corrected path |
| `pyproject.toml`, `package.xml` | |

---

### `global_fusion/`  🔲 Skeleton only
| File | Status |
|------|--------|
| `pyproject.toml`, `package.xml`, `__init__.py` | ✅ Created |
| `global_fusion/global_opt.py` | ❌ Missing |
| `global_fusion/ros_node.py` | ❌ Missing |

---

## What's Still Needed

### 1. `global_fusion/global_opt.py`
Mirrors `globalOpt.h/cpp` + `Factors.h`. Key points:
- Use `pyproj` (PROJ) to convert GPS lat/lon/alt → local ENU Cartesian
- Maintain anchor point for the coordinate origin
- `scipy.optimize.least_squares` with two cost terms:
  - **TError** — GPS position constraint
  - **RelativeRTError** — relative pose between consecutive VIO frames
- Estimate SE(3) transform between VIO frame and GPS frame
- Propagate correction to all poses

### 2. `global_fusion/ros_node.py`
Mirrors `globalOptNode.cpp`:
- Subscribe to `sensor_msgs/NavSatFix` (GPS) and `/vins_estimator/odometry`
- Call `GlobalOptimization.add_gps()` and `add_vio_pose()`
- Publish global path on `/global_fusion/path`

### 3. ROS2 build files (all 4 packages)
`colcon`/`ament_python` also needs `setup.py` + `setup.cfg` alongside `pyproject.toml`:

```python
# setup.py
from setuptools import setup
setup()
```

```ini
# setup.cfg  (example for vins package)
[metadata]
name = vins

[options]
packages = find:

[options.entry_points]
console_scripts =
    vins_node = vins.ros_node:main
```

Repeat for `camera_models`, `loop_fusion`, `global_fusion` with their respective entry points.

### 4. Launch file
`vins/launch/vins.launch.py` — Python launch API replacing the existing XML launch file:
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    config = LaunchConfiguration('config')
    return LaunchDescription([
        DeclareLaunchArgument('config', description='Path to YAML config'),
        Node(package='vins', executable='vins_node', arguments=[config]),
        Node(package='vins_loop_fusion', executable='loop_fusion_node', arguments=[config]),
    ])
```

---

## Key Design Decisions

| C++ | Python equivalent |
|-----|-------------------|
| Eigen | numpy |
| Ceres `TinySolver` / LM | `scipy.optimize.least_squares(method='lm')` |
| `cv::calcOpticalFlowPyrLK` | `cv2.calcOpticalFlowPyrLK` (same) |
| Quaternion `[w, x, y, z]` | same convention throughout |
| DBoW2 vocabulary tree | ORB + `cv2.BFMatcher` (simpler, no offline vocabulary file needed) |
| GeographicLib | `pyproj` (PROJ library, `pyproj.Transformer`) |
| `rclcpp` | `rclpy` |
| `camera_models` (camodocal) | `vins-camera-models` Python package |

---

## Build & Run

```bash
# Install all packages
cd python/
uv sync

# Build with colcon (after adding setup.py/setup.cfg to each package)
cd ..
colcon build --packages-select vins_camera_models vins vins_loop_fusion vins_global_fusion

# Run
source install/setup.bash
ros2 run vins vins_node /path/to/config.yaml
ros2 run vins_loop_fusion loop_fusion_node /path/to/config.yaml
ros2 run vins_global_fusion global_fusion_node
```
