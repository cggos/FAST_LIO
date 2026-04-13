# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

FAST-LIO2 is a ROS1 C++ package implementing a tightly-coupled LiDAR-inertial odometry system. It fuses LiDAR point clouds with IMU data via an iterated extended Kalman filter (iEKF) on manifolds, using an incremental KD-Tree (ikd-Tree) for fast map management.

## Build

This is a catkin package. Build from the workspace root (`ws_fastlio_ros1/`):

```bash
# Source livox_ros_driver first (required dependency)
source $LIVOX_WS/devel/setup.bash

cd ws_fastlio_ros1
catkin_make
source devel/setup.bash
```

The CMakeLists.txt sets `CMAKE_BUILD_TYPE` to `"Debug"` but applies `-O3` via `CMAKE_CXX_FLAGS`. On x86 with >4 cores, OpenMP parallelism (`MP_EN`, `MP_PROC_NUM=3`) is enabled automatically.

## Running

Each LiDAR type has its own launch file and YAML config:

| LiDAR | Launch | Config |
|-------|--------|--------|
| Livox Avia | `mapping_avia.launch` | `config/avia.yaml` |
| Livox Horizon | `mapping_horizon.launch` | `config/horizon.yaml` |
| Velodyne | `mapping_velodyne.launch` | `config/velodyne.yaml` |
| Ouster 64 | `mapping_ouster64.launch` | `config/ouster64.yaml` |
| Mid-360 | `mapping_mid360.launch` | `config/mid360.yaml` |
| MARSIM simulator | `mapping_marsim.launch` | `config/marsim.yaml` |

```bash
roslaunch fast_lio mapping_avia.launch
# Then play a rosbag or start the LiDAR driver
rosbag play YOUR_BAG.bag
```

PCD map output is saved to `FAST_LIO/PCD/scans.pcd` when `pcd_save_en: true`.

## Code Architecture

All core logic lives in the `src/` directory (flat structure — no subdirectories):

- **`src/laserMapping.cpp`** — Main node. Subscribes to LiDAR and IMU topics, runs the iEKF update loop, publishes odometry/path/point clouds. Contains the scan-to-map registration using nearest-neighbor search on the ikd-Tree.

- **`src/preprocess.cpp` / `include/preprocess.h`** — LiDAR point cloud preprocessing. Handles the `LID_TYPE` enum (`AVIA=1`, `VELO16=2`, `OUST64=3`, `MARSIM=4`). Converts raw sensor messages to `PointCloudXYZI` (`pcl::PointXYZINormal`) with per-point timestamps for motion undistortion.

- **`include/IMU_Processing.hpp`** — IMU forward propagation and backward propagation for motion undistortion. Manages the iEKF state prediction step using IMU measurements between LiDAR scans.

- **`include/use-ikfom.hpp`** — Defines the iEKF state manifold using IKFoM macros: `state_ikfom` (position, rotation SO3, LiDAR-IMU extrinsic R/T, velocity, gyro bias, accel bias, gravity S2) and the process model functions.

- **`include/ikd-Tree/`** — Incremental KD-Tree library for dynamic 3D map management (insert/delete/search without full rebuilds).

- **`include/IKFoM_toolkit/`** — Iterated Kalman Filter on Manifolds (IKFoM) library providing the on-manifold filter primitives.

- **`include/common_lib.h`** — Shared types, constants, and utility functions used across modules.

- **`include/so3_math.h`** / **`include/Exp_mat.h`** — SO(3) math utilities (exponential map, log map, hat/vee operators).

## Key Configuration Parameters

In any `config/*.yaml`, the critical parameters to tune when adapting to a new sensor:

```yaml
common:
    lid_topic: "/livox/lidar"   # LiDAR ROS topic
    imu_topic: "/livox/imu"     # IMU ROS topic

preprocess:
    lidar_type: 1   # 1=Livox, 2=Velodyne, 3=Ouster, 4=MARSIM
    scan_line: 6    # Number of scan lines (for spinning LiDARs)
    blind: 4        # Minimum range (meters) to ignore

mapping:
    extrinsic_est_en: false     # Set false when extrinsic is known
    extrinsic_T: [x, y, z]     # LiDAR position in IMU frame
    extrinsic_R: [3x3 matrix]  # LiDAR rotation in IMU frame (row-major)
    det_range: 450.0            # Map management radius (meters)
```

Extrinsics are defined as the LiDAR's pose **in the IMU body frame** (IMU is base frame).

## Dependencies

- ROS Melodic or newer (ROS1)
- PCL >= 1.8
- Eigen >= 3.3.4
- `livox_ros_driver` (must be sourced before build and runtime)
- OpenMP (optional, auto-detected)
- Python/matplotlib (for runtime plotting via `matplotlibcpp.h`)
