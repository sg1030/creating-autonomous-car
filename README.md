# Creating Autonomous Car - ROS2 Workspace

ROS2 Jazzy workspace for autonomous vehicle development.

## Features

- **VESC**: VESC-based Ackermann steering control
- **2D LiDAR**: Hokuyo LiDAR and pre-built IMU sensor
- **SLAM**: Cartographer-based 2D SLAM mapping and localization
- **Particle filter**: Alternative 2d LiDAR based Localization
- **Simulator**: F1TENTH gym-based simulator with obstacle support (static + dynamic)
- **Planner**: Centerline extraction from map + trajectory optimization (student implementation)

## Prerequisites

- Ubuntu 24.04
- ROS2 Jazzy

## Installation

### 1. Clone Repository

```bash
mkdir -p ~/creating_autonmous_car_ws/src
cd ~/creating_autonmous_car_ws/src
git clone https://github.com/HMCL-UNIST/creating_autonomous_car.git
```

### 2. Install Dependencies

```bash
cd ~/creating_autonmous_car_ws

# Initialize rosdep (first time only)
sudo apt update
sudo apt install python3-rosdep python3-pip
sudo rosdep init
rosdep update

# Install package dependencies
rosdep install --from-paths src --ignore-src -r -y

# Simulator dependencies
sudo apt install ros-jazzy-xacro
pip install --upgrade coverage --break-system-packages

# Planner dependencies (centerline extraction)
sudo apt install python3-skimage
```

### 3. Build Workspace

Add this alias to your `~/.bashrc`:

```bash
cd ~/creating_autonmous_car_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release


# (optional)
echo "alias cb='colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
source ~/.bashrc
```


### 4. Install Rangelibc for particle filter

```bash
cd ~/creating_autonmous_car_ws/src/creating_autonomous_car/slam/range_libc/pywrapper

sudo apt install cython3
pip3 install . --user --break-system-packages

python3 -c "import range_libc; print('range_libc import OK')"

```

### 5. Source Workspace

```bash
source ~/creating_autonmous_car_ws/install/setup.bash

# (optional) Add to .bashrc for automatic sourcing
echo "source ~/creating_autonmous_car_ws/install/setup.bash" >> ~/.bashrc
```



## Usage

### Launch Sensors

```bash
ros2 launch stack_master sensors.launch.xml
```

This launches:
- URG LiDAR node
- IMU sensor (Lord Microstrain GX5)
- VESC motor driver
- Ackermann to VESC converter
- VESC to odometry converter
- Simple command multiplexer

### Run joy node
Run the joy node at the appropriate location (e.g., onboard, remote laptop).
Depending on the connection method(bluetooth, wired connection to laptop, wireless connection with receiver), the joystick key mapping may vary.
```bash
ros2 run joy joy_node
```

### cartographer Mapping

Start mapping to create a new map:

```bash
ros2 launch stack_master mapping.launch.xml map:=<map_name>
```

**Save the map**:
1. Open RViz
2. Click **"2D Goal Pose"** button
3. Click anywhere on the map
4. Map will be saved to `src/stack_master/maps/<map_name>/`

Files saved:
- `<map_name>.pbstream` - Cartographer state file
- `<map_name>.png` - Occupancy grid image
- `<map_name>.yaml` - Map metadata

### cartographer Localization

Use an existing map for localization:

```bash
ros2 launch stack_master cartographer_localization.launch.xml map:=<map_name>
```

**Set initial pose**:
1. Open RViz
2. Click **"2D Pose Estimate"** button
3. Click on the map where the robot is located
4. Drag to set orientation
5. Cartographer will reinitialize localization at the specified pose

### Centerline Extraction & Trajectory Optimization

Extract centerline from a map image:

```bash
ros2 launch stack_master create_path.launch.xml map:=<map_name>
```

This will:
1. Extract track centerline via skeletonization
2. Save `centerline.csv` to `stack_master/maps/<map_name>/`
3. Show matplotlib visualization (skeleton, centerline, track bounds)
4. Publish RViz markers (`/centerline_waypoints/markers`, `/track_bounds/markers`)

Options:
- `reverse` - Reverse direction, `true`=CW (default: `false`=CCW)
- `output_csv` - Output filename (default: `centerline.csv`)
- `optimize` - Run trajectory optimization (default: `false`)

Output CSV format: `x_m, y_m, w_tr_right_m, w_tr_left_m`

Additionally, `boundary_right.csv` and `boundary_left.csv` (wall contour points) are saved for accurate wall-distance computation.

### Waypoint CSV Standard Format

All waypoint CSVs (centerline, global_waypoints, etc.) follow this column order:

| Column | Name | Description | Required |
|--------|------|-------------|----------|
| 1 | `x_m` | X coordinate [m] | Yes |
| 2 | `y_m` | Y coordinate [m] | Yes |
| 3 | `w_tr_right_m` | Right track width [m] | No |
| 4 | `w_tr_left_m` | Left track width [m] | No |
| 5 | `psi_rad` | Heading [rad] | No |
| 6 | `kappa_radpm` | Curvature [rad/m] | No |
| 7 | `vx_mps` | Velocity [m/s] | No |

Missing columns are handled automatically by `waypoint_publisher`:
- `psi_rad` → computed from atan2(dy, dx)
- `kappa_radpm` → 0.0
- `vx_mps` → 0.0
- `d_right/d_left` → computed from boundary CSVs if available, otherwise from `w_tr` columns

### Simulator

Launch the simulator with a map:

```bash
ros2 launch stack_master base_system.launch.xml sim:=true map:=<map_name>
```

This launches:
- F1TENTH gym simulator (gym_bridge)
- Static obstacle manager (interactive markers in RViz)
- `/map` topic published from simulator (with obstacle integration)
- LiDAR is configured to 2160 beams to match the class vehicle setup

#### Static Obstacles

- **Place**: Click in RViz using the **"Publish Point"** tool to add a static obstacle at that location
- **Clear all**: Left-click the green **"Clear Obstacles"** button in RViz (located at (0, -5) in map frame)

#### Dynamic Obstacle

Launch a dynamic obstacle that follows a trajectory CSV:

```bash
ros2 launch stack_master obstacle_publisher.launch.xml map:=<map_name>
```

Options:
- `trajectory_csv` - CSV file to follow (default: `centerline.csv`)
- `speed_scaler` - Speed multiplier (default: `0.7`)
- `reactive` - Enable lateral oscillation mode (default: `false`)
- `reactive_freq` - Oscillation frequency in Hz (default: `0.3`)
- `starting_s` - Starting position on trajectory in meters (default: `0.0`)

### particle-filter Localization
TODO

## License

Copyright 2026 HMCL-UNIST

Licensed under the Apache License, Version 2.0

## Contributors

- Jeongsang Ryu (ryujs@unist.ac.kr)
