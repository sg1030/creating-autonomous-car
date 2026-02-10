# Creating Autonomous Car - ROS2 Workspace

ROS2 Jazzy workspace for autonomous vehicle development.

## Features

- **SLAM**: Cartographer-based 2D SLAM mapping and localization
- **Motor Control**: VESC-based Ackermann steering control
- **Sensors**: URG LiDAR and IMU sensor integration
- **Map Management**: Map saving and initial pose setting via RViz interface

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
```

### 3. Setup Build Alias (Optional)

Add this alias to your `~/.bashrc`:

```bash
echo "alias cb='colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'" >> ~/.bashrc
source ~/.bashrc
```

### 4. Build Workspace

```bash
cd ~/creating_autonmous_car_ws
cb  # or: colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### 5. Source Workspace

```bash
source ~/creating_autonmous_car_ws/install/setup.bash

# Add to .bashrc for automatic sourcing
echo "source ~/creating_autonmous_car_ws/install/setup.bash" >> ~/.bashrc
```

### 5-1. Install Rangelibc for particle filter

```bash
cd ~/creating_autonmous_car_ws/src/creating_autonomous_car/slam/range_libc/pywrapper

sudo apt install cython3
pip3 install . --user --break-system-packages

python3 -c "import range_libc; print('range_libc import OK')"

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

### Mapping Mode

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

### Localization Mode

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

## Package Structure

```
src/
├── stack_master/           # Main integration package
│   ├── launch/            # Launch files
│   ├── config/            # Cartographer configurations
│   ├── scripts/           # Python nodes
│   └── maps/              # Saved maps
├── sensor/
│   ├── vesc/              # VESC motor driver
│   └── urg_node/          # Hokuyo URG LiDAR driver
└── slam/
    ├── cartographer/      # Cartographer SLAM core
    ├── cartographer_ros/  # Cartographer ROS2 wrapper
    ├── particle_filter/   # Particle filter implementation
    └── range_libc/        # Fast ray-casting library
```

## Configuration Files

### Mapping Configuration
- `config/mapping_2d.lua` - Cartographer mapping parameters
- Lower `min_score` for easier loop closure detection
- Higher `optimize_every_n_nodes` for better map quality

### Localization Configuration
- `config/localization_2d.lua` - Cartographer localization parameters
- Higher `min_score` for robust localization
- Lower `optimize_every_n_nodes` for faster updates
- `pure_localization_trimmer` enabled to prevent map updates

## Troubleshooting

### Build fails
```bash
# Clean build
cd ~/creating_autonmous_car_ws
rm -rf build install log
cb
```

### Sensor not detected
```bash
# Check USB permissions
ls -l /dev/ttyACM* /dev/ttyUSB*

# Add user to dialout group
sudo usermod -a -G dialout $USER
# Logout and login again
```

### Map not loading
- Verify map files exist in `src/stack_master/maps/<map_name>/`
- Check file permissions
- Ensure map name matches in launch command

## License

Copyright 2026 HMCL-UNIST

Licensed under the Apache License, Version 2.0

## Contributors

- Jeongsang Ryu (jeongsangryu@gmail.com)
- HMCL-UNIST Team
