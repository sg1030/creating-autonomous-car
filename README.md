# Creating Autonomous Car - ROS2 Workspace

ROS2 Jazzy workspace for autonomous vehicle development.

## Features

- **VESC**: VESC-based Ackermann steering control
- **2D LiDAR**: Hokuyo LiDAR and pre-built IMU sensor
- **SLAM**: Cartographer-based 2D SLAM mapping and localization
- **Particle filter**: Alternative 2d LiDAR based Localization

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

### particle-filter Localization
TODO

## License

Copyright 2026 HMCL-UNIST

Licensed under the Apache License, Version 2.0

## Contributors

- Jeongsang Ryu (ryujs@unist.ac.kr)
