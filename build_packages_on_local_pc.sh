#!/bin/bash
###############################################################################
# build_packages_on_local_pc.sh
#
# Build script for local PC (Ubuntu 24.04 + ROS2 Jazzy) - SIMULATION ONLY
# Car-only packages (sensors, SLAM, particle filter) are excluded via
# COLCON_IGNORE files already present in the repository.
#
# Usage: bash build_packages_on_local_pc.sh
###############################################################################

set -eo pipefail

# ============================================================================
# Resolve paths relative to this script
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "============================================"
echo " Building packages for LOCAL PC (Simulation)"
echo "============================================"
echo "Workspace : ${WS_DIR}"
echo "Source    : ${SCRIPT_DIR}"
echo ""

# ============================================================================
# 1. Install apt dependencies
# ============================================================================
echo "[1/6] Installing apt dependencies..."
sudo apt update
sudo apt install -y \
    python3-rosdep \
    python3-pip \
    python3-skimage \
    ros-jazzy-xacro \
    ros-jazzy-rmw-cyclonedds-cpp \
    gedit

# ============================================================================
# 2. Install Python dependencies (f1tenth_gym + transforms3d)
# ============================================================================
echo ""
echo "[2/6] Installing Python dependencies..."

# f1tenth_gym simulator (editable install)
echo "  Installing f1tenth_gym..."
pip install -e "${SCRIPT_DIR}/simulator/f1tenth_gym" --break-system-packages

# transforms3d (required by f1tenth_gym_ros)
echo "  Installing transforms3d..."
pip install transforms3d --break-system-packages

# coverage upgrade (fixes numba + system coverage conflict)
echo "  Upgrading coverage..."
pip install --upgrade coverage --break-system-packages

# pynput (keyboard/mouse input for teleop)
echo "  Installing pynput..."
pip install pynput --break-system-packages

# ============================================================================
# 3. Initialize rosdep and install ROS dependencies
# ============================================================================
echo ""
echo "[3/6] Running rosdep..."

# Initialize rosdep (skip if already initialized)
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
else
    echo "  rosdep already initialized, skipping init."
fi
rosdep update

# -r: continue despite unresolvable keys (car-only packages are ignored)
rosdep install --from-paths "${WS_DIR}/src" --ignore-src -r -y

# ============================================================================
# 4. Build workspace
# ============================================================================
echo ""
echo "[4/6] Building workspace with colcon..."
cd "${WS_DIR}"
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# ============================================================================
# 5. Source workspace
# ============================================================================
echo ""
echo "[5/6] Sourcing workspace..."
source "${WS_DIR}/install/setup.bash"

# ============================================================================
# 6. Update ~/.bashrc (idempotent)
# ============================================================================
echo ""
echo "[6/6] Updating ~/.bashrc..."

BASHRC="${HOME}/.bashrc"
SOURCE_LINE="source ${WS_DIR}/install/setup.bash"

if ! grep -qF "${SOURCE_LINE}" "${BASHRC}" 2>/dev/null; then
    echo "" >> "${BASHRC}"
    echo "# ROS2 creating_autonomous_car workspace" >> "${BASHRC}"
    echo "${SOURCE_LINE}" >> "${BASHRC}"
    echo "  Added source line to ~/.bashrc"
else
    echo "  Source line already in ~/.bashrc, skipping."
fi

# Add aliases (idempotent)
# Add RMW_IMPLEMENTATION (idempotent)
if ! grep -qF "RMW_IMPLEMENTATION" "${BASHRC}" 2>/dev/null; then
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> "${BASHRC}"
    echo "  Added RMW_IMPLEMENTATION to ~/.bashrc"
else
    echo "  RMW_IMPLEMENTATION already in ~/.bashrc, skipping."
fi

ALIASES=(
    "alias cb='cd ${WS_DIR} && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'"
    "alias sauce='source ${WS_DIR}/install/setup.bash'"
    "alias sb='source ~/.bashrc'"
    "alias gb='gedit ~/.bashrc'"
)
ALIAS_KEYS=("alias cb=" "alias sauce=" "alias sb=" "alias gb=")

for i in "${!ALIASES[@]}"; do
    if ! grep -qF "${ALIAS_KEYS[$i]}" "${BASHRC}" 2>/dev/null; then
        echo "${ALIASES[$i]}" >> "${BASHRC}"
        echo "  Added '${ALIAS_KEYS[$i]}' to ~/.bashrc"
    fi
done

echo ""
echo "============================================"
echo " Build complete! (LOCAL PC - Simulation)"
echo " Run 'source ~/.bashrc' or open a new terminal."
echo "============================================"
