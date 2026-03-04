#!/bin/bash
set -e

INPUT_GID=$(getent group input | cut -d: -f3 2>/dev/null || echo 995)

mkdir -p ~/creating_autonomous_car_ws/src/cache/build \
         ~/creating_autonomous_car_ws/src/cache/install \
         ~/creating_autonomous_car_ws/src/cache/log

cd ~/creating_autonomous_car_ws/src/creating_autonomous_car

INPUT_GID=$INPUT_GID docker compose build dev

code .
