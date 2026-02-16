# DEBUG

## 해결된 이슈

### 1. numba + coverage 충돌
- **증상**: `gym_bridge` 실행 시 `AttributeError: module 'coverage.types' has no attribute 'Tracer'`
- **원인**: 시스템 `coverage 7.4`와 `numba 0.63`의 API 비호환
- **해결**: `pip install --upgrade coverage --break-system-packages`

### 2. xacro 미설치
- **증상**: `gym_bridge_launch.py` 실행 시 `file not found: 'xacro'`
- **원인**: 시뮬레이터 launch에서 로봇 URDF 로드 시 xacro 필요
- **해결**: `sudo apt install ros-jazzy-xacro`

