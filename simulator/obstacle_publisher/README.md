# Obstacle Publisher

## ⚠️ 주의!! 미완성!! ⚠️

**이 패키지는 현재 미완성 상태입니다.**
- 시뮬레이터와의 통합 작업이 완료되지 않았습니다.
- LiDAR 스캔에 장애물이 감지되지 않는 문제가 있습니다.
- 사용하지 마세요!

---

ROS2 패키지로, F1TENTH 시뮬레이터에서 동적 및 정적 장애물을 발행합니다. (개발 중단)

## 기능

1. **동적 장애물 (Dynamic Obstacle)**
   - 센터라인을 따라 움직이는 직사각형 장애물 (기본: 65cm x 35cm)
   - Occupancy Grid를 실시간으로 수정하여 LiDAR FOV 제한을 자연스럽게 시뮬레이션
   - 센터라인 CSV 파일에서 경로 읽기

2. **정적 장애물 (Static Obstacle)**
   - RViz의 "Publish Point" 기능으로 클릭하여 추가
   - 원형 장애물 (기본: 50cm 직경)
   - 베이스 맵에 영구적으로 추가됨

## 사용법

### 1. 센터라인 CSV 파일 준비

`stack_master/maps/<map_name>/centerline.csv` 파일 생성:

```csv
x_m,y_m,vx_mps,heading
0.0,0.0,2.0,0.0
1.0,0.0,2.5,0.1
2.0,0.5,3.0,0.2
...
```

필수 컬럼:
- `x_m`: X 좌표 (미터)
- `y_m`: Y 좌표 (미터)

선택 컬럼:
- `vx_mps`: 속도 (m/s) - 없으면 1.0 m/s 기본값
- `heading`: 방향 (라디안) - 없으면 경로로부터 자동 계산

### 2. 런치 파일 실행

**직접 실행:**
```bash
ros2 launch obstacle_publisher obstacle_publisher.launch.xml map_name:=f
```

**stack_master에서 실행:**
```bash
ros2 launch stack_master obstacle_publisher.launch.xml map:=f
```

### 3. 정적 장애물 추가

RViz에서:
1. "Add" → "By topic" → `/clicked_point` → "PointStamped"
2. RViz 상단의 "Publish Point" 버튼 클릭
3. 맵 위에서 원하는 위치 클릭
4. 장애물이 즉시 추가됨

## 런치 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `map_name` | `f` | 맵 이름 (stack_master/maps/ 에서 찾음) |
| `enable_dynamic` | `true` | 동적 장애물 활성화 |
| `speed_scaler` | `1.0` | 동적 장애물 속도 배율 |
| `constant_speed` | `false` | 일정 속도 사용 (true면 speed_scaler 값 사용) |
| `starting_s` | `0.0` | 센터라인 상의 초기 위치 (미터) |
| `obstacle_length_m` | `0.65` | 장애물 길이 (진행 방향, 미터) |
| `obstacle_width_m` | `0.35` | 장애물 너비 (수직 방향, 미터) |
| `update_rate` | `20` | 업데이트 주기 (Hz) |

## 토픽

### Subscribe
- `/clicked_point` (geometry_msgs/PointStamped): RViz publish_point로부터 정적 장애물 추가

### Publish
- `/map` (nav_msgs/OccupancyGrid): 장애물이 추가된 점유 격자 맵
- `/dynamic_obstacle_marker` (visualization_msgs/MarkerArray): 동적 장애물 시각화 마커 (빨간색 큐브)
- `/static_obstacle_markers` (visualization_msgs/MarkerArray): 정적 장애물 시각화 마커 (파란색 실린더)

## 파일 구조

```
simulator/obstacle_publisher/
├── obstacle_publisher/
│   ├── __init__.py
│   └── obstacle_publisher_grid.py    # 메인 노드
├── launch/
│   └── obstacle_publisher.launch.xml # 런치 파일
├── package.xml
├── setup.py
└── README.md

stack_master/
├── launch/
│   └── obstacle_publisher.launch.xml # stack_master 통합 런치
└── maps/
    └── <map_name>/
        ├── <map_name>.png
        ├── <map_name>.yaml
        └── centerline.csv            # 센터라인 경로 (사용자 제공)
```

## 예제

### 동적 장애물만 사용 (2배 속도):
```bash
ros2 launch obstacle_publisher obstacle_publisher.launch.xml map_name:=f speed_scaler:=2.0
```

### 정적 장애물만 사용 (동적 비활성화):
```bash
ros2 launch obstacle_publisher obstacle_publisher.launch.xml map_name:=f enable_dynamic:=false
```

### 커스텀 장애물 크기:
```bash
ros2 launch obstacle_publisher obstacle_publisher.launch.xml map_name:=f \
  obstacle_length_m:=0.8 obstacle_width_m:=0.4
```

## 참고

- 센터라인 CSV가 없으면 자동으로 더미 원형 경로 생성 (테스트용)
- 동적 장애물은 센터라인의 heading을 따라 회전
- 정적 장애물은 베이스 맵에 영구적으로 추가되며, 노드 재시작 시 초기화됨
- Occupancy Grid 수정 방식이므로 LiDAR 스캔에서 자연스럽게 감지됨

---

## 개발 중단 사유

F1TENTH gym 시뮬레이터는 시작 시 맵을 한 번 로드하고 내부적으로 레이캐스팅(raycasting)에 사용합니다.
ROS 토픽으로 발행되는 `/map`은 RViz 시각화용이며, 시뮬레이터의 LiDAR 스캔 생성에는 영향을 주지 않습니다.

장애물을 LiDAR에 감지되게 하려면 시뮬레이터 내부의 맵 이미지와 distance transform을 직접 수정해야 하는데,
이는 gym_bridge.py를 대폭 수정해야 하므로 현재는 보류되었습니다.
