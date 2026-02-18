# TODO

## 1. 제어 명령 인터페이스 통일 (Sim/Real)

### 목적

센서 런치를 통해 구동되는 제어 패키지들 (simple_mux, joystick 등)이 시뮬레이터에서도 실제 차량과 동일하게 작동하도록 통일한다.

### 현재 상황

**실제 차량:**
- Joystick → `/joy` topic
- simple_mux → `/ackermann_cmd` 출력
  - Joy 명령, High-level planner 명령 등을 mux
- `/ackermann_cmd` → VESC driver → 모터 제어

**시뮬레이터:**
- Keyboard teleop → `/cmd_vel` (Twist)
- gym_bridge가 직접 구독 → 시뮬레이터 제어
- simple_mux 없음

### 문제점

1. 시뮬레이터에서 simple_mux가 작동하지 않음
2. High-level planner 명령이 시뮬레이터에 전달되지 않음
3. 제어 인터페이스가 sim/real에서 다름 (`/cmd_vel` vs `/ackermann_cmd`)

### 해결 방안

**시뮬레이터도 동일한 제어 파이프라인 사용:**

```
Joystick/Planner → simple_mux → /ackermann_cmd → gym_bridge → Simulator
```

#### 수정 사항:

1. **gym_bridge 수정:**
   - `/cmd_vel` 구독 제거
   - `/ackermann_cmd` (AckermannDriveStamped) 구독 추가

2. **simple_mux를 시뮬레이터에서도 실행:**
   - `base_system.launch.xml`의 COMMON COMPONENTS에 simple_mux 추가
   - sim/real 모두 동일한 mux 로직 사용

3. **Joystick/Keyboard 통일:**
   - Keyboard → Joy message 변환 노드 추가
   - 또는 joy_node를 키보드 입력으로 사용

### 기대 효과

- High-level planner를 시뮬레이터에서 테스트 가능
- 제어 인터페이스 통일로 코드 재사용성 증가
- 시뮬레이터와 실차 간 전환 간편화

---

## 2. Localization 출력을 시뮬레이터 Topic 형식에 맞추기

### 목적

실제 차량의 Localization 출력(Cartographer/Particle Filter)을 시뮬레이터의 `/car_state/odom` topic 형식으로 통일하여, 동일한 플래너/컨트롤러 코드를 sim/real에서 사용 가능하게 한다.

### 현재 상황

**시뮬레이터 모드:**
- `/car_state/odom` (Odometry) - ground truth pose
- `/car_state/pose` (PoseStamped) - ground truth pose
- Frame: `car_state/base_link`

**실제 차량 모드:**
- `/odom` (Odometry) - wheel encoder 기반 odometry (drift 발생, **유지 필요**)
- Cartographer/PF - `map` frame에서의 추정 위치
  - 현재 TF만 제공: `map` → `base_link`
  - Odometry topic 없음

### 해결 방안

**실차 Localization이 시뮬과 동일한 형식으로 publish:**

1. Cartographer/PF가 추정 pose를 `/car_state/odom` (Odometry)로 publish
2. `/car_state/pose` (PoseStamped)도 함께 publish
3. Frame: `map` → `base_link` TF는 기존대로 유지
4. Wheel odometry (`/odom`)는 그대로 유지 (다른 용도로 사용 가능)

### 작업 내용

#### Localization Wrapper Node 작성

Cartographer/PF의 TF를 읽어서 Odometry message로 변환:

```python
# localization_odom_publisher.py
- Input: TF lookup (map → base_link)
- Output: /car_state/odom (Odometry), /car_state/pose (PoseStamped)
```

### 기대 효과

- **코드 재사용**: 플래너/컨트롤러가 시뮬/실차 구분 없이 `/car_state/odom` 사용
- **Wheel odometry 보존**: `/odom`은 그대로 유지 (디버깅, 슬립 검출 등에 활용)
- **인터페이스 통일**: 시뮬레이터 개발 후 실차 적용 간편화

---

## 3. Trajectory Optimization

### 목적

센터라인(`centerline.csv`)을 기반으로 최적 주행 경로를 생성한다.

### 현재 상황

- 센터라인 추출 완료 (`ros2 launch stack_master create_path.launch.xml map:=<map_name>`)
- `planner/trajectory_optimizer` 노드 자리만 생성 (미구현)
- `create_path.launch.xml`에서 `optimize:=true`로 실행 가능

### 작업 내용

1. `planner/planner/trajectory_optimizer.py` 구현
2. 센터라인 + 트랙 폭(`w_tr_right_m`, `w_tr_left_m`) 활용
3. 최적화된 경로를 `global_waypoints.csv`로 출력 (속도 프로파일 `vx_mps` 포함)
