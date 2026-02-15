# TODO

## 1. VESC 변환 파라미터 중앙 관리 구조로 변경

### 현재 문제점

VESC Ackermann 변환 파라미터(`speed_to_erpm_gain`, `steering_angle_to_servo_gain` 등)가 각 launch 파일에 하드코딩되어 있어 관리가 어렵고, 실제로 값이 불일치하는 버그가 존재한다.

| 파라미터 | ackermann_to_vesc (명령 변환) | vesc_to_odom (오돔 변환) |
|---------|------|------|
| `speed_to_erpm_gain` | 4614.0 | **1.0 (잘못됨)** |
| `speed_to_erpm_offset` | 0.0 | 0.0 |
| `steering_angle_to_servo_gain` | -1.2135 | **1.0 (잘못됨)** |
| `steering_angle_to_servo_offset` | 0.5304 | **0.0 (잘못됨)** |
| `wheelbase` | - | 0.2 (확인 필요) |

이로 인해 `/odom` 토픽의 속도와 위치가 **완전히 틀린 값**으로 발행되고 있다.

### 관련 파일

- `sensor/vesc/vesc_ackermann/launch/ackermann_to_vesc_node.launch.xml` - 명령 변환 launch
- `sensor/vesc/vesc_ackermann/launch/vesc_to_odom_node.launch.xml` - 오돔 변환 launch
- `sensor/vesc/vesc_driver/params/vesc_config.yaml` - VESC 하드웨어 설정 (현재 게인 미포함)
- `stack_master/launch/sensors.launch.xml` - 상위 launch (파라미터 오버라이드 없이 include만 함)
- `stack_master/scripts/simple_mux_node.py` - 서보 게인 기본값 별도 하드코딩

### 작업 내용

#### 1-1. `vesc_config.yaml`에 변환 파라미터 추가

```yaml
# 기존 하드웨어 설정 아래에 추가
speed_to_erpm_gain: 4614.0
speed_to_erpm_offset: 0.0
steering_angle_to_servo_gain: -1.2135
steering_angle_to_servo_offset: 0.5304
wheelbase: 0.33
```

- `wheelbase` 값은 실차 측정 후 정확한 값으로 입력 (현재 launch에는 0.2로 되어 있으나 확인 필요)
- `speed_to_erpm_gain`은 실차 캘리브레이션 후 정확한 값으로 조정

#### 1-2. 각 launch 파일에서 yaml 파라미터를 읽도록 수정

**방법 A: `sensors.launch.xml`에서 arg로 전달**

`sensors.launch.xml`에서 yaml을 읽고, 각 하위 launch에 arg로 넘기는 방식.

**방법 B: 각 노드에서 직접 yaml을 로드**

각 launch 파일에서 `<param from="..."/>` 으로 yaml을 직접 로드. 단, 네임스페이스 충돌에 주의.

#### 1-3. `simple_mux_node.py`의 하드코딩 기본값도 통일

현재 `simple_mux_node.py`에 `steering_angle_to_servo_gain = -1.2135`, `steering_angle_to_servo_offset = 0.5` 로 별도 하드코딩되어 있음. yaml에서 읽거나 launch에서 전달하도록 변경.

### 기대 효과

- 파라미터를 `vesc_config.yaml` 한 곳에서 관리하여 불일치 방지
- 차량 튜닝 시 yaml 파일 하나만 수정하면 모든 노드에 반영
- `/odom` 토픽의 속도/위치가 정확한 값으로 발행됨
