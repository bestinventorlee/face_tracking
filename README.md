# 얼굴 트래킹 ROS2 패키지

로봇 팔의 엔드이펙트에 장착된 카메라로 사람 얼굴을 인식하고, 얼굴의 움직임에 따라 카메라를 추적하도록 로봇을 제어하는 ROS2 패키지입니다.

## 빠른 시작

### 1. 자동 설치 (권장)

```bash
cd face_tracking
chmod +x install.sh
./install.sh
```

### 2. 수동 설치

```bash
# 1. 워크스페이스로 패키지 복사
cp -r face_tracking ~/ros2_ws/src/

# 2. 의존성 설치
pip install -r face_tracking/requirements_face_tracking.txt

# 3. 빌드
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select face_tracking
source install/setup.bash
```

### 3. 실행

```bash
# Launch 파일로 실행
ros2 launch face_tracking face_tracking.launch.py

# 또는 직접 실행
ros2 run face_tracking face_tracking_node.py
```

## 패키지 구조

```
face_tracking/
├── CMakeLists.txt              # 빌드 설정
├── package.xml                 # 패키지 메타데이터
├── requirements_face_tracking.txt  # Python 의존성
├── install.sh                  # 자동 설치 스크립트
├── scripts/
│   └── face_tracking_node.py  # 메인 노드
└── launch/
    └── face_tracking.launch.py # Launch 파일
```

## 의존성

- ROS2 (Humble 또는 Iron 권장)
- Python 3.8+
- opencv-python
- numpy
- scipy
- rclpy

## 자세한 문서

- 설치 가이드: `INSTALL_face_tracking.md` (상위 디렉토리)
- 사용 설명서: `README_face_tracking.md` (상위 디렉토리)

