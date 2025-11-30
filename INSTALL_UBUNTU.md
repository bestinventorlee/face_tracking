# 우분투에서 얼굴 트래킹 패키지 설치 가이드

## 사전 준비

### 1. ROS2 설치 확인

```bash
# ROS2 Humble (Ubuntu 22.04)
source /opt/ros/humble/setup.bash

# 또는 ROS2 Iron (Ubuntu 24.04)
source /opt/ros/iron/setup.bash
```

### 2. 시스템 패키지 업데이트

```bash
sudo apt update
sudo apt upgrade -y
```

## Python 패키지 설치 방법

### 방법 1: pip로 설치 (권장)

```bash
# pip 업그레이드
pip install --upgrade pip

# 각 패키지 개별 설치
pip install opencv-python>=4.5.0
pip install numpy>=1.19.0
pip install scipy>=1.7.0
pip install rclpy>=3.0.0
pip install std-msgs>=4.0.0

# 또는 한 번에 설치
pip install -r requirements_face_tracking.txt
```

### 방법 2: apt로 시스템 패키지 설치 (대안)

일부 패키지는 apt로도 설치 가능합니다:

```bash
# 시스템 패키지 설치
sudo apt install -y \
    python3-opencv \
    python3-numpy \
    python3-scipy \
    python3-rclpy

# std-msgs는 ROS2 패키지이므로 ROS2 설치 시 자동 포함됨
```

**참고**: `rclpy`와 `std-msgs`는 ROS2 설치 시 이미 포함되어 있을 수 있습니다.

## 전체 설치 절차

### 1단계: ROS2 워크스페이스 준비

```bash
# 워크스페이스 생성 (없는 경우)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
```

### 2단계: 패키지 복사

```bash
# face_tracking 패키지를 src 디렉토리로 복사
cp -r /path/to/face_tracking ~/ros2_ws/src/

# 또는 심볼릭 링크 (개발 중인 경우)
ln -s /path/to/face_tracking ~/ros2_ws/src/face_tracking
```

### 3단계: Python 의존성 설치

```bash
# pip로 설치 (권장)
pip3 install --user -r ~/ros2_ws/src/face_tracking/requirements_face_tracking.txt

# 또는 sudo로 시스템 전체 설치 (권장하지 않음)
sudo pip3 install -r ~/ros2_ws/src/face_tracking/requirements_face_tracking.txt
```

### 4단계: ROS2 의존성 설치

```bash
# rosdep 초기화 (처음 한 번만)
sudo rosdep init
rosdep update

# 패키지 의존성 설치
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 5단계: 패키지 빌드

```bash
cd ~/ros2_ws

# ROS2 환경 설정
source /opt/ros/humble/setup.bash  # 또는 iron

# 패키지 빌드
colcon build --packages-select face_tracking

# 워크스페이스 환경 설정
source install/setup.bash
```

### 6단계: 실행

```bash
# Launch 파일로 실행
ros2 launch face_tracking face_tracking.launch.py

# 또는 직접 실행
ros2 run face_tracking face_tracking_node.py
```

## 개별 패키지 설치 명령어

### opencv-python

```bash
# pip로 설치
pip3 install --user opencv-python>=4.5.0

# 또는 apt로 설치 (시스템 패키지)
sudo apt install -y python3-opencv

# 설치 확인
python3 -c "import cv2; print(cv2.__version__)"
```

### numpy

```bash
# pip로 설치
pip3 install --user numpy>=1.19.0

# 또는 apt로 설치
sudo apt install -y python3-numpy

# 설치 확인
python3 -c "import numpy; print(numpy.__version__)"
```

### scipy

```bash
# pip로 설치
pip3 install --user scipy>=1.7.0

# 또는 apt로 설치
sudo apt install -y python3-scipy

# 설치 확인
python3 -c "import scipy; print(scipy.__version__)"
```

### rclpy

**주의**: `python3-rclpy`는 일반 apt 패키지가 아닙니다. ROS2 패키지입니다.

```bash
# 방법 1: ROS2 설치 시 자동 포함 (권장)
# ROS2를 설치하면 rclpy가 자동으로 포함됨
source /opt/ros/humble/setup.bash  # 또는 iron
python3 -c "import rclpy; print(rclpy.__version__)"

# 방법 2: pip로 설치 (ROS2 미설치 시)
# 가상환경 사용
source ~/face_tracking_venv/bin/activate
pip install rclpy>=3.0.0

# 또는 --break-system-packages 사용
pip3 install --break-system-packages --user rclpy>=3.0.0

# 설치 확인
python3 -c "import rclpy; print(rclpy.__version__)"
```

자세한 내용은 `INSTALL_RCLPY.md` 참고

### std-msgs

```bash
# ROS2 패키지이므로 ROS2 설치 시 자동 포함됨
# 별도 설치 불필요

# 설치 확인
python3 -c "from std_msgs.msg import Float32MultiArray; print('OK')"
```

## 한 번에 설치하는 스크립트

```bash
#!/bin/bash
# 얼굴 트래킹 패키지 의존성 설치 스크립트

echo "Python 패키지 설치 중..."

# pip 업그레이드
pip3 install --upgrade pip --user

# 각 패키지 설치
pip3 install --user opencv-python>=4.5.0
pip3 install --user numpy>=1.19.0
pip3 install --user scipy>=1.7.0
pip3 install --user rclpy>=3.0.0

echo "설치 완료!"
```

## 설치 확인

모든 패키지가 제대로 설치되었는지 확인:

```bash
python3 << EOF
try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except ImportError:
    print("✗ OpenCV 설치 필요")

try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except ImportError:
    print("✗ NumPy 설치 필요")

try:
    import scipy
    print(f"✓ SciPy: {scipy.__version__}")
except ImportError:
    print("✗ SciPy 설치 필요")

try:
    import rclpy
    print(f"✓ rclpy: {rclpy.__version__}")
except ImportError:
    print("✗ rclpy 설치 필요")

try:
    from std_msgs.msg import Float32MultiArray, Empty
    print("✓ std_msgs: OK")
except ImportError:
    print("✗ std_msgs 설치 필요")
EOF
```

## 문제 해결

### pip 권한 오류

```bash
# --user 플래그 사용 (권장)
pip3 install --user <package>

# 또는 가상환경 사용
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install <package>
```

### 패키지 버전 충돌

```bash
# 특정 버전 설치
pip3 install --user opencv-python==4.8.0

# 또는 최신 버전
pip3 install --user --upgrade opencv-python
```

### ROS2 패키지 인식 안 됨

```bash
# ROS2 환경 재설정
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 패키지 경로 확인
echo $ROS_PACKAGE_PATH
```

