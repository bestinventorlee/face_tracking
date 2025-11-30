# 우분투 PEP 668 오류 해결 방법

최신 우분투(Ubuntu 23.04+)에서는 Python 환경 보호를 위해 `externally-managed-environment` 오류가 발생합니다.

## 해결 방법

### 방법 1: 가상환경 사용 (가장 권장) ⭐

```bash
# 1. 가상환경 생성
python3 -m venv ~/face_tracking_venv

# 2. 가상환경 활성화
source ~/face_tracking_venv/bin/activate

# 3. pip 업그레이드
pip install --upgrade pip

# 4. 패키지 설치
pip install opencv-python>=4.5.0
pip install numpy>=1.19.0
pip install scipy>=1.7.0
pip install rclpy>=3.0.0

# 또는 requirements 파일 사용
pip install -r requirements_face_tracking.txt

# 5. 사용 시마다 가상환경 활성화 필요
# (터미널을 열 때마다)
source ~/face_tracking_venv/bin/activate
```

**장점**: 시스템 Python과 분리되어 안전함

### 방법 2: apt로 시스템 패키지 설치

```bash
# 시스템 패키지로 설치
sudo apt update
sudo apt install -y \
    python3-opencv \
    python3-numpy \
    python3-scipy \
    python3-rclpy

# std-msgs는 ROS2 설치 시 자동 포함됨
```

**장점**: 시스템 전체에서 사용 가능, 관리가 쉬움

### 방법 3: --break-system-packages 플래그 사용 (비권장)

```bash
# ⚠️ 주의: 시스템 Python 환경을 변경하므로 위험할 수 있음
pip3 install --break-system-packages --user opencv-python>=4.5.0
pip3 install --break-system-packages --user numpy>=1.19.0
pip3 install --break-system-packages --user scipy>=1.7.0
pip3 install --break-system-packages --user rclpy>=3.0.0
```

**주의**: 시스템 패키지와 충돌할 수 있음

### 방법 4: pipx 사용 (애플리케이션용)

```bash
# pipx 설치
sudo apt install -y pipx

# pipx로 설치 (애플리케이션용)
pipx install opencv-python
```

## 권장 설치 절차 (가상환경 사용)

### 1단계: 가상환경 생성 및 활성화

```bash
# 가상환경 생성
python3 -m venv ~/face_tracking_venv

# 활성화
source ~/face_tracking_venv/bin/activate

# 활성화 확인 (프롬프트에 (face_tracking_venv) 표시됨)
which python
```

### 2단계: 패키지 설치

```bash
# pip 업그레이드
pip install --upgrade pip

# 모든 패키지 한 번에 설치
pip install opencv-python>=4.5.0 numpy>=1.19.0 scipy>=1.7.0 rclpy>=3.0.0

# 또는 requirements 파일 사용
pip install -r /path/to/requirements_face_tracking.txt
```

### 3단계: ROS2 패키지 빌드 및 실행

```bash
# ROS2 환경 설정
source /opt/ros/humble/setup.bash

# 가상환경 활성화 (매번 필요)
source ~/face_tracking_venv/bin/activate

# 패키지 빌드
cd ~/ros2_ws
colcon build --packages-select face_tracking
source install/setup.bash

# 실행
ros2 launch face_tracking face_tracking.launch.py
```

## 편의 스크립트 생성

가상환경을 자동으로 활성화하는 스크립트:

```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
alias face_tracking='source ~/face_tracking_venv/bin/activate && source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash'
```

사용법:
```bash
face_tracking
ros2 launch face_tracking face_tracking.launch.py
```

## 설치 확인

```bash
# 가상환경 활성화 후
source ~/face_tracking_venv/bin/activate

# 패키지 확인
python3 << 'EOF'
import cv2
import numpy
import scipy
import rclpy
from std_msgs.msg import Float32MultiArray

print(f"✓ OpenCV: {cv2.__version__}")
print(f"✓ NumPy: {numpy.__version__}")
print(f"✓ SciPy: {scipy.__version__}")
print(f"✓ rclpy: {rclpy.__version__}")
print("✓ std_msgs: OK")
EOF
```

## 문제 해결

### 가상환경이 활성화되지 않음

```bash
# python3-venv 패키지 설치
sudo apt install -y python3-venv python3-full

# 가상환경 재생성
rm -rf ~/face_tracking_venv
python3 -m venv ~/face_tracking_venv
```

### ROS2와 가상환경 충돌

```bash
# ROS2 환경을 먼저 소스한 후 가상환경 활성화
source /opt/ros/humble/setup.bash
source ~/face_tracking_venv/bin/activate
```

