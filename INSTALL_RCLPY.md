# rclpy 설치 가이드

`rclpy`는 ROS2의 Python 클라이언트 라이브러리입니다. 일반 apt 패키지가 아닙니다.

## rclpy 설치 방법

### 방법 1: ROS2 설치 시 자동 포함 (가장 일반적)

ROS2를 설치하면 `rclpy`가 자동으로 포함됩니다:

```bash
# ROS2 Humble 설치 (Ubuntu 22.04)
sudo apt install -y ros-humble-desktop

# 또는 ROS2 Iron 설치 (Ubuntu 24.04)
sudo apt install -y ros-iron-desktop
```

설치 확인:
```bash
source /opt/ros/humble/setup.bash  # 또는 iron
python3 -c "import rclpy; print(rclpy.__version__)"
```

### 방법 2: pip로 설치 (ROS2 미설치 시)

```bash
# 가상환경 사용 (권장)
source ~/face_tracking_venv/bin/activate
pip install rclpy>=3.0.0

# 또는 --break-system-packages 사용 (비권장)
pip3 install --break-system-packages --user rclpy>=3.0.0
```

### 방법 3: ROS2 패키지로만 설치

```bash
# ROS2 저장소 추가 (이미 ROS2가 설치되어 있다면 불필요)
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
sudo apt update

# rclpy만 설치 (ROS2 전체 설치 대신)
sudo apt install -y ros-humble-rclpy  # 또는 ros-iron-rclpy
```

## 현재 상황 확인

ROS2가 이미 설치되어 있는지 확인:

```bash
# ROS2 설치 확인
ls /opt/ros/

# rclpy 확인
source /opt/ros/humble/setup.bash  # 또는 iron
python3 -c "import rclpy; print('rclpy OK')"
```

## 권장 해결 방법

### ROS2가 이미 설치되어 있는 경우

```bash
# ROS2 환경만 소스하면 됨
source /opt/ros/humble/setup.bash  # 또는 iron

# 확인
python3 -c "import rclpy; print(rclpy.__version__)"
```

### ROS2가 설치되지 않은 경우

**옵션 A: ROS2 전체 설치 (권장)**
```bash
# ROS2 Humble 설치
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
sudo apt update
sudo apt install -y ros-humble-desktop
source /opt/ros/humble/setup.bash
```

**옵션 B: pip로만 설치 (제한적)**
```bash
# 가상환경 사용
python3 -m venv ~/face_tracking_venv
source ~/face_tracking_venv/bin/activate
pip install rclpy>=3.0.0
```

## 전체 설치 순서 (ROS2 미설치 시)

```bash
# 1. ROS2 설치
# (위의 ROS2 전체 설치 명령 실행)

# 2. ROS2 환경 설정
source /opt/ros/humble/setup.bash

# 3. 가상환경 생성 (선택적, PEP 668 오류 방지)
python3 -m venv ~/face_tracking_venv
source ~/face_tracking_venv/bin/activate

# 4. 나머지 패키지 설치
pip install opencv-python>=4.5.0 numpy>=1.19.0 scipy>=1.7.0

# 5. 확인
python3 -c "import rclpy, cv2, numpy, scipy; print('All OK')"
```

## 문제 해결

### rclpy를 찾을 수 없음

```bash
# ROS2 환경이 소스되었는지 확인
echo $ROS_DISTRO

# ROS2 환경 소스
source /opt/ros/humble/setup.bash

# Python 경로 확인
python3 -c "import sys; print(sys.path)"
```

### 가상환경에서 rclpy 사용

```bash
# 가상환경 활성화
source ~/face_tracking_venv/bin/activate

# ROS2 환경 소스 (가상환경 활성화 후)
source /opt/ros/humble/setup.bash

# rclpy 확인
python3 -c "import rclpy; print(rclpy.__version__)"
```

