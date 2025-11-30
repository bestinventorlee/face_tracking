# 얼굴 트래킹 패키지 빠른 설치 가이드

## 오류 해결: "Package 'face_tracking' not found"

이 오류는 패키지가 워크스페이스에 빌드되지 않았거나 환경이 소스되지 않았을 때 발생합니다.

## 해결 방법

### 1단계: 워크스페이스 확인

```bash
# 현재 워크스페이스 확인
echo $COLCON_PREFIX_PATH

# 워크스페이스가 없으면 생성 (microros_ws 사용)
mkdir -p /microros_ws/src
cd /microros_ws/src
```

### 2단계: 패키지 복사

```bash
# face_tracking 패키지를 워크스페이스 src로 복사
# 현재 face_tracking 디렉토리가 있는 위치에서
cp -r face_tracking /microros_ws/src/

# 또는 심볼릭 링크 (개발 중인 경우)
ln -s /path/to/face_tracking /microros_ws/src/face_tracking
```

### 3단계: 의존성 설치

```bash
# ROS2 환경 설정
source /opt/ros/jazzy/setup.bash  # 또는 사용 중인 ROS2 버전

# 가상환경 생성 (PEP 668 오류 방지)
python3 -m venv ~/face_tracking_venv
source ~/face_tracking_venv/bin/activate

# Python 패키지 설치
pip install --upgrade pip
pip install opencv-python numpy scipy rclpy
```

### 4단계: 패키지 빌드

```bash
cd /microros_ws

# ROS2 환경 설정
source /opt/ros/jazzy/setup.bash

# 패키지 빌드
colcon build --packages-select face_tracking

# 빌드 성공 확인
# "Summary: X packages finished" 메시지 확인
```

### 5단계: 환경 소스

```bash
# 워크스페이스 환경 소스 (매번 필요)
source /microros_ws/install/setup.bash

# 또는 .bashrc에 추가 (영구적)
echo "source /microros_ws/install/setup.bash" >> ~/.bashrc
```

### 6단계: 실행

```bash
# ROS2 환경 + 워크스페이스 환경 소스
source /opt/ros/jazzy/setup.bash
source /microros_ws/install/setup.bash

# 가상환경 활성화 (pip로 설치한 경우)
source ~/face_tracking_venv/bin/activate

# Launch 파일 실행
ros2 launch face_tracking face_tracking.launch.py
```

## 전체 설치 스크립트

```bash
#!/bin/bash
# 얼굴 트래킹 패키지 전체 설치

set -e  # 오류 발생 시 중단

echo "=== 얼굴 트래킹 패키지 설치 ==="

# 1. 워크스페이스 준비
WORKSPACE=/microros_ws
mkdir -p $WORKSPACE/src
cd $WORKSPACE/src

# 2. 패키지 복사 (현재 디렉토리에 face_tracking이 있다고 가정)
if [ ! -d "face_tracking" ]; then
    echo "패키지 디렉토리 경로를 입력하세요:"
    read PACKAGE_PATH
    cp -r "$PACKAGE_PATH" .
fi

# 3. ROS2 환경 설정
source /opt/ros/jazzy/setup.bash  # 또는 사용 중인 버전

# 4. 가상환경 생성 및 패키지 설치
if [ ! -d ~/face_tracking_venv ]; then
    python3 -m venv ~/face_tracking_venv
fi
source ~/face_tracking_venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy scipy rclpy

# 5. 패키지 빌드
cd $WORKSPACE
colcon build --packages-select face_tracking

# 6. 환경 소스
source install/setup.bash

echo ""
echo "=== 설치 완료 ==="
echo ""
echo "다음 명령으로 실행하세요:"
echo "  source /opt/ros/jazzy/setup.bash"
echo "  source /microros_ws/install/setup.bash"
echo "  source ~/face_tracking_venv/bin/activate"
echo "  ros2 launch face_tracking face_tracking.launch.py"

## 설치 확인

```bash
# 패키지 인식 확인
source ~/ros2_ws/install/setup.bash
ros2 pkg list | grep face_tracking

# Launch 파일 확인
ros2 launch face_tracking

# 실행 파일 확인
ros2 run face_tracking face_tracking_node.py --help
```

## 문제 해결

### 문제 1: 여전히 패키지를 찾을 수 없음

```bash
# 워크스페이스 환경이 제대로 소스되었는지 확인
echo $COLCON_PREFIX_PATH

# 수동으로 소스
source /microros_ws/install/setup.bash

# 패키지 경로 확인
ros2 pkg prefix face_tracking
```

### 문제 2: 빌드 오류

```bash
# 의존성 재설치
cd /microros_ws
rosdep install --from-paths src --ignore-src -r -y

# 깨끗한 빌드
rm -rf build install log
colcon build --packages-select face_tracking
```

### 문제 3: 실행 시 모듈을 찾을 수 없음

```bash
# 가상환경이 활성화되었는지 확인
which python
# 출력: ~/face_tracking_venv/bin/python 이어야 함

# 가상환경 활성화
source ~/face_tracking_venv/bin/activate
```

## 편의 스크립트 생성

매번 환경을 소스하는 것이 번거롭다면:

```bash
# ~/.bashrc에 추가
cat >> ~/.bashrc << 'EOF'

# Face Tracking 환경 설정
alias ft_env='source /opt/ros/jazzy/setup.bash && source /microros_ws/install/setup.bash && source ~/face_tracking_venv/bin/activate'
EOF

source ~/.bashrc

# 사용법
ft_env
ros2 launch face_tracking face_tracking.launch.py
```

