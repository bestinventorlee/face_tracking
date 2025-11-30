# 얼굴 트래킹 패키지 설치 단계별 가이드

## 오류: "Package 'face_tracking' not found"

이 오류는 패키지가 ROS2 워크스페이스에 빌드되지 않았을 때 발생합니다.

## 해결 방법 (단계별)

### 1단계: 워크스페이스 확인 및 생성

```bash
# 워크스페이스가 있는지 확인
ls /microros_ws 2>/dev/null || echo "워크스페이스 없음"

# 워크스페이스가 없으면 생성
mkdir -p /microros_ws/src
cd /microros_ws/src
```

### 2단계: 패키지 복사

**옵션 A: Windows에서 개발한 경우 (SCP 또는 공유 폴더 사용)**

```bash
# Windows의 face_tracking 폴더를 Ubuntu로 복사
# 예: SCP 사용
scp -r user@windows_ip:/path/to/face_tracking /microros_ws/src/

# 또는 공유 폴더가 있다면
cp -r /mnt/shared/face_tracking /microros_ws/src/
```

**옵션 B: 이미 Ubuntu에 있는 경우**

```bash
# 현재 위치에서 워크스페이스로 복사
cp -r /path/to/face_tracking /microros_ws/src/

# 또는 심볼릭 링크 (개발 중인 경우)
ln -s /path/to/face_tracking /microros_ws/src/face_tracking
```

**옵션 C: 직접 생성 (패키지 파일이 있는 경우)**

```bash
cd ~/ros2_ws/src
mkdir -p face_tracking/scripts face_tracking/launch

# 파일들을 복사
# (Windows에서 파일을 Ubuntu로 전송한 후)
```

### 3단계: 패키지 구조 확인

```bash
cd /microros_ws/src/face_tracking
ls -la

# 다음 파일들이 있어야 합니다:
# - CMakeLists.txt
# - package.xml
# - scripts/face_tracking_node.py
# - launch/face_tracking.launch.py
# - requirements_face_tracking.txt
```

### 4단계: Python 의존성 설치

```bash
# 가상환경 생성 (권장)
python3 -m venv ~/face_tracking_venv
source ~/face_tracking_venv/bin/activate

# pip 업그레이드
pip install --upgrade pip

# 패키지 설치
pip install opencv-python numpy scipy

# rclpy는 ROS2와 함께 설치되어 있어야 합니다
# 확인:
python3 -c "import rclpy; print('rclpy OK')"
```

**또는 시스템 패키지로 설치:**

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-scipy
```

### 5단계: ROS2 환경 설정

```bash
# ROS2 환경 소스 (Jazzy 사용 중)
source /opt/ros/jazzy/setup.bash

# 또는 다른 버전:
# source /opt/ros/humble/setup.bash
# source /opt/ros/foxy/setup.bash
```

### 6단계: 패키지 빌드

```bash
cd /microros_ws

# 패키지 빌드
colcon build --packages-select face_tracking

# 빌드 성공 확인
# "Summary: 1 packages finished" 메시지 확인
```

**빌드 오류가 발생하면:**

```bash
cd /microros_ws

# 의존성 확인
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# 깨끗한 빌드
rm -rf build install log
colcon build --packages-select face_tracking
```

### 7단계: 워크스페이스 환경 소스

```bash
# 빌드 후 반드시 소스
source /microros_ws/install/setup.bash

# 영구적으로 설정하려면 .bashrc에 추가
echo "source /microros_ws/install/setup.bash" >> ~/.bashrc
```

### 8단계: 패키지 확인

```bash
# 패키지가 인식되는지 확인
ros2 pkg list | grep face_tracking

# Launch 파일 확인
ros2 launch face_tracking

# 실행 파일 확인
ros2 run face_tracking face_tracking_node.py --help
```

### 9단계: 실행

```bash
# 터미널 1: ROS2 환경 설정
source /opt/ros/jazzy/setup.bash
source /microros_ws/install/setup.bash

# 가상환경 활성화 (pip로 설치한 경우)
source ~/face_tracking_venv/bin/activate

# Launch 파일 실행
ros2 launch face_tracking face_tracking.launch.py
```

## 전체 설치 스크립트 (한 번에 실행)

```bash
#!/bin/bash
set -e

echo "=== 얼굴 트래킹 패키지 설치 ==="

# 1. 워크스페이스 준비
WORKSPACE=/microros_ws
mkdir -p $WORKSPACE/src
cd $WORKSPACE/src

# 2. 패키지 경로 확인
if [ ! -d "face_tracking" ]; then
    echo "패키지 디렉토리 경로를 입력하세요 (예: /path/to/face_tracking):"
    read PACKAGE_PATH
    if [ -d "$PACKAGE_PATH" ]; then
        cp -r "$PACKAGE_PATH" .
    else
        echo "오류: 패키지 디렉토리를 찾을 수 없습니다: $PACKAGE_PATH"
        exit 1
    fi
fi

# 3. ROS2 환경 설정
source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash

# 4. 가상환경 생성 및 패키지 설치
if [ ! -d ~/face_tracking_venv ]; then
    python3 -m venv ~/face_tracking_venv
fi
source ~/face_tracking_venv/bin/activate
pip install --upgrade pip --quiet
pip install opencv-python numpy scipy --quiet

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

## 문제 해결

### 문제 1: "Package 'face_tracking' not found" 지속

```bash
# 워크스페이스 환경 확인
echo $COLCON_PREFIX_PATH
# 출력에 /microros_ws/install 이 포함되어야 함

# 수동으로 소스
source /microros_ws/install/setup.bash

# 패키지 경로 확인
ros2 pkg prefix face_tracking
```

### 문제 2: "No module named 'rclpy'"

```bash
# ROS2가 제대로 설치되었는지 확인
which ros2
ros2 --version

# rclpy 확인
python3 -c "import rclpy; print(rclpy.__file__)"

# ROS2 재설치 필요할 수 있음
```

### 문제 3: "No module named 'cv2'"

```bash
# 가상환경 활성화 확인
which python
# 출력: ~/face_tracking_venv/bin/python 이어야 함

# 가상환경 활성화 후 재설치
source ~/face_tracking_venv/bin/activate
pip install opencv-python
```

### 문제 4: 빌드 실패

```bash
# CMakeLists.txt와 package.xml 확인
cd /microros_ws/src/face_tracking
cat CMakeLists.txt
cat package.xml

# 의존성 재설치
cd /microros_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# 깨끗한 빌드
rm -rf build install log
colcon build --packages-select face_tracking --cmake-args -DCMAKE_VERBOSE_MAKEFILE=ON
```

## 편의 스크립트 생성

매번 환경을 소스하는 것이 번거롭다면:

```bash
# ~/.bashrc에 추가
cat >> ~/.bashrc << 'EOF'

# Face Tracking 환경 설정
alias ft_env='source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash && source ~/face_tracking_venv/bin/activate'
EOF

source ~/.bashrc

# 사용법
ft_env
ros2 launch face_tracking face_tracking.launch.py
```

## 빠른 설치 명령어 (microros_ws 사용)

```bash
# 1. 워크스페이스로 이동
cd /microros_ws/src

# 2. 패키지 복사 (face_tracking 폴더를 여기로 복사)
# cp -r /path/to/face_tracking .

# 3. ROS2 환경 설정
source /opt/ros/jazzy/setup.bash

# 4. Python 의존성 설치
python3 -m venv ~/face_tracking_venv
source ~/face_tracking_venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy scipy

# 5. 패키지 빌드
cd /microros_ws
colcon build --packages-select face_tracking

# 6. 환경 소스
source /microros_ws/install/setup.bash

# 7. 실행
source ~/face_tracking_venv/bin/activate
ros2 launch face_tracking face_tracking.launch.py
```

