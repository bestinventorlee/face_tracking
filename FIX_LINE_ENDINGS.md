# 줄바꿈 문제 해결 가이드

## 문제 1: `/usr/bin/env: 'python3\r': No such file or directory`

이 오류는 Windows에서 생성된 파일의 줄바꿈 문자(`\r\n`)가 Linux에서 문제를 일으키는 것입니다.

## 해결 방법

### 방법 1: dos2unix로 줄바꿈 변환 (권장)

```bash
# 1. dos2unix 설치
sudo apt install dos2unix

# 2. 소스 파일 줄바꿈 변환
cd ~/microros_ws/src/face_tracking/scripts
dos2unix face_tracking_node.py

# 3. Launch 파일도 변환
cd ~/microros_ws/src/face_tracking/launch
dos2unix face_tracking.launch.py

# 4. 실행 권한 추가
chmod +x ~/microros_ws/src/face_tracking/scripts/face_tracking_node.py

# 5. 재빌드
cd ~/microros_ws
colcon build --packages-select face_tracking

# 6. 환경 소스
source install/setup.bash

# 7. 실행
ros2 launch face_tracking face_tracking.launch.py
```

### 방법 2: sed로 줄바꿈 변환 (dos2unix 없이)

```bash
# 1. 소스 파일 줄바꿈 변환
cd ~/microros_ws/src/face_tracking/scripts
sed -i 's/\r$//' face_tracking_node.py

# 2. Launch 파일도 변환
cd ~/microros_ws/src/face_tracking/launch
sed -i 's/\r$//' face_tracking.launch.py

# 3. 실행 권한 추가
chmod +x ~/microros_ws/src/face_tracking/scripts/face_tracking_node.py

# 4. 재빌드
cd ~/microros_ws
colcon build --packages-select face_tracking

# 5. 환경 소스 및 실행
source install/setup.bash
ros2 launch face_tracking face_tracking.launch.py
```

### 방법 3: 직접 Python으로 실행 (가장 확실)

```bash
# 1. 환경 소스
cd ~/microros_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# 2. 가상환경 활성화 (Python 패키지 설치한 경우)
source ~/face_tracking_venv/bin/activate

# 3. 직접 Python으로 실행
python3 src/face_tracking/scripts/face_tracking_node.py
```

## 문제 2: Launch 파일 이름

Launch 파일 이름은 `face_tracking.launch.py`입니다 (`.launch`가 아님).

```bash
# 올바른 명령어
ros2 launch face_tracking face_tracking.launch.py

# 잘못된 명령어
ros2 launch face_tracking face_tracking.launch  # ❌
```

## 한 번에 해결하는 스크립트

```bash
#!/bin/bash
# face_tracking 패키지 줄바꿈 문제 해결

cd ~/microros_ws/src/face_tracking

# dos2unix 설치 확인
if ! command -v dos2unix &> /dev/null; then
    echo "dos2unix 설치 중..."
    sudo apt install -y dos2unix
fi

# 줄바꿈 변환
echo "줄바꿈 형식 변환 중..."
dos2unix scripts/face_tracking_node.py 2>/dev/null
dos2unix launch/face_tracking.launch.py 2>/dev/null

# 실행 권한 추가
chmod +x scripts/face_tracking_node.py

# 재빌드
echo "패키지 재빌드 중..."
cd ~/microros_ws
colcon build --packages-select face_tracking

# 환경 소스
source install/setup.bash

echo ""
echo "✅ 완료!"
echo ""
echo "실행 명령어:"
echo "  ros2 launch face_tracking face_tracking.launch.py"
```

## 빠른 해결 (한 줄 명령어)

```bash
cd ~/microros_ws/src/face_tracking && \
sudo apt install -y dos2unix && \
dos2unix scripts/face_tracking_node.py launch/face_tracking.launch.py && \
chmod +x scripts/face_tracking_node.py && \
cd ~/microros_ws && \
colcon build --packages-select face_tracking && \
source install/setup.bash && \
ros2 launch face_tracking face_tracking.launch.py
```

## 확인 방법

```bash
# 줄바꿈 형식 확인
file ~/microros_ws/src/face_tracking/scripts/face_tracking_node.py

# 정상: "Python script, ASCII text executable"
# 문제: "Python script, ASCII text, with CRLF line terminators"

# 실행 권한 확인
ls -la ~/microros_ws/src/face_tracking/scripts/face_tracking_node.py
# 정상: -rwxr-xr-x (x가 있어야 함)
```

