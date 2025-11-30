# dos2unix 없이 줄바꿈 문제 해결

## 방법 1: sed 사용 (가장 간단) ⭐

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

# 5. 환경 소스
source install/setup.bash

# 6. 실행
ros2 launch face_tracking face_tracking.launch.py
```

## 방법 2: tr 사용

```bash
cd ~/microros_ws/src/face_tracking/scripts
tr -d '\r' < face_tracking_node.py > face_tracking_node.py.tmp
mv face_tracking_node.py.tmp face_tracking_node.py

cd ~/microros_ws/src/face_tracking/launch
tr -d '\r' < face_tracking.launch.py > face_tracking.launch.py.tmp
mv face_tracking.launch.py.tmp face_tracking.launch.py

chmod +x ~/microros_ws/src/face_tracking/scripts/face_tracking_node.py
cd ~/microros_ws
colcon build --packages-select face_tracking
source install/setup.bash
ros2 launch face_tracking face_tracking.launch.py
```

## 방법 3: 직접 Python으로 실행 (가장 확실) ⭐⭐⭐

줄바꿈 문제를 완전히 우회하는 방법:

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

이 방법은 shebang 라인을 우회하므로 줄바꿈 문제가 없습니다!

## 한 줄 명령어 (sed 사용)

```bash
cd ~/microros_ws/src/face_tracking && sed -i 's/\r$//' scripts/face_tracking_node.py launch/face_tracking.launch.py && chmod +x scripts/face_tracking_node.py && cd ~/microros_ws && colcon build --packages-select face_tracking && source install/setup.bash && ros2 launch face_tracking face_tracking.launch.py
```

## 확인 방법

```bash
# 줄바꿈 형식 확인
file ~/microros_ws/src/face_tracking/scripts/face_tracking_node.py

# 정상: "Python script, ASCII text executable"
# 문제: "Python script, ASCII text, with CRLF line terminators"

# 첫 줄 확인 (CR 문자 확인)
head -1 ~/microros_ws/src/face_tracking/scripts/face_tracking_node.py | od -c
# \r이 있으면 문제, 없으면 정상
```

