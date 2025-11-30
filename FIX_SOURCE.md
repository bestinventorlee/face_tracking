# 패키지를 찾을 수 없는 문제 해결

## 문제: 빌드는 성공했지만 패키지를 찾을 수 없음

빌드는 성공했지만 (`Summary: 1 package finished`), 패키지를 찾지 못하는 경우는 **환경 소스 경로가 잘못되었기 때문**입니다.

## 해결 방법

### 올바른 소스 명령어

```bash
# 현재 워크스페이스 디렉토리에서 (~/microros_ws)
cd ~/microros_ws

# 올바른 경로로 소스 (상대 경로 사용)
source install/setup.bash

# 또는 절대 경로 사용
source ~/microros_ws/install/setup.bash
```

### ❌ 잘못된 명령어

```bash
# 이렇게 하면 안 됩니다!
source ~/install/setup.bash  # /home/burublee/install/setup.bash를 찾음 (존재하지 않음)
source ./setup.bash          # 워크스페이스 루트에 setup.bash가 없음
```

### 확인 방법

```bash
# 1. 워크스페이스로 이동
cd ~/microros_ws

# 2. install 디렉토리 확인
ls install/face_tracking

# 3. 환경 소스
source install/setup.bash

# 4. 패키지 확인
ros2 pkg list | grep face_tracking

# 5. 패키지 경로 확인
ros2 pkg prefix face_tracking
# 출력: /home/burublee/microros_ws/install/face_tracking

# 6. Launch 파일 확인
ros2 launch face_tracking
```

## 전체 실행 순서

```bash
# 1. 워크스페이스로 이동
cd ~/microros_ws

# 2. ROS2 환경 설정
source /opt/ros/jazzy/setup.bash

# 3. 워크스페이스 환경 소스 (매우 중요!)
source install/setup.bash

# 4. 가상환경 활성화 (Python 패키지 설치한 경우)
source ~/face_tracking_venv/bin/activate

# 5. 패키지 확인
ros2 pkg list | grep face_tracking

# 6. 실행
ros2 launch face_tracking face_tracking.launch.py
```

## 영구 설정 (선택사항)

매번 소스하는 것이 번거롭다면 `.bashrc`에 추가:

```bash
# ~/.bashrc에 추가
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source ~/microros_ws/install/setup.bash" >> ~/.bashrc

# 적용
source ~/.bashrc
```

## 문제 해결 체크리스트

1. ✅ 빌드 성공 확인: `colcon build --packages-select face_tracking` 성공
2. ✅ install 디렉토리 확인: `ls ~/microros_ws/install/face_tracking` 존재 확인
3. ✅ 올바른 경로로 소스: `source ~/microros_ws/install/setup.bash`
4. ✅ 패키지 인식 확인: `ros2 pkg list | grep face_tracking`
5. ✅ Launch 파일 확인: `ros2 launch face_tracking`

## 디버깅 명령어

```bash
# 환경 변수 확인
echo $COLCON_PREFIX_PATH
# 출력에 /home/burublee/microros_ws/install 이 포함되어야 함

# AMENT_PREFIX_PATH 확인
echo $AMENT_PREFIX_PATH
# 출력에 /home/burublee/microros_ws/install 이 포함되어야 함

# 패키지 검색 경로 확인
ros2 pkg list --all | grep face_tracking

# 패키지 정보 확인
ros2 pkg prefix face_tracking
```

