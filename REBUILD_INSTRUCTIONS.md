# 재빌드 및 실행 가이드

## 문제: 코드 수정 후에도 같은 오류 발생

코드는 수정되었지만, Ubuntu에서 재빌드하지 않으면 변경사항이 반영되지 않습니다.

## 해결 방법

### 1단계: 수정된 파일을 Ubuntu로 복사

Windows에서 개발한 경우, 수정된 파일을 Ubuntu로 복사해야 합니다:

```bash
# Ubuntu에서
cd ~/microros_ws/src/face_tracking/scripts

# Windows에서 파일을 복사한 후 (SCP, 공유 폴더 등)
# 또는 직접 편집
```

### 2단계: 줄바꿈 변환 (Windows에서 복사한 경우)

```bash
cd ~/microros_ws/src/face_tracking
sed -i 's/\r$//' scripts/face_tracking_node.py
chmod +x scripts/face_tracking_node.py
```

### 3단계: OpenCV 데이터 설치

```bash
sudo apt install -y opencv-data
```

### 4단계: 재빌드

```bash
cd ~/microros_ws
colcon build --packages-select face_tracking
```

### 5단계: 환경 소스 및 실행

```bash
source install/setup.bash
ros2 launch face_tracking face_tracking.launch.py
```

## 빠른 해결 (한 줄)

```bash
sudo apt install -y opencv-data && cd ~/microros_ws/src/face_tracking && sed -i 's/\r$//' scripts/face_tracking_node.py && chmod +x scripts/face_tracking_node.py && cd ~/microros_ws && colcon build --packages-select face_tracking && source install/setup.bash && ros2 launch face_tracking face_tracking.launch.py
```

## 대안: 직접 Python으로 실행 (빌드 없이 테스트)

빌드 전에 코드가 제대로 작동하는지 테스트:

```bash
# 1. 환경 소스
cd ~/microros_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# 2. 가상환경 활성화
source ~/face_tracking_venv/bin/activate

# 3. 직접 실행 (수정된 소스 파일)
python3 src/face_tracking/scripts/face_tracking_node.py
```

이 방법으로 오류가 해결되었는지 확인한 후, 정상 작동하면 재빌드하세요.

## 파일 확인

수정된 코드가 Ubuntu에 있는지 확인:

```bash
# 얼굴 인식기 초기화 부분 확인
grep -A 5 "얼굴 인식기 초기화" ~/microros_ws/src/face_tracking/scripts/face_tracking_node.py

# 출력에 "cv2.data가 없는 경우를 대비한 fallback 처리"가 있어야 함
```

## 문제 해결 체크리스트

- [ ] 수정된 `face_tracking_node.py` 파일이 Ubuntu에 있음
- [ ] 줄바꿈 변환 완료 (`sed -i 's/\r$//'`)
- [ ] 실행 권한 추가 (`chmod +x`)
- [ ] `opencv-data` 패키지 설치 완료
- [ ] 재빌드 완료 (`colcon build`)
- [ ] 환경 소스 완료 (`source install/setup.bash`)

