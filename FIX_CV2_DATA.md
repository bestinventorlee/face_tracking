# OpenCV cv2.data 오류 해결

## 문제: `module 'cv2' has no attribute 'data'`

이 오류는 OpenCV 설치 방법에 따라 `cv2.data` 속성이 없을 때 발생합니다.

## 해결 방법

### 방법 1: OpenCV 데이터 패키지 설치 (권장) ⭐

```bash
# OpenCV Haar Cascade 파일 설치
sudo apt update
sudo apt install -y opencv-data

# 재빌드
cd ~/microros_ws
colcon build --packages-select face_tracking

# 환경 소스 및 실행
source install/setup.bash
ros2 launch face_tracking face_tracking.launch.py
```

### 방법 2: 코드 수정 (이미 완료)

코드가 이미 수정되어 여러 경로에서 Haar Cascade 파일을 찾도록 개선되었습니다.

```bash
# 재빌드만 하면 됩니다
cd ~/microros_ws
colcon build --packages-select face_tracking
source install/setup.bash
ros2 launch face_tracking face_tracking.launch.py
```

### 방법 3: 수동으로 파일 다운로드

```bash
# Haar Cascade 파일 다운로드
mkdir -p ~/.local/share/opencv/haarcascades
cd ~/.local/share/opencv/haarcascades
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

# 코드에서 이 경로를 사용하도록 수정 필요
```

## 확인 방법

```bash
# OpenCV 데이터 파일 확인
ls /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml
# 또는
ls /usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml

# Python에서 확인
python3 -c "import cv2; print(cv2.__file__)"
```

## 빠른 해결 (한 줄)

```bash
sudo apt install -y opencv-data && cd ~/microros_ws && colcon build --packages-select face_tracking && source install/setup.bash && ros2 launch face_tracking face_tracking.launch.py
```

