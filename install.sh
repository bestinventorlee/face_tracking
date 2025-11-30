#!/bin/bash
# 얼굴 트래킹 패키지 설치 스크립트

echo "=========================================="
echo "  얼굴 트래킹 패키지 설치"
echo "=========================================="
echo ""

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 현재 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo -e "${YELLOW}패키지 디렉토리: ${SCRIPT_DIR}${NC}"
echo ""

# ROS2 워크스페이스 확인
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}ROS2 환경을 찾는 중...${NC}"
    if [ -f /opt/ros/humble/setup.bash ]; then
        source /opt/ros/humble/setup.bash
        echo -e "${GREEN}✓ ROS2 Humble 환경 설정됨${NC}"
    elif [ -f /opt/ros/iron/setup.bash ]; then
        source /opt/ros/iron/setup.bash
        echo -e "${GREEN}✓ ROS2 Iron 환경 설정됨${NC}"
    else
        echo "❌ ROS2 환경을 찾을 수 없습니다."
        echo "다음 명령으로 ROS2 환경을 설정하세요:"
        echo "  source /opt/ros/humble/setup.bash"
        exit 1
    fi
fi

# 워크스페이스 확인
if [ -z "$COLCON_PREFIX_PATH" ]; then
    echo -e "${YELLOW}워크스페이스 경로를 입력하세요 (기본값: ~/ros2_ws):${NC}"
    read -p "워크스페이스: " WORKSPACE
    WORKSPACE=${WORKSPACE:-~/ros2_ws}
    
    if [ ! -d "$WORKSPACE/src" ]; then
        echo "워크스페이스 src 디렉토리를 생성합니다..."
        mkdir -p "$WORKSPACE/src"
    fi
else
    WORKSPACE=$(dirname $(dirname $COLCON_PREFIX_PATH))
    echo -e "${GREEN}✓ 워크스페이스 감지: ${WORKSPACE}${NC}"
fi

# 패키지 복사
echo ""
echo -e "${YELLOW}[1/4] 패키지를 워크스페이스로 복사 중...${NC}"
if [ -d "$WORKSPACE/src/face_tracking" ]; then
    echo "기존 패키지가 있습니다. 덮어쓰시겠습니까? (y/n)"
    read -p "답변: " OVERWRITE
    if [ "$OVERWRITE" = "y" ] || [ "$OVERWRITE" = "Y" ]; then
        rm -rf "$WORKSPACE/src/face_tracking"
        cp -r "$SCRIPT_DIR" "$WORKSPACE/src/face_tracking"
        echo -e "${GREEN}✓ 패키지 복사 완료${NC}"
    else
        echo "기존 패키지를 유지합니다."
    fi
else
    cp -r "$SCRIPT_DIR" "$WORKSPACE/src/face_tracking"
    echo -e "${GREEN}✓ 패키지 복사 완료${NC}"
fi

# 의존성 설치
echo ""
echo -e "${YELLOW}[2/4] Python 의존성 설치 중...${NC}"
if [ -f "$SCRIPT_DIR/requirements_face_tracking.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements_face_tracking.txt"
    echo -e "${GREEN}✓ 의존성 설치 완료${NC}"
else
    echo "⚠️ requirements_face_tracking.txt를 찾을 수 없습니다."
    echo "수동으로 설치하세요: pip install opencv-python numpy scipy rclpy"
fi

# 패키지 빌드
echo ""
echo -e "${YELLOW}[3/4] 패키지 빌드 중...${NC}"
cd "$WORKSPACE"
colcon build --packages-select face_tracking
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 빌드 완료${NC}"
else
    echo "❌ 빌드 실패"
    exit 1
fi

# 환경 설정
echo ""
echo -e "${YELLOW}[4/4] 환경 설정...${NC}"
source "$WORKSPACE/install/setup.bash"
echo -e "${GREEN}✓ 환경 설정 완료${NC}"

echo ""
echo "=========================================="
echo "  설치 완료!"
echo "=========================================="
echo ""
echo "실행 방법:"
echo "  ros2 launch face_tracking face_tracking.launch.py"
echo ""
echo "또는:"
echo "  ros2 run face_tracking face_tracking_node.py"
echo ""

