#!/usr/bin/env python3
"""
얼굴 인식 및 트래킹 ROS2 노드
로봇 팔의 엔드이펙트에 장착된 카메라로 사람 얼굴을 인식하고,
얼굴의 움직임에 따라 카메라를 추적하도록 로봇을 제어합니다.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32
import cv2
import numpy as np
from typing import Optional, Tuple
import time
import math
from scipy.optimize import least_squares

class CobotKinematics:
    """6축 협동로봇 운동학 클래스 (Forward & Inverse Kinematics)"""
    def __init__(self):
        # DH 파라미터
        self.a = [0, 0.2805, 0.2495, 0, 0, 0]  # 링크 길이 (m)
        self.alpha = [-np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]  # 링크 비틀림 (rad)
        self.d = [0.235, 0, 0, 0.258, 0.180, 0.123]  # 조인트 오프셋 (m)
        self.n_joints = 6
        
        # 조인트 오프셋 정의
        self.joint_offsets = [-np.pi/2, -np.pi/2, 0, np.pi/2, np.pi/2, 0]
        
        # 조인트 회전 방향
        self.joint_direction = [1, 1, -1, 1, 1, 1]
        
        # 조인트 범위 (degree)
        self.joint_limits = np.array([[-135, 135]] * self.n_joints)
    
    def _apply_joint_offset(self, user_angles):
        """사용자 입력 각도에 오프셋 적용"""
        adjusted_angles = np.array(user_angles) * np.array(self.joint_direction)
        return adjusted_angles + np.array(self.joint_offsets)
    
    def dh_transform(self, a, alpha, d, theta):
        """DH 변환 행렬"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T = np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
        return T
    
    def forward_kinematics(self, user_joint_angles_deg):
        """순방향 운동학 (각도는 degree 단위 입력)"""
        # 각도를 라디안으로 변환
        user_joint_angles_rad = np.deg2rad(user_joint_angles_deg)
        dh_angles = self._apply_joint_offset(user_joint_angles_rad)
        
        T = np.eye(4)
        
        for i in range(self.n_joints):
            T_i = self.dh_transform(self.a[i], self.alpha[i], self.d[i], dh_angles[i])
            T = T @ T_i
        
        position = T[:3, 3]  # 미터 단위
        rotation_matrix = T[:3, :3]
        euler_angles = self.rotation_matrix_to_euler(rotation_matrix)  # 라디안
        
        return position, euler_angles
    
    def rotation_matrix_to_euler(self, R):
        """회전 행렬을 오일러 각도로 변환 (ZYX 순서)"""
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def inverse_kinematics(self, target_position, target_orientation, initial_guess=None):
        """
        역방향 운동학 (Inverse Kinematics)
        
        Args:
            target_position: 목표 위치 [x, y, z] (미터 단위)
            target_orientation: 목표 방향 [roll, pitch, yaw] (라디안 단위)
            initial_guess: 초기 추정 각도 (라디안 단위, None이면 자동 설정)
        
        Returns:
            (joint_angles_deg, success): (각도 배열 [degree], 성공 여부)
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.n_joints)
        else:
            # degree를 radian으로 변환
            if np.max(np.abs(initial_guess)) > 10:  # degree로 입력된 경우
                initial_guess = np.deg2rad(initial_guess)
        
        # 초기값을 조인트 범위 내로 클리핑
        initial_guess_deg = np.degrees(initial_guess)
        initial_guess_deg = np.clip(initial_guess_deg, 
                                   self.joint_limits[:, 0], 
                                   self.joint_limits[:, 1])
        initial_guess = np.radians(initial_guess_deg)
        
        def objective_function(user_joint_angles):
            # user_joint_angles는 라디안 단위
            # forward_kinematics는 degree를 받으므로 변환 필요
            user_joint_angles_deg = np.rad2deg(user_joint_angles)
            pos, euler = self.forward_kinematics(user_joint_angles_deg)
            
            # 위치 오차 (미터 단위)
            pos_error = target_position - pos
            
            # 방향 오차 (라디안 단위)
            orient_error = target_orientation - euler
            
            # 전체 오차 (위치 오차에 더 큰 가중치)
            error = np.concatenate([pos_error * 1000, orient_error])
            return error
        
        # 조인트 범위를 라디안으로 변환
        lower_bounds = np.radians(self.joint_limits[:, 0])
        upper_bounds = np.radians(self.joint_limits[:, 1])
        
        try:
            result = least_squares(objective_function, initial_guess, 
                                 bounds=(lower_bounds, upper_bounds),
                                 method='trf',
                                 ftol=1e-6, xtol=1e-6, max_nfev=1000)
            
            solution_rad = result.x
            solution_deg = np.rad2deg(solution_rad)
            
            # 해의 유효성 검증
            pos, euler = self.forward_kinematics(solution_deg)
            pos_error = np.linalg.norm(target_position - pos)
            orient_error = np.linalg.norm(target_orientation - euler)
            
            if pos_error < 0.001 and orient_error < 0.01:
                return solution_deg, True
            else:
                return solution_deg, False
        except Exception as e:
            # IK 실패 시 초기 추정값 반환
            initial_guess_deg = np.rad2deg(initial_guess)
            return initial_guess_deg, False


class FaceTrackingNode(Node):
    def __init__(self):
        super().__init__('face_tracking_node')
        
        # 운동학 객체 생성
        self.kinematics = CobotKinematics()
        
        # ROS2 퍼블리셔 생성 (로봇 제어 명령 전송)
        # servo_angles: 속도/가속도 없이 즉시 이동 (트래킹에 적합)
        # servo_angles_with_speed: 속도/가속도 포함 (부드러운 이동)
        self.angle_pub = self.create_publisher(
            Float32MultiArray,
            'servo_angles',  # 속도/가속도 없이 즉시 이동
            10
        )
        
        # 선택적: 부드러운 이동이 필요한 경우 사용
        self.angle_speed_pub = self.create_publisher(
            Float32MultiArray,
            'servo_angles_with_speed',
            10
        )
        
        # 디버깅용: 좌표 퍼블리셔 (선택적)
        self.coord_pub = self.create_publisher(
            Float32MultiArray,
            'robot_coords_with_speed',
            10
        )
        
        # ROS2 구독자 생성 (현재 서보 상태 수신)
        self.servo_status_sub = self.create_subscription(
            Float32MultiArray,
            'servo_status',
            self.servo_status_callback,
            10
        )
        
        # 서보 상태 요청 퍼블리셔 (선택적 - 마스터가 지원하는 경우)
        # Int32 메시지로 모든 서보 상태 요청 (data=0: 모든 서보, data=1~6: 특정 서보)
        self.request_status_pub = self.create_publisher(
            Int32,
            'request_servo_status',
            10
        )
        
        # 초기 상태 요청 (시작 시 한 번)
        self.request_initial_status = True
        self.last_status_request_time = 0.0
        self.status_request_interval = 1.0  # 1초마다 요청 (선택적)
        
        # 얼굴 인식기 초기화
        # cv2.data를 완전히 사용하지 않고 시스템/패키지 경로만 사용 (근본적 해결)
        import os
        
        # 가능한 경로 목록 (우선순위 순서)
        # cv2.data는 사용하지 않음 - 시스템 경로와 Python 패키지 경로만 사용
        possible_paths = [
            # 시스템 경로 (opencv-data 패키지 설치 시)
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
            '/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        ]
        
        # Python 패키지 경로 (opencv-python 또는 opencv-contrib-python 설치 시)
        try:
            cv2_path = os.path.dirname(cv2.__file__)
            pkg_path = os.path.join(cv2_path, 'data', 'haarcascade_frontalface_default.xml')
            possible_paths.append(pkg_path)
        except:
            pass
        
        # 첫 번째로 존재하는 경로 사용
        cascade_path = None
        for path in possible_paths:
            if os.path.exists(path):
                cascade_path = path
                break
        
        if cascade_path is None:
            raise RuntimeError(
                'Haar Cascade 파일을 찾을 수 없습니다.\n'
                '다음 중 하나를 실행하세요:\n'
                '  1. sudo apt install -y opencv-data\n'
                '  2. pip install opencv-python 또는 opencv-contrib-python'
            )
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f'Haar Cascade 파일을 로드할 수 없습니다: {cascade_path}')
        self.get_logger().info(f'얼굴 인식기 초기화 완료: {cascade_path}')
        
        # 카메라 초기화
        self._init_camera()
    
    def _find_available_cameras(self, max_test=10):
        """
        사용 가능한 카메라 인덱스를 찾습니다.
        외부 USB 카메라를 우선적으로 선택합니다.
        
        Args:
            max_test: 테스트할 최대 카메라 인덱스 수
        
        Returns:
            (사용 가능한 카메라 인덱스 리스트, USB 카메라 인덱스 리스트)
        """
        import os
        available = []
        usb_cameras = []
        
        # 먼저 모든 사용 가능한 카메라 찾기
        for idx in range(max_test):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # 실제로 프레임을 읽을 수 있는지 테스트
                ret, frame = cap.read()
                if ret and frame is not None:
                    available.append(idx)
                    # USB 카메라인지 확인 (/dev/video* 경로 확인)
                    video_device = f'/dev/video{idx}'
                    if os.path.exists(video_device):
                        # USB 카메라인지 확인 (udevadm 또는 sysfs 사용)
                        try:
                            import subprocess
                            result = subprocess.run(
                                ['udevadm', 'info', '--query=property', '--name', video_device],
                                capture_output=True, text=True, timeout=1
                            )
                            if result.returncode == 0:
                                # USB 관련 속성이 있으면 USB 카메라로 간주
                                if 'ID_USB' in result.stdout or 'ID_BUS=usb' in result.stdout:
                                    usb_cameras.append(idx)
                                    self.get_logger().debug(f'USB 카메라 {idx} 발견: {video_device}')
                        except:
                            # udevadm가 없거나 실패하면, 낮은 인덱스를 USB로 간주 (일반적으로 USB가 먼저 할당됨)
                            if idx < 2:  # 0, 1은 보통 USB 카메라
                                usb_cameras.append(idx)
                    
                    self.get_logger().debug(f'카메라 {idx} 발견: 해상도 {frame.shape[1]}x{frame.shape[0]}')
                cap.release()
            else:
                cap.release()
        
        # USB 카메라가 없으면 낮은 인덱스를 우선 (일반적으로 USB가 낮은 인덱스 사용)
        if not usb_cameras and len(available) > 0:
            # 인덱스 0, 1을 USB로 간주 (일반적인 경우)
            usb_cameras = [idx for idx in available if idx < 2]
        
        return available, usb_cameras
    
    def _init_camera(self):
        """카메라 초기화 - 외부 USB 카메라를 우선적으로 선택"""
        # 트래킹 파라미터 (ROS2 파라미터에서 로드) - 카메라 인덱스를 먼저 읽기 위해
        self.declare_parameter('camera_index', -1)  # 카메라 인덱스 (-1: 자동, 0 이상: 특정 인덱스)
        requested_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        
        # 사용 가능한 카메라 찾기 (USB 카메라 우선)
        available_cameras, usb_cameras = self._find_available_cameras()
        self.get_logger().info(f'사용 가능한 카메라: {available_cameras}')
        if usb_cameras:
            self.get_logger().info(f'USB 카메라: {usb_cameras}')
        
        # 카메라 초기화
        if requested_index == -1:
            # 자동 선택: USB 카메라 우선, 없으면 첫 번째 사용 가능한 카메라
            if usb_cameras:
                self.camera_index = usb_cameras[0]
                self.get_logger().info(f'자동으로 USB 카메라 {self.camera_index} 선택됨')
            elif len(available_cameras) > 0:
                self.camera_index = available_cameras[0]
                self.get_logger().info(f'USB 카메라가 없어 첫 번째 사용 가능한 카메라 {self.camera_index} 선택됨')
            else:
                raise RuntimeError('사용 가능한 카메라를 찾을 수 없습니다!')
        elif requested_index in available_cameras:
            # 요청한 인덱스가 사용 가능한 경우
            self.camera_index = requested_index
            camera_type = "USB" if requested_index in usb_cameras else "내장"
            self.get_logger().info(f'요청한 {camera_type} 카메라 {self.camera_index} 사용')
        else:
            # 요청한 인덱스가 사용 불가능한 경우, USB 카메라 우선으로 폴백
            if usb_cameras:
                self.camera_index = usb_cameras[0]
                self.get_logger().warn(
                    f'카메라 {requested_index}를 사용할 수 없습니다. '
                    f'대신 USB 카메라 {self.camera_index}를 사용합니다.'
                )
            elif len(available_cameras) > 0:
                self.camera_index = available_cameras[0]
                self.get_logger().warn(
                    f'카메라 {requested_index}를 사용할 수 없습니다. '
                    f'대신 카메라 {self.camera_index}를 사용합니다.'
                )
            else:
                raise RuntimeError(f'카메라 {requested_index}를 열 수 없고, 다른 카메라도 없습니다!')
        
        # 선택한 카메라로 초기화
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f'카메라 {self.camera_index} 초기화 실패!')
        
        # 카메라가 실제로 프레임을 제공하는지 테스트
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            raise RuntimeError(f'카메라 {self.camera_index}에서 프레임을 읽을 수 없습니다!')
        
        # 카메라 해상도 설정 (HD 해상도)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # 현재 엔드이펙트 위치 및 각도 (mm, degree)
        # servo_status 토픽이 실패할 수 있으므로 안전한 초기화 전략 사용
        self.current_position = np.array([0.0, 0.0, 0.0])  # x, y, z (mm)
        self.current_orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw (degree)
        self.current_angles = np.zeros(6)  # 서보 각도 (degree)
        self.has_position = False  # 실제 위치를 받을 때까지 False
        
        # 초기 위치 요청 타임아웃 (초) - 파라미터로 설정됨
        self.start_time = time.time()
        self.position_request_sent = False
        self.first_command_sent = False  # 첫 번째 명령 전송 여부
        
        # 얼굴 추적 변수
        self.face_center = None  # 화면상 얼굴 중심 좌표 (x, y)
        self.previous_face_center = None
        self.face_detected = False
        self.last_face_time = 0.0
        
        # 트래킹 파라미터 (ROS2 파라미터에서 로드)
        self.declare_parameter('tracking_speed', 30.0)
        self.declare_parameter('tracking_accel', 50.0)
        self.declare_parameter('max_movement', 50.0)
        self.declare_parameter('tracking_sensitivity', 0.5)
        self.declare_parameter('camera_fov_horizontal', 60.0)
        self.declare_parameter('camera_fov_vertical', 45.0)
        self.declare_parameter('estimated_face_distance', 1000.0)
        self.declare_parameter('movement_threshold', 5.0)  # 픽셀 단위
        self.declare_parameter('position_timeout', 5.0)  # 초기 위치 요청 타임아웃 (초)
        self.declare_parameter('safe_first_movement_limit', 10.0)  # 첫 번째 명령 최대 이동 거리 (mm)
        
        self.tracking_speed = self.get_parameter('tracking_speed').get_parameter_value().double_value
        self.tracking_accel = self.get_parameter('tracking_accel').get_parameter_value().double_value
        self.max_movement = self.get_parameter('max_movement').get_parameter_value().double_value
        self.tracking_sensitivity = self.get_parameter('tracking_sensitivity').get_parameter_value().double_value
        self.camera_fov_horizontal = self.get_parameter('camera_fov_horizontal').get_parameter_value().double_value
        self.camera_fov_vertical = self.get_parameter('camera_fov_vertical').get_parameter_value().double_value
        self.estimated_face_distance = self.get_parameter('estimated_face_distance').get_parameter_value().double_value
        self.movement_threshold = self.get_parameter('movement_threshold').get_parameter_value().double_value
        self.position_timeout = self.get_parameter('position_timeout').get_parameter_value().double_value
        self.safe_first_movement_limit = self.get_parameter('safe_first_movement_limit').get_parameter_value().double_value
        
        # 화면 중심 좌표 (카메라 해상도에 따라 동적으로 설정)
        ret, test_frame = self.cap.read()
        if ret:
            self.frame_height, self.frame_width = test_frame.shape[:2]
        else:
            self.frame_width = 640
            self.frame_height = 480
        self.screen_center = np.array([self.frame_width / 2, self.frame_height / 2])
        
        # ROS2 타이머 생성 (카메라 프레임 처리)
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30 FPS
        
        # 서보 상태 요청 타이머 (선택적 - 주기적으로 상태 요청)
        # 주의: 마스터가 request_servo_status 토픽을 구독하는 경우에만 동작
        # self.status_request_timer = self.create_timer(
        #     self.status_request_interval,
        #     self.request_servo_status
        # )
        
        self.get_logger().info('얼굴 트래킹 노드가 시작되었습니다.')
        self.get_logger().info(f'사용 중인 카메라: {self.camera_index}')
        self.get_logger().info(f'카메라 해상도: {self.frame_width}x{self.frame_height}')
        self.get_logger().info(f'트래킹 속도: {self.tracking_speed} deg/s')
        self.get_logger().info(f'트래킹 가속도: {self.tracking_accel} deg/s²')
        self.get_logger().info(f'최대 이동 거리: {self.max_movement} mm')
        self.get_logger().info(f'트래킹 민감도: {self.tracking_sensitivity}')
        
        # 초기 상태 요청 (시작 시) - 선택적 기능
        # 주의: request_servo_status가 실패해도 servo_status 토픽을 구독하므로 작동 가능
        if self.request_initial_status:
            time.sleep(0.5)  # ROS2 초기화 대기
            self.request_servo_status()  # 실패해도 경고만 출력하고 계속 진행
            self.request_initial_status = False
            self.position_request_sent = True
            self.get_logger().info(
                '초기 서보 상태 요청 전송. '
                f'{self.position_timeout}초 내에 응답이 없으면 안전한 위치로 가정합니다.'
            )
        
    def request_servo_status(self):
        """서보 상태 요청 (마스터가 지원하는 경우, data=0: 모든 서보 요청)"""
        if self.request_status_pub is not None:
            msg = Int32()
            msg.data = 0  # 0 = 모든 서보 요청
            self.request_status_pub.publish(msg)
            self.get_logger().debug('서보 상태 요청 전송 (모든 서보)')
    
    def servo_status_callback(self, msg):
        """서보 상태 메시지 수신 콜백"""
        try:
            # servo_status 메시지 형식: [servo_id, current_angle, target_angle, encoder_angle, max_speed] * 6
            if len(msg.data) >= 30:  # 6개 서보 * 5개 데이터
                angles = []
                for i in range(6):
                    base_idx = i * 5
                    current_angle = msg.data[base_idx + 1]  # current_angle
                    angles.append(current_angle)
                
                self.current_angles = np.array(angles)
                
                # Forward Kinematics로 현재 위치 계산
                try:
                    position_m, orientation_rad = self.kinematics.forward_kinematics(self.current_angles)
                    
                    # 미터를 밀리미터로 변환
                    self.current_position = position_m * 1000.0
                    
                    # 라디안을 도로 변환
                    self.current_orientation = np.rad2deg(orientation_rad)
                    
                    self.has_position = True
                    
                    self.get_logger().info(
                        f'서보 상태 수신: 현재 위치 업데이트 ({self.current_position[0]:.1f}, '
                        f'{self.current_position[1]:.1f}, {self.current_position[2]:.1f}) mm, '
                        f'각도: [{self.current_angles[0]:.1f}, {self.current_angles[1]:.1f}, '
                        f'{self.current_angles[2]:.1f}, {self.current_angles[3]:.1f}, '
                        f'{self.current_angles[4]:.1f}, {self.current_angles[5]:.1f}] deg'
                    )
                except Exception as fk_error:
                    self.get_logger().warn(f'FK 계산 오류: {fk_error}')
                    self.has_position = False
                
        except Exception as e:
            self.get_logger().error(f'서보 상태 처리 오류: {e}')
    
    def detect_face(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """프레임에서 얼굴 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # 가장 큰 얼굴 선택
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face
            return (x, y, w, h)
        
        return None
    
    def calculate_face_center(self, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """얼굴 사각형에서 중심점 계산"""
        x, y, w, h = face_rect
        center_x = x + w / 2
        center_y = y + h / 2
        return np.array([center_x, center_y])
    
    def screen_to_3d_offset(self, screen_offset: np.ndarray) -> np.ndarray:
        """
        화면상의 오프셋을 3D 공간의 오프셋으로 변환
        카메라 좌표계를 로봇 베이스 좌표계로 변환
        
        Args:
            screen_offset: 화면상 오프셋 [dx, dy] (픽셀)
        
        Returns:
            3D 공간 오프셋 [dx, dy, dz] (mm) - 로봇 베이스 좌표계
        """
        # 화면 좌표를 정규화된 좌표로 변환 (-1 ~ 1)
        normalized_x = (screen_offset[0] / (self.frame_width / 2))
        normalized_y = -(screen_offset[1] / (self.frame_height / 2))  # Y축 반전
        
        # 카메라 좌표계에서의 각도 계산
        angle_x = normalized_x * (self.camera_fov_horizontal / 2) * np.pi / 180.0
        angle_y = normalized_y * (self.camera_fov_vertical / 2) * np.pi / 180.0
        
        # 거리를 기반으로 3D 오프셋 계산
        # 카메라가 얼굴을 향하고 있다고 가정
        distance = self.estimated_face_distance
        
        # 카메라 좌표계에서의 오프셋 계산
        # 카메라 좌표계: X=우측, Y=아래, Z=전방
        dx_camera = distance * np.tan(angle_x)  # 좌우
        dy_camera = distance * np.tan(angle_y)  # 상하
        dz_camera = 0.0  # 전후 (얼굴 크기 변화 기반으로 추후 개선 가능)
        
        # 카메라 좌표계를 로봇 베이스 좌표계로 변환
        # 카메라 좌표계: X=우측, Y=아래, Z=전방
        # 로봇 베이스 좌표계: X=전방, Y=좌측, Z=위
        # 
        # 카메라가 엔드이펙트에 장착되어 있고, 전방을 향한다고 가정
        # 카메라 X(우측) → 로봇 Y(좌측/우측, 반대 방향)
        # 카메라 Y(아래) → 로봇 -Z(상하, 반대 방향)
        # 카메라 Z(전방) → 로봇 X(전후)
        
        # 기본 변환 (엔드이펙트 방향이 기본 자세라고 가정)
        dx_robot = dz_camera  # 전후 (카메라 전방 → 로봇 전방)
        dy_robot = -dx_camera  # 좌우 (카메라 우측 → 로봇 좌측, 반대 방향)
        dz_robot = -dy_camera  # 상하 (카메라 아래 → 로봇 위, 반대 방향)
        
        # TODO: 현재 엔드이펙트 방향(roll, pitch, yaw)을 고려한 정확한 변환
        # 현재는 기본 변환만 사용 (대부분의 경우 충분함)
        
        return np.array([dx_robot, dy_robot, dz_robot])
    
    def calculate_target_position(self, face_center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        얼굴 중심 좌표를 기반으로 목표 위치 계산
        
        Args:
            face_center: 화면상 얼굴 중심 좌표 [x, y]
        
        Returns:
            (target_position, target_orientation): 목표 위치(mm)와 방향(degree)
        """
        if not self.has_position:
            # 현재 위치를 모르면 초기 위치 사용
            self.current_position = np.array([0.0, 0.0, 500.0])  # 기본 Z 높이
            self.current_orientation = np.array([0.0, 0.0, 0.0])
        
        # 화면 중심과의 오프셋 계산
        screen_offset = face_center - self.screen_center
        
        # 화면 오프셋을 3D 공간 오프셋으로 변환 (mm 단위)
        offset_3d = self.screen_to_3d_offset(screen_offset)
        
        # 민감도 적용
        offset_3d *= self.tracking_sensitivity
        
        # 최대 이동 거리 제한
        offset_magnitude = np.linalg.norm(offset_3d)
        if offset_magnitude > self.max_movement:
            offset_3d = offset_3d / offset_magnitude * self.max_movement
        
        # 목표 위치 계산 (현재 위치 + 오프셋) - mm 단위
        target_position = self.current_position + offset_3d
        
        # 목표 방향 계산 (얼굴을 향하도록)
        # 현재는 기본 방향 유지 (추후 개선 가능)
        target_orientation = self.current_orientation.copy()
        
        return target_position, target_orientation
    
    def calculate_target_angles(self, target_position: np.ndarray, target_orientation: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        목표 위치와 방향을 각도로 변환 (IK 계산)
        
        Args:
            target_position: 목표 위치 [x, y, z] (mm 단위)
            target_orientation: 목표 방향 [roll, pitch, yaw] (degree 단위)
        
        Returns:
            (target_angles, success): (각도 배열 [degree], 성공 여부)
        """
        # mm를 미터로 변환
        target_pos_m = target_position / 1000.0
        
        # degree를 라디안으로 변환
        target_orient_rad = np.deg2rad(target_orientation)
        
        # 초기 추정값으로 현재 각도 사용 (라디안)
        initial_guess_rad = np.deg2rad(self.current_angles)
        
        # IK 계산
        target_angles_deg, success = self.kinematics.inverse_kinematics(
            target_pos_m,
            target_orient_rad,
            initial_guess=initial_guess_rad
        )
        
        return target_angles_deg, success
    
    def send_robot_command(self, target_angles: np.ndarray, success: bool):
        """
        로봇 제어 명령 전송 (각도 기반)
        
        Args:
            target_angles: 목표 각도 배열 [degree] (6개)
            success: IK 계산 성공 여부
        """
        if not success:
            self.get_logger().warn('IK 계산 실패로 명령을 전송하지 않습니다.')
            return
        
        # servo_angles 토픽 사용: 속도/가속도 없이 즉시 이동 (트래킹에 적합)
        msg = Float32MultiArray()
        # servo_angles 형식: [angle1~6] (6개만)
        msg.data = [
            float(target_angles[0]),  # 서보 1 각도 (degree)
            float(target_angles[1]),  # 서보 2 각도 (degree)
            float(target_angles[2]),  # 서보 3 각도 (degree)
            float(target_angles[3]),  # 서보 4 각도 (degree)
            float(target_angles[4]),  # 서보 5 각도 (degree)
            float(target_angles[5]),  # 서보 6 각도 (degree)
        ]
        
        self.angle_pub.publish(msg)
        self.get_logger().debug(
            f'로봇 명령 전송 (즉시 이동): [{target_angles[0]:.1f}, {target_angles[1]:.1f}, '
            f'{target_angles[2]:.1f}, {target_angles[3]:.1f}, {target_angles[4]:.1f}, '
            f'{target_angles[5]:.1f}]°'
        )
    
    def process_frame(self):
        """카메라 프레임 처리 및 얼굴 트래킹"""
        # 초기 위치 타임아웃 체크 (servo_status 토픽이 실패한 경우 대비)
        if not self.has_position and self.position_request_sent:
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.position_timeout:
                # 타임아웃: 안전한 중립 위치로 가정 (홈 포지션)
                # 주의: 실제 위치와 다를 수 있으므로 첫 번째 명령은 작은 이동만 수행
                try:
                    # 안전한 중립 위치: 모든 서보 0도 (홈 포지션)
                    # 하지만 첫 번째 명령은 상대적 이동만 수행하도록 제한
                    position_m, orientation_rad = self.kinematics.forward_kinematics(self.current_angles)
                    self.current_position = position_m * 1000.0
                    self.current_orientation = np.rad2deg(orientation_rad)
                    self.has_position = True
                    self.get_logger().warn(
                        f'⚠️ 서보 상태 수신 타임아웃 ({self.position_timeout}초). '
                        f'안전한 위치로 가정: ({self.current_position[0]:.1f}, '
                        f'{self.current_position[1]:.1f}, {self.current_position[2]:.1f}) mm'
                    )
                    self.get_logger().warn(
                        f'⚠️ 실제 위치와 다를 수 있으므로 첫 번째 명령은 '
                        f'{self.safe_first_movement_limit}mm 이하로만 이동합니다.'
                    )
                except Exception as e:
                    self.get_logger().error(f'초기 위치 계산 실패: {e}')
        
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('프레임을 읽을 수 없습니다.')
            return
        
        # 얼굴 검출
        face_rect = self.detect_face(frame)
        
        if face_rect is not None:
            # 얼굴 중심 계산
            self.face_center = self.calculate_face_center(face_rect)
            self.face_detected = True
            self.last_face_time = time.time()
            
            # 얼굴 사각형 그리기
            x, y, w, h = face_rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 얼굴 중심점 표시
            center_x, center_y = int(self.face_center[0]), int(self.face_center[1])
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # 화면 중심선 표시
            cv2.line(frame, 
                    (int(self.screen_center[0]), 0),
                    (int(self.screen_center[0]), self.frame_height),
                    (255, 0, 0), 1)
            cv2.line(frame, 
                    (0, int(self.screen_center[1])),
                    (self.frame_width, int(self.screen_center[1])),
                    (255, 0, 0), 1)
            
            # 얼굴이 처음 검출되었을 때 현재 위치 요청 (선택적)
            if self.previous_face_center is None:
                # 얼굴이 처음 검출되면 현재 서보 상태를 요청
                # 주의: request_servo_status가 실패해도 servo_status 토픽을 구독하므로 작동 가능
                if not self.has_position:
                    self.request_servo_status()
                    self.get_logger().info('얼굴 검출됨: 서보 상태 요청 전송 (현재 위치 없음)')
                else:
                    self.get_logger().info('얼굴 검출됨: 현재 위치 있음, 트래킹 시작')
            
            # 얼굴이 이전 프레임에서 이동했는지 확인
            if self.previous_face_center is not None:
                movement = np.linalg.norm(self.face_center - self.previous_face_center)
                
                # 움직임이 감지되면 로봇 제어 명령 전송
                if movement > self.movement_threshold:
                    if self.has_position:
                        # 1. 얼굴 이동량을 기반으로 목표 위치 계산
                        target_pos, target_orient = self.calculate_target_position(self.face_center)
                        
                        # 2. 첫 번째 명령인 경우 이동 거리 제한 (안전)
                        if not self.first_command_sent:
                            # 현재 위치에서 목표 위치까지의 거리 계산
                            movement_distance = np.linalg.norm(target_pos - self.current_position)
                            if movement_distance > self.safe_first_movement_limit:
                                # 이동 거리가 너무 크면 제한
                                direction = (target_pos - self.current_position) / movement_distance
                                target_pos = self.current_position + direction * self.safe_first_movement_limit
                                self.get_logger().warn(
                                    f'첫 번째 명령: 이동 거리를 {self.safe_first_movement_limit}mm로 제한 '
                                    f'(원래: {movement_distance:.1f}mm)'
                                )
                        
                        # 3. 목표 위치를 IK로 풀어서 각도로 변환
                        target_angles, ik_success = self.calculate_target_angles(target_pos, target_orient)
                        
                        # 4. 각도 명령 전송
                        if ik_success:
                            self.send_robot_command(target_angles, ik_success)
                            if not self.first_command_sent:
                                self.first_command_sent = True
                                self.get_logger().info(
                                    '✅ 첫 번째 명령 전송 완료. 이후 명령은 제한 없이 수행됩니다.'
                                )
                        else:
                            self.get_logger().warn('IK 계산 실패: 목표 위치에 도달할 수 없습니다.')
                    else:
                        # has_position이 False인 경우: 타임아웃 대기 중
                        elapsed_time = time.time() - self.start_time
                        if elapsed_time < self.position_timeout:
                            self.get_logger().debug(
                                f'서보 상태 대기 중... ({elapsed_time:.1f}/{self.position_timeout}초)'
                            )
                        else:
                            # 타임아웃 후에도 위치를 모르는 경우 (이론적으로는 발생하지 않아야 함)
                            self.get_logger().warn(
                                '현재 로봇 위치를 알 수 없습니다. '
                                '서보 상태(servo_status)를 기다리는 중... '
                                '(로봇을 조금 움직이면 자동으로 위치를 파악합니다)'
                            )
                        # 서보 상태 요청 (선택적)
                        self.request_servo_status()
            
            self.previous_face_center = self.face_center.copy()
            
            # 상태 텍스트 표시
            status_text = f'Face Detected: ({center_x}, {center_y})'
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            self.face_detected = False
            
            # 얼굴이 3초 이상 감지되지 않으면 경고
            if time.time() - self.last_face_time > 3.0 and self.last_face_time > 0:
                cv2.putText(frame, 'No Face Detected', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 화면 표시
        cv2.imshow('Face Tracking', frame)
        cv2.waitKey(1)
    
    def destroy_node(self):
        """노드 종료 시 리소스 정리"""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = FaceTrackingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'오류 발생: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

