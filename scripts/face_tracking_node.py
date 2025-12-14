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
        # IK를 직접 계산하여 각도로 전송하므로 servo_angles_with_speed 사용
        self.angle_pub = self.create_publisher(
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
        # cv2.data가 없는 경우를 대비한 fallback 처리
        try:
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            else:
                # cv2.data가 없는 경우 직접 경로 찾기
                import os
                # OpenCV 설치 경로에서 찾기
                possible_paths = [
                    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                ]
                cascade_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        cascade_path = path
                        break
                
                if cascade_path is None:
                    # Python 패키지 경로에서 찾기
                    try:
                        # opencv-python-headless나 opencv-contrib-python의 경우
                        cv2_path = os.path.dirname(cv2.__file__)
                        cascade_path = os.path.join(cv2_path, 'data', 'haarcascade_frontalface_default.xml')
                        if not os.path.exists(cascade_path):
                            raise FileNotFoundError
                    except:
                        raise RuntimeError(
                            'Haar Cascade 파일을 찾을 수 없습니다. '
                            '다음 명령으로 설치하세요: '
                            'sudo apt install opencv-data'
                        )
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError(f'Haar Cascade 파일을 로드할 수 없습니다: {cascade_path}')
            self.get_logger().info(f'얼굴 인식기 초기화 완료: {cascade_path}')
        except Exception as e:
            self.get_logger().error(f'얼굴 인식기 초기화 실패: {e}')
            raise
        
        # 카메라 초기화
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('카메라를 열 수 없습니다!')
            raise RuntimeError('카메라 초기화 실패')
        
        # 카메라 해상도 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 현재 엔드이펙트 위치 및 각도 (mm, degree)
        self.current_position = np.array([0.0, 0.0, 0.0])  # x, y, z (mm)
        self.current_orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw (degree)
        self.current_angles = np.zeros(6)  # 서보 각도 (degree)
        self.has_position = False
        
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
        
        self.tracking_speed = self.get_parameter('tracking_speed').get_parameter_value().double_value
        self.tracking_accel = self.get_parameter('tracking_accel').get_parameter_value().double_value
        self.max_movement = self.get_parameter('max_movement').get_parameter_value().double_value
        self.tracking_sensitivity = self.get_parameter('tracking_sensitivity').get_parameter_value().double_value
        self.camera_fov_horizontal = self.get_parameter('camera_fov_horizontal').get_parameter_value().double_value
        self.camera_fov_vertical = self.get_parameter('camera_fov_vertical').get_parameter_value().double_value
        self.estimated_face_distance = self.get_parameter('estimated_face_distance').get_parameter_value().double_value
        self.movement_threshold = self.get_parameter('movement_threshold').get_parameter_value().double_value
        
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
            if not self.has_position:
                self.get_logger().warn(
                    '초기 서보 상태를 받지 못했습니다. '
                    'servo_status 토픽을 기다리는 중... '
                    '(서보가 움직이거나 상태를 보내면 자동으로 업데이트됩니다)'
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
                    
                    self.get_logger().debug(
                        f'현재 위치 업데이트: ({self.current_position[0]:.1f}, '
                        f'{self.current_position[1]:.1f}, {self.current_position[2]:.1f}) mm'
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
        
        msg = Float32MultiArray()
        # servo_angles_with_speed 형식: [angle1~6, speed, accel] (8개)
        msg.data = [
            float(target_angles[0]),  # 서보 1 각도 (degree)
            float(target_angles[1]),  # 서보 2 각도 (degree)
            float(target_angles[2]),  # 서보 3 각도 (degree)
            float(target_angles[3]),  # 서보 4 각도 (degree)
            float(target_angles[4]),  # 서보 5 각도 (degree)
            float(target_angles[5]),  # 서보 6 각도 (degree)
            self.tracking_speed,      # speed (deg/s)
            self.tracking_accel       # accel (deg/s²)
        ]
        
        self.angle_pub.publish(msg)
        self.get_logger().debug(
            f'로봇 명령 전송 (각도): [{target_angles[0]:.1f}, {target_angles[1]:.1f}, '
            f'{target_angles[2]:.1f}, {target_angles[3]:.1f}, {target_angles[4]:.1f}, '
            f'{target_angles[5]:.1f}]°, 속도={self.tracking_speed} deg/s'
        )
    
    def process_frame(self):
        """카메라 프레임 처리 및 얼굴 트래킹"""
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
                        
                        # 2. 목표 위치를 IK로 풀어서 각도로 변환
                        target_angles, ik_success = self.calculate_target_angles(target_pos, target_orient)
                        
                        # 3. 각도 명령 전송
                        if ik_success:
                            self.send_robot_command(target_angles, ik_success)
                        else:
                            self.get_logger().warn('IK 계산 실패: 목표 위치에 도달할 수 없습니다.')
                    else:
                        self.get_logger().warn('현재 로봇 위치를 알 수 없어 명령을 전송할 수 없습니다.')
            
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

