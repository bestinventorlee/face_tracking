#!/usr/bin/env python3
"""
얼굴 트래킹 노드 Launch 파일
"""

from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 패키지에서 노드 실행
    node = Node(
        package='face_tracking',
        executable='face_tracking_node.py',
        name='face_tracking_node',
        output='screen',
        parameters=[
            {'tracking_speed': 30.0},              # deg/s
            {'tracking_accel': 50.0},              # deg/s²
            {'max_movement': 50.0},                # mm
            {'tracking_sensitivity': 0.5},         # 0.0 ~ 1.0
            {'camera_fov_horizontal': 60.0},       # degree
            {'camera_fov_vertical': 45.0},         # degree
            {'estimated_face_distance': 1000.0},   # mm
            {'movement_threshold': 5.0},          # 픽셀
            {'camera_index': 0},                  # 카메라 인덱스 (0 또는 1)
        ]
    )
    
    return LaunchDescription([node])

