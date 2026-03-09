#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import time

class MotionPlanner(Node):
    def __init__(self):
        super().__init__('motion_planning_node')
        self.publisher_ = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )
        self.joints = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # A set of typical UR5 poses that sweep around the space in front
        # [pan, lift, elbow, wrist1, wrist2, wrist3]
        self.scanning_poses = [
            [0.0, -1.0, 1.0, -1.57, -1.57, 0.0],
            [0.5, -1.0, 1.0, -1.57, -1.57, 0.0],
            [-0.5, -1.0, 1.0, -1.57, -1.57, 0.0],
            [0.0, -0.5, 0.5, -1.57, -1.57, 0.0],
            [1.0, -0.8, 1.0, -1.57, -1.57, 0.0],
            [-1.0, -0.8, 1.0, -1.57, -1.57, 0.0]
        ]
        
        self.timer = self.create_timer(5.0, self.timer_callback)
        self.pose_idx = 0
        self.get_logger().info('Motion Planner Started')

    def timer_callback(self):
        if self.pose_idx >= len(self.scanning_poses):
            self.get_logger().info('Scanning Finished.')
            return

        pos = self.scanning_poses[self.pose_idx]
        msg = JointTrajectory()
        msg.joint_names = self.joints
        
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start.sec = 4
        point.time_from_start.nanosec = 0
        
        msg.points.append(point)
        self.get_logger().info(f'Moving to pose {self.pose_idx}')
        self.publisher_.publish(msg)
        
        self.pose_idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
