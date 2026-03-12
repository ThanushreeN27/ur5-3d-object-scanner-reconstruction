#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32
import time

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class MotionPlanner(Node):
    def __init__(self):
        super().__init__('motion_planning_node')
        self.publisher_ = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )
        self.marker_pub = self.create_publisher(Marker, '/ur5_scanner/path_preview', 10)
        self.hud_pub = self.create_publisher(Marker, '/ur5_scanner/status_hud', 10)
        self.srv = self.create_service(Trigger, '/ur5_scanner/start_scan', self.start_scan_callback)
        
        # Coverage Subscription
        self.coverage_sub = self.create_subscription(Float32, '/ur5_scanner/coverage', self.coverage_callback, 10)
        self.current_coverage = 0.0
        self.current_status = "READY"
        
        self.joints = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        self.scanning_poses = [
            [0.0, -1.0, 1.0, -1.57, -1.57, 0.0],
            [0.5, -1.0, 1.0, -1.57, -1.57, 0.0],
            [-0.5, -1.0, 1.0, -1.57, -1.57, 0.0],
            [0.0, -0.5, 0.5, -1.57, -1.57, 0.0],
            [1.0, -0.8, 1.0, -1.57, -1.57, 0.0],
            [-1.0, -0.8, 1.0, -1.57, -1.57, 0.0]
        ]
        
        self.pose_idx = 0
        self.is_scanning = False
        self.get_logger().info('===== Motion Planner Initialized =====')
        self.get_logger().info('Watching for Service: /ur5_scanner/start_scan')
        
        # Heartbeat to prove the node is ALIVE
        self.create_timer(2.0, self.heartbeat_callback)
        
        # Publish initial HUD status (delayed to ensure clock is ready)
        self.create_timer(3.0, lambda: self.update_status_hud("READY"), once=True)
        
        # Publish preview
        self.create_timer(4.0, self.publish_path_preview, once=True)

    def heartbeat_callback(self):
        self.get_logger().info('Heartbeat: Motion Planner is SPINNING...')

    def coverage_callback(self, msg):
        self.current_coverage = msg.data
        if self.is_scanning:
            self.update_status_hud(self.current_status)

    def update_status_hud(self, text):
        try:
            self.current_status = text
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "hud"
            marker.id = 2
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 1.2 # Slightly higher
            marker.scale.z = 0.12 
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0 # Yellow for better visibility
            marker.color.a = 1.0
            
            marker.text = f"UR5 SCANNER: {text} | COV: {self.current_coverage:.1f}%"
            self.hud_pub.publish(marker)
        except Exception as e:
            self.get_logger().warn(f"HUD Update failed (likely waiting for clock): {e}")

    def publish_path_preview(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.02 # Width of the line
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0 # Cyan
        marker.color.a = 0.8
        
        # We don't have a full IK solver here, but we can visualize the poses 
        # roughly or just publish the joint targets. For a true preview, we'd 
        # calculate the Forward Kinematics. Since we want a "Flight Path" visual,
        # let's approximate the camera link positions for these poses.
        # These are rough estimates for the scanning area.
        approx_positions = [
            [0.4, 0.0, 0.5],
            [0.5, 0.3, 0.45],
            [0.5, -0.3, 0.45],
            [0.3, 0.0, 0.6],
            [0.6, 0.4, 0.5],
            [0.6, -0.4, 0.5]
        ]
        
        for pos in approx_positions:
            p = Point()
            p.x = pos[0]
            p.y = pos[1]
            p.z = pos[2]
            marker.points.append(p)
            
        self.marker_pub.publish(marker)
        self.get_logger().info("Published trajectory preview to Rviz")

    def start_scan_callback(self, request, response):
        if self.is_scanning:
            response.success = False
            response.message = "Scan already in progress"
            return response
            
        self.get_logger().info('Starting Scan sequence...')
        self.is_scanning = True
        self.pose_idx = 0
        self.update_status_hud(f"SCANNING (0/{len(self.scanning_poses)})")
        self.timer = self.create_timer(5.0, self.timer_callback)
        
        response.success = True
        response.message = "Scan sequence triggered"
        return response

    def timer_callback(self):
        if self.pose_idx >= len(self.scanning_poses):
            self.get_logger().info('Scanning Finished.')
            self.update_status_hud("FINISHED")
            self.timer.cancel()
            self.is_scanning = False
            # Wait 5 seconds then go back to READY
            self.create_timer(5.0, lambda: self.update_status_hud("READY"), once=True)
            return

        pos = self.scanning_poses[self.pose_idx]
        self.update_status_hud(f"SCANNING ({self.pose_idx + 1}/{len(self.scanning_poses)})")
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
