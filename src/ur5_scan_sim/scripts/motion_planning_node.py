#!/usr/bin/env python3

# --- STEP 1: Import the tools we need ---
import math
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32
import time

# These tools help us talk to the robot's motors and draw shapes in Rviz
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class MotionPlanner(Node):
    """
    This is the "BRAIN" of the robot. 
    It tells the robot where to move and shows us what the robot is doing on the screen.
    """
    def __init__(self):
        # Give this node a name
        super().__init__('motion_planning_node')
        
        # --- STEP 2: Setup "Pipes" to send information ---
        
        # This pipe sends movement commands to the robot's arms
        self.publisher_ = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )
        
        # This pipe draws a blue line on the screen to show the "flight path"
        self.marker_pub = self.create_publisher(Marker, '/ur5_scanner/path_preview', 10)
        
        # This pipe writes yellow text (HUD) above the robot in the simulation
        self.hud_pub = self.create_publisher(Marker, '/ur5_scanner/status_hud', 10)
        
        # This creates a "Button" (Service) that starts the scanning when clicked
        self.srv = self.create_service(Trigger, '/ur5_scanner/start_scan', self.start_scan_callback)
        
        # This pipe listens to the "Scanner Node" to know how much has been scanned (0% to 100%)
        self.coverage_sub = self.create_subscription(Float32, '/ur5_scanner/coverage', self.coverage_callback, 10)
        
        # Variables to remember things
        self.current_coverage = 0.0
        self.current_status = "READY"
        
        # Names of the robot's "bones" (joints) we want to move
        self.joints = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # A list of "Checkpoints" (poses). The robot will visit these one by one.
        # Each row is a set of angles for the robot's joints.
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
        
        # Print a message to the black terminal box to say "Hello"
        self.get_logger().info('===== Motion Planner Initialized =====')
        self.get_logger().info('Watching for Service: /ur5_scanner/start_scan')
        
        # Periodically check if the node is still alive every 2 seconds
        self.create_timer(2.0, self.heartbeat_callback)
        
        # Show "READY" text on screen after 3 seconds
        self.create_timer(3.0, lambda: self.update_status_hud("READY"), once=True)
        
        # Draw the blue path preview on screen after 4 seconds
        self.create_timer(4.0, self.publish_path_preview, once=True)

    def heartbeat_callback(self):
        """Simple message to say: 'I am still running!'"""
        self.get_logger().info('Heartbeat: Motion Planner is SPINNING...')

    def coverage_callback(self, msg):
        """When the scanner tells us the coverage, we save it here."""
        self.current_coverage = msg.data
        if self.is_scanning:
            # Update the text on screen if we are currently scanning
            self.update_status_hud(self.current_status)

    def update_status_hud(self, text):
        """This function writes text in the 3D world for us to read."""
        try:
            self.current_status = text
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "hud"
            marker.id = 2
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            # Position the text high up in the air
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 1.2 
            marker.scale.z = 0.12 # Make text readable size
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0 # Make text yellow
            marker.color.a = 1.0
            
            # Combine the status name and the coverage percentage
            marker.text = f"UR5 SCANNER: {text} | COV: {self.current_coverage:.1f}%"
            self.hud_pub.publish(marker)
        except Exception as e:
            # If the clock is not ready yet, just print a warning
            self.get_logger().warn(f"HUD Update failed (waiting for clock?): {e}")

    def publish_path_preview(self):
        """This function draws special cyan lines in Rviz to show where the robot will move."""
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
        marker.color.b = 1.0 # Color: Cyan
        marker.color.a = 0.8 # Make it slightly see-through
        
        # These are the rough 3D spots the camera will go to
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
        """This is called when you click the 'Start' button."""
        if self.is_scanning:
            response.success = False
            response.message = "Scan already in progress"
            return response
            
        self.get_logger().info('Starting Scan sequence...')
        self.is_scanning = True
        self.pose_idx = 0
        self.update_status_hud(f"SCANNING (0/{len(self.scanning_poses)})")
        # Every 5 seconds, move to the next pose
        self.timer = self.create_timer(5.0, self.timer_callback)
        
        response.success = True
        response.message = "Scan sequence triggered"
        return response

    def timer_callback(self):
        """This is the 'Clock' that tells the robot: 'Move to the next spot now!'"""
        # If we visited all spots, stop.
        if self.pose_idx >= len(self.scanning_poses):
            self.get_logger().info('Scanning Finished.')
            self.update_status_hud("FINISHED")
            self.timer.cancel()
            self.is_scanning = False
            # Wait 5 seconds, then set status back to READY
            self.create_timer(5.0, lambda: self.update_status_hud("READY"), once=True)
            return

        # Pick the next spot from our list
        pos = self.scanning_poses[self.pose_idx]
        self.update_status_hud(f"SCANNING ({self.pose_idx + 1}/{len(self.scanning_poses)})")
        
        # Put the instructions in a "Letter" (Message)
        msg = JointTrajectory()
        msg.joint_names = self.joints
        
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start.sec = 4 # Take 4 seconds to reach the spot smoothly
        point.time_from_start.nanosec = 0
        
        msg.points.append(point)
        self.get_logger().info(f'Moving to pose {self.pose_idx}')
        # Send the letter to the robot!
        self.publisher_.publish(msg)
        
        # Move to the next spot in the list next time
        self.pose_idx += 1

# --- MAIN LOOP ---
def main(args=None):
    # Start ROS 2
    rclpy.init(args=args)
    # Start our Brain
    node = MotionPlanner()
    try:
        # Keep the Brain running until we stop it
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Safely close everything
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
