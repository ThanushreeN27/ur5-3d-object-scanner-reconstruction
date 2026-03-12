#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import tf2_ros

class CameraSaver(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.subscription_rgb = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10)
        self.subscription_depth = self.create_subscription(
            Image,
            '/camera/depth_image',
            self.depth_callback,
            10)
        self.bridge = CvBridge()
        self.declare_parameter('save_dir', '~/ur5_ws/dataset')
        self.save_dir = os.path.expanduser(self.get_parameter('save_dir').get_parameter_value().string_value)
        
        self.latest_depth = None

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.pose_file = open(os.path.join(self.save_dir, "poses.txt"), "a")
        self.get_logger().info("Camera node started, waiting for images...")
        self.last_save_time = self.get_clock().now()

    def depth_callback(self, msg):
        self.latest_depth = msg

    def image_callback(self, msg):
        current_time = self.get_clock().now()
        # Save one image every 1 second
        if (current_time - self.last_save_time).nanoseconds > 1e9 and self.latest_depth is not None:
            try:
                # Try to get the latest transform from world to camera_link
                trans = self.tf_buffer.lookup_transform('world', 'camera_link', rclpy.time.Time())
                
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
                
                rgb_filename = f"image_{self.img_count:04d}.png"
                depth_filename = f"depth_{self.img_count:04d}.png"
                
                cv2.imwrite(os.path.join(self.save_dir, rgb_filename), cv_img)
                # Save depth as 16-bit PNG or 32-bit TIFF
                # Assuming 32FC1 from Gazebo, normalize or save as tiff
                # We can also convert to 16UC1 (millimeter scale)
                import numpy as np
                depth_16u = np.nan_to_num(cv_depth, posinf=0.0, neginf=0.0)
                depth_16u = (depth_16u * 1000.0).astype(np.uint16)
                cv2.imwrite(os.path.join(self.save_dir, depth_filename), depth_16u)
                
                # Save pose
                tx = trans.transform.translation.x
                ty = trans.transform.translation.y
                tz = trans.transform.translation.z
                qx = trans.transform.rotation.x
                qy = trans.transform.rotation.y
                qz = trans.transform.rotation.z
                qw = trans.transform.rotation.w
                
                self.pose_file.write(f"{self.img_count:04d} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
                self.pose_file.flush()
                
                self.get_logger().info(f"Saved: {rgb_filename} and {depth_filename}")
                self.img_count += 1
                self.last_save_time = current_time
            except Exception as e:
                self.get_logger().error(f"Error saving image or tf: {e}")

    def destroy_node(self):
        self.pose_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
