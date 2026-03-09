#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
import std_msgs.msg

class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__('image_processing_node')
        
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
            
        self.publisher_features = self.create_publisher(Image, '/vision/features_image', 10)
        self.publisher_pc = self.create_publisher(PointCloud2, '/vision/pointcloud', 10)
        
        self.bridge = CvBridge()
        self.orb = cv2.ORB_create(nfeatures=500)
        
        self.latest_depth = None
        
        # Camera Intrinsics (approximated for Gazebo 800x600 1.047 fov)
        width = 800
        height = 600
        fov = 1.047
        self.fx = (width / 2) / np.tan(fov / 2)
        self.fy = self.fx
        self.cx = width / 2
        self.cy = height / 2

        self.get_logger().info('Image Processing Node started: Publishing features and colored PointCloud2')

    def depth_callback(self, msg):
        self.latest_depth = msg

    def image_callback(self, msg):
        if self.latest_depth is None:
            return
            
        try:
            # Convert ROS Image messages to OpenCV images
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='32FC1')
            
            # --- Feature Extraction ---
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
            
            # Draw keypoints
            img_with_keypoints = cv2.drawKeypoints(cv_image, keypoints, None, color=(0, 255, 0), flags=0)
            
            # Publish image with features drawn
            viz_msg = self.bridge.cv2_to_imgmsg(img_with_keypoints, encoding="bgr8")
            viz_msg.header = msg.header # Keep same timestamp and frame_id
            self.publisher_features.publish(viz_msg)
            
            # --- PointCloud2 Generation ---
            pc_msg = self.create_point_cloud_message(cv_image, cv_depth, msg.header)
            self.publisher_pc.publish(pc_msg)
            
        except Exception as e:
            self.get_logger().error('Error processing images: %r' % e)

    def create_point_cloud_message(self, color_img, depth_img, header):
        # We sample the image to avoid massive point clouds per frame
        # e.g., take every 4th pixel
        step = 4
        rows, cols = depth_img.shape
        points = []

        for v in range(0, rows, step):
            for u in range(0, cols, step):
                z = depth_img[v, u]
                if np.isnan(z) or np.isinf(z) or z <= 0.1 or z > 10.0:
                    continue
                
                # Back-project to 3D roughly
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                
                # Colors
                b, g, r = color_img[v, u]
                a = 255
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
                points.append([x, y, z, rgb])

        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(points)
        msg.is_dense = False
        msg.is_bigendian = False

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        
        buffer = []
        for p in points:
            buffer.append(struct.pack('<fffI', p[0], p[1], p[2], p[3]))
        
        msg.data = b''.join(buffer)
        return msg

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
