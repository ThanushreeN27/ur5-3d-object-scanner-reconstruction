#!/usr/bin/env python3

# --- STEP 1: Import the tools we need ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge # Helps us change ROS images to standard Python images
import cv2
import numpy as np
import struct

class ImageProcessingNode(Node):
    """
    This is the "AI VISION" node.
    It looks at the camera photos and:
    1. Finds interesting spots (features) to show us.
    2. Turns flat photos into 3D "Point Clouds" (dots in space).
    """
    def __init__(self):
        # Give this node a name
        super().__init__('image_processing_node')
        
        # --- Listeners (Subscribers) ---
        # Listen for the color photo
        self.subscription_rgb = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        # Listen for the depth image (how far away things are)
        self.subscription_depth = self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
            
        # --- Publishers ---
        # Send out the "Dots" (Point Cloud) for the 3D map
        self.publisher_features = self.create_publisher(Image, '/vision/features_image', 10)
        self.publisher_pc = self.create_publisher(PointCloud2, '/vision/pointcloud', 10)
        
        self.bridge = CvBridge()
        
        # This is a tool that "detects" interesting corners and edges in a photo
        self.orb = cv2.ORB_create(nfeatures=500)
        
        self.latest_depth = None
        
        # Secret camera numbers that help us turn flat pixels into 3D spots
        # (Calculated for our specific simulation screen size)
        width, height, fov = 800, 600, 1.047
        self.fx = (width / 2) / np.tan(fov / 2)
        self.fy = self.fx
        self.cx, self.cy = width / 2, height / 2

        self.get_logger().info('Vision Node started: Processing dots and features...')

    def depth_callback(self, msg):
        """Saves the latest depth photo so we can use it when the color photo arrives."""
        self.latest_depth = msg

    def image_callback(self, msg):
        """This runs every time a new color photo comes in."""
        if self.latest_depth is None: return
            
        try:
            # Change the ROS photos into standard photos we can work with
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='32FC1')
            
            # --- 1. FIND INTERESTING SPOTS ---
            # Make the photo black and white first (easier for the computer to see)
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # Find the 500 most interesting spots
            keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
            
            # Draw green circles on those spots so we can see them in Rviz
            img_with_keypoints = cv2.drawKeypoints(cv_image, keypoints, None, color=(0, 255, 0), flags=0)
            viz_msg = self.bridge.cv2_to_imgmsg(img_with_keypoints, encoding="bgr8")
            viz_msg.header = msg.header 
            self.publisher_features.publish(viz_msg)
            
            # --- 2. CREATE 3D DOTS (Point Cloud) ---
            pc_msg = self.create_point_cloud_message(cv_image, cv_depth, msg.header)
            self.publisher_pc.publish(pc_msg)
            
        except Exception as e:
            # If there's a mistake, tell us in the terminal
            self.get_logger().error(f'Vision Processing Error: {e}')

    def create_point_cloud_message(self, color_img, depth_img, header):
        """This turns a flat photo into a 3D shape made of dots."""
        # Optimization: Look at every 4th pixel (so the computer doesn't get too hot!)
        step = 4
        rows, cols = depth_img.shape
        points = []

        for v in range(0, rows, step):
            for u in range(0, cols, step):
                # How far away is this specific pixel (point)?
                z = depth_img[v, u]
                
                # If the point is too close, too far, or broken (broken = NaN), ignore it.
                if np.isnan(z) or np.isinf(z) or z <= 0.1 or z > 10.0:
                    continue
                
                # MATH: Calculate where this pixel is in the 3D room
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                
                # Pick the color of this pixel
                b, g, r = color_img[v, u]
                # Combine Red, Green, Blue into a "secret code" number (32-bit integer)
                rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
                
                # Add this 3D dot to our list
                points.append([x, y, z, rgb])

        # Build the final message to send to Rviz
        msg = PointCloud2()
        msg.header = header
        msg.height, msg.width = 1, len(points)
        msg.is_dense, msg.is_bigendian = False, False

        # Define how the data is stored (X, Y, Z, and Color)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        msg.point_step = 16 # Each dot uses 16 bytes of memory
        msg.row_step = msg.point_step * len(points)
        
        # Pack all the dots into a binary buffer (very fast for computers)
        buffer = []
        for p in points:
            buffer.append(struct.pack('<fffI', p[0], p[1], p[2], p[3]))
        
        msg.data = b''.join(buffer)
        return msg

def main(args=None):
    # Start ROS 2
    rclpy.init(args=args)
    # Start the vision node
    node = ImageProcessingNode()
    try:
        # Keep running
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Safely close
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
