#!/usr/bin/env python3

# --- STEP 1: Import the tools we need ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge # Helps us change ROS images to standard Python images
import cv2
import os
import tf2_ros
import json
import numpy as np

class CameraSaver(Node):
    """
    This is the "PHOTOGRAPHER" node.
    It takes pictures (RGB), remembers depth information, and saves where the 
    camera was standing when it took each photo.
    """
    def __init__(self):
        # Give this node a name
        super().__init__('camera_node')
        self.img_count = 0
        
        # This draws a "View Cone" in Rviz so we can see what the camera sees
        self.fov_pub = self.create_publisher(Marker, '/camera/fov_visual', 10)
        self.create_timer(1.0, self.publish_fov_visual)
        
        # --- Listeners (Subscribers) ---
        # Listen for the actual color photo
        self.subscription_rgb = self.create_subscription(Image, '/camera/image', self.image_callback, 10)
        # Listen for the depth image (how far away things are)
        self.subscription_depth = self.create_subscription(Image, '/camera/depth_image', self.depth_callback, 10)
        # Listen for camera specifications (like lens zoom level)
        self.subscription_info = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, 10)
        
        self.camera_info = None
        self.nerf_frames = []
        self.bridge = CvBridge()
        
        # Pick a folder to save our photos (dataset)
        self.declare_parameter('save_dir', '~/ur5_ws/dataset')
        self.save_dir = os.path.expanduser(self.get_parameter('save_dir').get_parameter_value().string_value)
        
        self.latest_depth = None

        # Create the folder if it doesn't exist yet
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # This acts like a "GPS" to tell us exactly where the camera is in the 3D world
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # A simple text file to list all the camera positions
        self.pose_file = open(os.path.join(self.save_dir, "poses.txt"), "a")
        self.get_logger().info("Camera node started, waiting for images...")
        self.last_save_time = self.get_clock().now()

    def publish_fov_visual(self):
        """Draws a little blue 'cone' or pyramid to show the camera's lens in Rviz."""
        marker = Marker()
        marker.header.frame_id = "camera_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "fov"
        marker.id = 1
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01 
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0 # Light Blue
        marker.color.a = 0.5 # Slightly see-through
        
        # Define the size of our blue cone (half a meter long)
        d, w, h = 0.5, 0.3, 0.2
        
        p0 = Point(x=0.0, y=0.0, z=0.0) # The tip of the lens
        p1, p2, p3, p4 = Point(x=d, y=w, z=h), Point(x=d, y=-w, z=h), Point(x=d, y=-w, z=-h), Point(x=d, y=w, z=-h)
        
        # Connect the dots to make the shape
        marker.points.extend([p0, p1, p0, p2, p0, p3, p0, p4]) 
        marker.points.extend([p1, p2, p2, p3, p3, p4, p4, p1]) 
        
        self.fov_pub.publish(marker)

    def depth_callback(self, msg):
        """Remembers the latest depth photo so we can save it later."""
        self.latest_depth = msg

    def info_callback(self, msg):
        """Learns the camera's secret numbers (like focal length) once."""
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("Camera Intrinsics Captured for NeRF Export")

    def image_callback(self, msg):
        """This is called whenever a new color photo arrives."""
        current_time = self.get_clock().now()
        # Save only one photo every 1 second (so we don't fill up the hard drive too fast)
        if (current_time - self.last_save_time).nanoseconds > 1e9 and self.latest_depth is not None:
            try:
                # Ask our GPS: 'Where is the camera link in the world right now?'
                trans = self.tf_buffer.lookup_transform('world', 'camera_link', rclpy.time.Time())
                
                # Turn the photos from ROS format into standard Python images
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                cv_depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')
                
                # Come up with names for the files (like image_0001.png)
                rgb_filename = f"image_{self.img_count:04d}.png"
                depth_filename = f"depth_{self.img_count:04d}.png"
                
                # 1. Save the color photo
                cv2.imwrite(os.path.join(self.save_dir, rgb_filename), cv_img)
                
                # 2. Save the depth photo (convert it to millimeters so it looks right)
                depth_16u = np.nan_to_num(cv_depth, posinf=0.0, neginf=0.0)
                depth_16u = (depth_16u * 1000.0).astype(np.uint16)
                cv2.imwrite(os.path.join(self.save_dir, depth_filename), depth_16u)
                
                # 3. Save the camera's location (translation and rotation)
                tx, ty, tz = trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z
                qx, qy, qz, qw = trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w
                
                # Write a line in our text file
                self.pose_file.write(f"{self.img_count:04d} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
                self.pose_file.flush()
                
                # --- Advanced Math for 3D Experts (NeRF) ---
                # Turn the location into a special 4x4 matrix
                from scipy.spatial.transform import Rotation as R
                rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rot_matrix
                transform_matrix[:3, 3] = [tx, ty, tz]
                
                # Add this photo information to our big JSON list
                frame = {
                    "file_path": rgb_filename,
                    "transform_matrix": transform_matrix.tolist()
                }
                self.nerf_frames.append(frame)
                self.save_nerf_json() # Update the master master file
                
                self.get_logger().info(f"Saved: {rgb_filename} (NeRF Frame {self.img_count})")
                self.img_count += 1
                self.last_save_time = current_time
            except Exception as e:
                # If something went wrong, print an error message
                self.get_logger().error(f"Sync/Save Error: {e}")

    def save_nerf_json(self):
        """This saves a file called 'transforms.json' which is used for 3D AI (NeRF)."""
        if self.camera_info is None: return
        
        # Get focal length and center point of the lens
        fx, fy = self.camera_info.k[0], self.camera_info.k[4]
        cx, cy = self.camera_info.k[2], self.camera_info.k[5]
        
        # Organize all the secret camera data into one block
        nerf_data = {
            "fl_x": fx, "fl_y": fy,
            "cx": cx, "cy": cy,
            "w": self.camera_info.width, "h": self.camera_info.height,
            "camera_model": "OPENCV",
            "frames": self.nerf_frames
        }
        
        # Write it to the file
        with open(os.path.join(self.save_dir, "transforms.json"), "w") as f:
            json.dump(nerf_data, f, indent=4)

    def destroy_node(self):
        """Wait! Before the program closes, make sure we save the files one last time."""
        self.save_nerf_json()
        self.pose_file.close()
        super().destroy_node()

def main(args=None):
    # Start up ROS 2
    rclpy.init(args=args)
    # Start the photographer
    node = CameraSaver()
    try:
        # Keep running
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Clean up when finished
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
