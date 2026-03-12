#!/usr/bin/env python3

# --- STEP 1: Import the tools we need ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
import open3d as o3d # This is a powerful 3D math tool
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation
import threading
import copy
import os

from rcl_interfaces.msg import SetParametersResult

class ReconstructionNode(Node):
    """
    This is the "MASTER BUILDER" node.
    It takes small 3D snapshots from the camera and glues them together 
    to make one big, detailed 3D model of the object.
    """
    def __init__(self):
        # Give this node a name
        super().__init__('reconstruction_node')
        
        # --- File Settings ---
        self.declare_parameter('mesh_output_path', '~/ur5_ws/live_reconstructed_mesh.obj')
        self.declare_parameter('pc_output_path', '~/ur5_ws/live_point_cloud.ply')
        
        # --- Magic Numbers for Cleaning the Model ---
        self.declare_parameter('nb_neighbors', 20)      # How many neighbor dots to look at
        self.declare_parameter('std_ratio', 2.0)        # How strict to be when deleting "flying" dots
        self.declare_parameter('radius', 0.05)          # Circle size for checking dots
        self.declare_parameter('nb_points', 16)         # Minimum dots needed in that circle
        self.declare_parameter('use_icp', True)         # Enable "Auto-Align" (ICP)
        
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # --- Listeners (Subscribers) ---
        # Listen for the incoming stream of 3D dots
        self.subscription_pc = self.create_subscription(
            PointCloud2,
            '/vision/pointcloud',
            self.pc_callback,
            10)
        
        # This acts like a "GPS" to tell us how to move the dots to the right place in the room
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # This is where we store the BIG model (it starts empty)
        self.accumulated_pcd = o3d.geometry.PointCloud()
        
        # --- Publishers (Speaking to Rviz) ---
        # Tells Rviz how much of the object we have scanned (0% - 100%)
        self.coverage_pub = self.create_publisher(Float32, '/ur5_scanner/coverage', 10)
        # Draws a red cube in Rviz wherever there is a "gap" in the scan
        self.gap_pub = self.create_publisher(Marker, '/ur5_scanner/gap_marker', 10)
        
        self.get_logger().info('Master Builder Node Started. Waiting for dots...')

        # This "Lock" prevents the computer from getting confused if two things happen at once
        self.pcd_lock = threading.Lock()
        
        # Every 10 seconds, save the 3D model to a file
        self.timer = self.create_timer(10.0, self.save_mesh)
        
        self.cloud_count = 0

    def pc_callback(self, msg):
        """This runs every time a small set of 3D dots arrives."""
        try:
            # 1. FIND THE CAMERA: Where was the camera when it took this snapshot?
            trans = self.tf_buffer.lookup_transform('world', msg.header.frame_id, msg.header.stamp, rclpy.duration.Duration(seconds=0.5))
            
            # 2. EXTRACT DOTS: Turn the ROS message into a list of points and colors
            pc_data = list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
            if not pc_data: return

            points, colors = [], []
            import struct
            for p in pc_data:
                points.append([p[0], p[1], p[2]])
                # Unpack the "secret code" for the color
                b_raw = struct.pack('f', p[3])
                val = struct.unpack('I', b_raw)[0]
                blue, green, red = (val & 0xFF), (val >> 8 & 0xFF), (val >> 16 & 0xFF)
                colors.append([red / 255.0, green / 255.0, blue / 255.0])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
            
            # 3. MOVE DOTS: Move the dots from "Camera Space" to the "World Space" coordinates
            tx, ty, tz = trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z
            qx, qy, qz, qw = trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w
            rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            extrinsic = np.eye(4)
            extrinsic[:3, :3], extrinsic[:3, 3] = rot, [tx, ty, tz]
            pcd.transform(extrinsic)
            
            # 4. AUTO-ALIGN (ICP): This "shakes" the dots slightly to find the perfect fit
            if self.get_parameter('use_icp').value and len(self.accumulated_pcd.points) > 1000:
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd, self.accumulated_pcd, 0.02, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint())
                pcd.transform(reg_p2p.transformation)

            # 5. MERGE: Add the new dots to our BIG master model
            pcd = pcd.voxel_down_sample(voxel_size=0.01) # Keep only one dot per 1cm cube
            with self.pcd_lock:
                self.accumulated_pcd += pcd
                # Every 5 snapshots, clean up the master model to keep it smooth
                if self.cloud_count % 5 == 0:
                    self.accumulated_pcd = self.accumulated_pcd.voxel_down_sample(voxel_size=0.005)

            self.cloud_count += 1
            self.analyze_coverage() # Check if we missed any spots

        except Exception as e:
            self.get_logger().error(f"Pointcloud Error: {e}")

    def analyze_coverage(self):
        """This calculates how much of the object is finished."""
        if len(self.accumulated_pcd.points) < 500: return
        
        # Look only at the area where the object is supposed to be
        points = np.asarray(self.accumulated_pcd.points)
        target_box = (points[:, 0] > 0.2) & (points[:, 0] < 0.6) & \
                     (points[:, 1] > -0.2) & (points[:, 1] < 0.2) & \
                     (points[:, 2] > 0.0) & (points[:, 2] < 0.4)
        
        target_points = points[target_box]
        
        # If we have 20,000 dots in that area, we say it is 100% finished
        coverage = min(100.0, (len(target_points) / 20000.0) * 100.0)
        self.coverage_pub.publish(Float32(data=coverage))
        
        # If it's not finished, find a "hole" or "gap" where we need more dots
        if coverage < 95.0 and len(target_points) > 0:
            mean_pos = np.mean(target_points, axis=0)
            # Find the opposite side of the center to find a likely gap
            gap_pos = [0.8 - mean_pos[0], -mean_pos[1], mean_pos[2]]
            self.publish_gap_marker(gap_pos)

    def publish_gap_marker(self, pos):
        """This puts a red cube in Rviz to tell us 'SCAN HERE!'."""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "nbv_lite"; marker.id = 3
        marker.type = Marker.CUBE; marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, pos)
        marker.scale.x = marker.scale.y = marker.scale.z = 0.08
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 0.6
        self.gap_pub.publish(marker)

    def parameter_callback(self, params):
        """This is called if we change a setting while the program is running."""
        return SetParametersResult(successful=True)

    def save_mesh(self):
        """This turns the 3D dots into a solid object (mesh) and saves it to a file."""
        with self.pcd_lock:
            if len(self.accumulated_pcd.points) < 1000: return
            
            self.get_logger().info("Generating Surface Mesh (Connecting the dots)...")
            pcd_copy = copy.deepcopy(self.accumulated_pcd)
            pcd_copy.estimate_normals() # Figure out which way is "up" for each dot
            pcd_copy.orient_normals_consistent_tangent_plane(100)
            
            # --- CLEANING ---
            # Remove "stray" dots that are floating in the air by themselves
            cl, ind = pcd_copy.remove_statistical_outlier(
                nb_neighbors=self.get_parameter('nb_neighbors').value, 
                std_ratio=self.get_parameter('std_ratio').value)
            pcd_copy = pcd_copy.select_by_index(ind)
            
            try:
                # Find the location on the computer to save the files
                mesh_path = os.path.expanduser(self.get_parameter('mesh_output_path').value)
                pc_path = os.path.expanduser(self.get_parameter('pc_output_path').value)
                os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
                
                # Turn the dots into a solid "skin" (surface)
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_copy, depth=9)
                
                # Save the solid object (.obj) and the clean dots (.ply)
                o3d.io.write_triangle_mesh(mesh_path, mesh)
                o3d.io.write_point_cloud(pc_path, pcd_copy)
                self.get_logger().info(f"Updated 3D results in dataset directory.")
            except Exception as e:
                self.get_logger().error(f"Mesh Export Failed: {e}")

def main(args=None):
    # Start ROS 2
    rclpy.init(args=args)
    # Start the master builder
    node = ReconstructionNode()
    try:
        # Keep running
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Save one last time before closing
    node.save_mesh()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
