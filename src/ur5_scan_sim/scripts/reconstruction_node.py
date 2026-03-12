#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation
import threading
import copy

from rcl_interfaces.msg import SetParametersResult

class ReconstructionNode(Node):
    def __init__(self):
        super().__init__('reconstruction_node')
        self.declare_parameter('mesh_output_path', '~/ur5_ws/live_reconstructed_mesh.obj')
        self.declare_parameter('pc_output_path', '~/ur5_ws/live_point_cloud.ply')
        
        # Advanced Filtering Parameters
        self.declare_parameter('nb_neighbors', 20)
        self.declare_parameter('std_ratio', 2.0)
        self.declare_parameter('radius', 0.05)
        self.declare_parameter('nb_points', 16)
        
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        # Subscribe to the generated pointclouds
        self.subscription_pc = self.create_subscription(
            PointCloud2,
            '/vision/pointcloud',
            self.pc_callback,
            10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # We accumulate point clouds directly (or we can use TSDF if we had depth images)
        # Here we accumulate the colorized point clouds transformed to world coordinates.
        self.accumulated_pcd = o3d.geometry.PointCloud()
        
        self.get_logger().info('Reconstruction Node Started. Waiting for /vision/pointcloud...')

        # Protect against concurrent access to self.accumulated_pcd
        self.pcd_lock = threading.Lock()
        
        # Setup a timer to periodically (e.g. every 10 seconds) extract and save the mesh
        self.timer = self.create_timer(10.0, self.save_mesh)
        
        self.cloud_count = 0

    def pc_callback(self, msg):
        try:
            # Get the transform from world -> camera_link (where the point cloud is generated)
            trans = self.tf_buffer.lookup_transform('world', msg.header.frame_id, msg.header.stamp, rclpy.duration.Duration(seconds=0.5))
            
            # Extract point cloud data
            pc_data = list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))
            if not pc_data:
                return

            points = []
            colors = []
            for p in pc_data:
                points.append([p[0], p[1], p[2]])
                
                # Unpack RGB
                rgb_float = p[3]
                import struct
                # Convert float to bytes
                b = struct.pack('f', rgb_float)
                # Unpack bytes to integers
                # Format: BGRA or similar depending on how it was packed
                # If we pack BBBR into I and then view as FLOAT32 (which ros2 point_cloud2 does)
                val = struct.unpack('I', b)[0]
                
                # We packed as b, g, r, a
                blue = (val & 0x000000FF)
                green = (val & 0x0000FF00) >> 8
                red = (val & 0x00FF0000) >> 16
                colors.append([red / 255.0, green / 255.0, blue / 255.0])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
            
            # Transform point cloud to world frame using the tf
            tx = trans.transform.translation.x
            ty = trans.transform.translation.y
            tz = trans.transform.translation.z
            qx = trans.transform.rotation.x
            qy = trans.transform.rotation.y
            qz = trans.transform.rotation.z
            qw = trans.transform.rotation.w
            
            rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rot
            extrinsic[:3, 3] = [tx, ty, tz]
            
            pcd.transform(extrinsic)
            
            # Downsample to avoid exploding memory
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            
            with self.pcd_lock:
                self.accumulated_pcd += pcd
                # Periodically voxel downsample the accumulated cloud
                if self.cloud_count % 5 == 0:
                    self.accumulated_pcd = self.accumulated_pcd.voxel_down_sample(voxel_size=0.005)

            self.cloud_count += 1
            self.get_logger().info(f'Accumulated cloud {self.cloud_count}. Current points: {len(self.accumulated_pcd.points)}')

        except Exception as e:
            self.get_logger().error(f"Error accumulating point cloud: {e}")

    def parameter_callback(self, params):
        for param in params:
            self.get_logger().info(f"Parameter '{param.name}' updated to: {param.value}")
        return SetParametersResult(successful=True)

    def save_mesh(self):
        with self.pcd_lock:
            if len(self.accumulated_pcd.points) < 1000:
                self.get_logger().info("Not enough points to reconstruct yet.")
                return
            
            self.get_logger().info("Generating Surface Mesh using Poisson reconstruction...")
            pcd = self.accumulated_pcd
            
            # Keep a working copy for estimation
            pcd_copy = copy.deepcopy(pcd)
            pcd_copy.estimate_normals()
            pcd_copy.orient_normals_consistent_tangent_plane(100)
            
            # --- Advanced Filtering ---
            nb_neighbors = self.get_parameter('nb_neighbors').value
            std_ratio = self.get_parameter('std_ratio').value
            radius = self.get_parameter('radius').value
            nb_points = self.get_parameter('nb_points').value
            
            self.get_logger().info(f"Applying filters (neighbors={nb_neighbors}, std={std_ratio}, radius={radius})")
            
            cl, ind = pcd_copy.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            pcd_copy = pcd_copy.select_by_index(ind)
            
            cl, ind = pcd_copy.remove_radius_outlier(nb_points=nb_points, radius=radius)
            pcd_copy = pcd_copy.select_by_index(ind)
            
            try:
                mesh_path = os.path.expanduser(self.get_parameter('mesh_output_path').get_parameter_value().string_value)
                pc_path = os.path.expanduser(self.get_parameter('pc_output_path').get_parameter_value().string_value)
                
                os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
                
                # Poisson reconstruction often works better with filtered clouds
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_copy, depth=9)
                
                o3d.io.write_triangle_mesh(mesh_path, mesh)
                o3d.io.write_point_cloud(pc_path, pcd_copy)
                self.get_logger().info(f"Saved filtered mesh to {mesh_path} and PC to {pc_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to generate mesh: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ReconstructionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    # Save a final time upon exit
    node.save_mesh()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
