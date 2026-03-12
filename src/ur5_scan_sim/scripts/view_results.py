#!/usr/bin/env python3

import open3d as o3d
import os
import sys

def main():
    # Default save directory
    save_dir = os.path.expanduser("~/ur5_ws/dataset")
    
    # Try to find the latest mesh or ply
    mesh_path = os.path.join(save_dir, "reconstructed_mesh.obj")
    pcd_path = os.path.join(save_dir, "accumulated_cloud.ply")
    
    if os.path.exists(mesh_path):
        print(f"Loading mesh: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], window_name="UR5 Scan Results - 3D Mesh")
    elif os.path.exists(pcd_path):
        print(f"Loading point cloud: {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)
        o3d.visualization.draw_geometries([pcd], window_name="UR5 Scan Results - Point Cloud")
    else:
        print(f"No results found in {save_dir}. Please run a scan first.")
        sys.exit(1)

if __name__ == "__main__":
    main()
