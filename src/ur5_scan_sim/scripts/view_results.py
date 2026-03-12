#!/usr/bin/env python3

# --- STEP 1: Import the tools we need ---
import open3d as o3d
import os
import sys

def main():
    """
    This is the "3D VIEWER" tool.
    It opens a window so you can rotate and zoom into your finished model.
    """
    # Look in the dataset folder
    save_dir = os.path.expanduser("~/ur5_ws/dataset")
    
    # Part 1: Try to find the solid mesh (Skin)
    mesh_path = os.path.join(save_dir, "reconstructed_mesh.obj")
    # Part 2: Look for the point cloud (Dots) if the mesh isn't there
    pcd_path = os.path.join(save_dir, "accumulated_cloud.ply")
    
    if os.path.exists(mesh_path):
        print(f"Opening your 3D Mesh: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        # Open the interactive 3D window
        o3d.visualization.draw_geometries([mesh], window_name="UR5 Scan Results - 3D Mesh")
        
    elif os.path.exists(pcd_path):
        print(f"Opening your Point Cloud (Dots): {pcd_path}")
        pcd = o3d.io.read_point_cloud(pcd_path)
        # Open the interactive 3D window
        o3d.visualization.draw_geometries([pcd], window_name="UR5 Scan Results - Point Cloud")
        
    else:
        # If nothing is found, tell the user how to fix it
        print(f"Error: No 3D results found in {save_dir}.")
        print("Tip: Run the scanner first to create a model!")
        sys.exit(1)

if __name__ == "__main__":
    main()
