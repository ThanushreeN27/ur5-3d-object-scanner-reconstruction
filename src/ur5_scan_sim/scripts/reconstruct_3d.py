#!/usr/bin/env python3

# --- STEP 1: Import the tools we need ---
import os
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

def extract_and_match_features(image1_path, image2_path):
    """
    This function finds "Matching Spots" in two different photos.
    It helps the computer understand how the two photos relate to each other.
    """
    print(f"Looking for matching spots in {image1_path} and {image2_path}...")
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Tool that detects interesting corners/edges
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is not None and des2 is not None:
        # Match the spots between the two images
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        print(f"-> Found {len(matches)} matching spots!")

        # Draw lines between matching spots and save a picture for us to see
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("feature_matches.png", img_matches)
        print("-> Saved feature_matches.png (Open this to see the lines!)")

def reconstruct(dataset_dir):
    """
    This is the "Offline Reconstruction" tool.
    It takes a whole folder of images and turns them into a 3D model.
    """
    print(f"Starting 3D Build from folder: {dataset_dir}")
    poses_file = os.path.join(dataset_dir, "poses.txt")
    if not os.path.exists(poses_file):
        print("Error: poses.txt not found. Did you record a scan first?")
        return

    # Secret camera numbers (Intrinsics)
    width, height, fov = 800, 600, 1.047
    fx = (width / 2) / np.tan(fov / 2)
    fy = fx
    cx, cy = width / 2, height / 2
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # Create the "Volumetric" bucket (where we store all overlapping dots)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.005,
        sdf_trunc=0.02,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                idx = parts[0]
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
                
                # Math: Calculate exactly where the camera was in 3D space
                rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = rot
                extrinsic[:3, 3] = [tx, ty, tz]
                
                # Turn the matrix around (Open3D requirement)
                world_to_cam = np.linalg.inv(extrinsic)
                
                color_file = os.path.join(dataset_dir, f"image_{idx}.png")
                depth_file = os.path.join(dataset_dir, f"depth_{idx}.png")
                
                if os.path.exists(color_file) and os.path.exists(depth_file):
                    # Load the color and depth photos
                    color = o3d.io.read_image(color_file)
                    depth = o3d.io.read_image(depth_file)
                    
                    # Glue color and depth together into one "3D Photo"
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
                    
                    # Add this 3D photo to the main model
                    volume.integrate(rgbd, intrinsic, world_to_cam)
                    poses.append(idx)
                    
    if len(poses) > 1:
        # Show us how the first two photos match up
        extract_and_match_features(
            os.path.join(dataset_dir, f"image_{poses[0]}.png"),
            os.path.join(dataset_dir, f"image_{poses[1]}.png")
        )

    print("Step 1: Extracting the 3D Point Cloud (Many dots)...")
    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud("dense_point_cloud.ply", pcd)
    print("Saved dense_point_cloud.ply")
    
    print("Step 2: Connecting the dots to make a solid Skin (Mesh)...")
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # Use "Poisson Reconstruction" (Magic math that creates a smooth surface)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    o3d.io.write_triangle_mesh("reconstructed_mesh.obj", mesh)
    print("Saved reconstructed_mesh.obj (Use a 3D viewer to open this!)")
    
    print("FINISHED! You successfully built a 3D model.")

if __name__ == "__main__":
    import sys
    # Look for the dataset folder on the external SSD or local home
    dataset_path = "/media/thanushree/0679d7ea-20c8-4f40-a87c-9f188eef32cd/ur5_ws/dataset"
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    reconstruct(dataset_path)
