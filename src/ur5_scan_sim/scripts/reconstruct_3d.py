#!/usr/bin/env python3

import os
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

def extract_and_match_features(image1_path, image2_path):
    print(f"Extracting ORB features for {image1_path} and {image2_path}...")
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        print(f"-> Found {len(matches)} matches")

        # Draw first 50 matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite("feature_matches.png", img_matches)
        print("-> Saved feature_matches.png")

def reconstruct(dataset_dir):
    print(f"Starting reconstruction from {dataset_dir}")
    poses_file = os.path.join(dataset_dir, "poses.txt")
    if not os.path.exists(poses_file):
        print("poses.txt not found. Cannot reconstruct.")
        return

    # Camera intrinsics (approximate based on gazebo 1.047 fov and 800x600)
    width = 800
    height = 600
    fov = 1.047
    fx = (width / 2) / np.tan(fov / 2)
    fy = fx
    cx = width / 2
    cy = height / 2
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
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
                
                # Construct transformation matrix
                rot = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = rot
                extrinsic[:3, 3] = [tx, ty, tz]
                
                # Invert extrinsic because Open3D expects world-to-camera matrix
                # world_to_camera = np.linalg.inv(camera_to_world)
                world_to_cam = np.linalg.inv(extrinsic)
                
                color_file = os.path.join(dataset_dir, f"image_{idx}.png")
                depth_file = os.path.join(dataset_dir, f"depth_{idx}.png")
                
                if os.path.exists(color_file) and os.path.exists(depth_file):
                    color = o3d.io.read_image(color_file)
                    depth = o3d.io.read_image(depth_file)
                    
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
                    
                    volume.integrate(rgbd, intrinsic, world_to_cam)
                    poses.append(idx)
                    
    if len(poses) > 1:
        # Just match the first two images to show feature extraction
        extract_and_match_features(
            os.path.join(dataset_dir, f"image_{poses[0]}.png"),
            os.path.join(dataset_dir, f"image_{poses[1]}.png")
        )

    print("Extracting Point Cloud...")
    pcd = volume.extract_point_cloud()
    
    # Save sparse/dense point cloud
    o3d.io.write_point_cloud("dense_point_cloud.ply", pcd)
    print("Saved dense_point_cloud.ply")
    
    print("Generating Surface Mesh using Poisson reconstruction...")
    # Estimate normals for poisson
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    # Crop mesh if needed, or just save
    o3d.io.write_triangle_mesh("reconstructed_mesh.obj", mesh)
    print("Saved reconstructed_mesh.obj")
    
    print("Reconstruction Complete!")

if __name__ == "__main__":
    import sys
    dataset_path = "/media/thanushree/0679d7ea-20c8-4f40-a87c-9f188eef32cd/ur5_ws/dataset"
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    reconstruct(dataset_path)
