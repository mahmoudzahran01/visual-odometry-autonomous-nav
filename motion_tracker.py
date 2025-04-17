"""
Visual odometry motion tracker.

This module implements the main motion tracking system that recovers 
camera motion from image sequences using epipolar geometry.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from camera_model import CameraModel
from feature_matcher import FeatureMatcher
from ransac_fundamental_matrix import RANSACFundamentalMatrix

class MotionTracker:
    """
    Motion tracker class that recovers camera motion from image sequences.
    """
    
    def __init__(self, models_dir='./model', images_dir='./images', output_dir='./results'):
        """
        Initialize the motion tracker.
        
        Args:
            models_dir: Path to camera model files
            images_dir: Path to image sequence
            output_dir: Path to output directory for results
        """
        self.models_dir = os.path.abspath(models_dir)
        self.images_dir = os.path.abspath(images_dir)
        self.output_dir = os.path.abspath(output_dir)
        
        print(f"Absolute models directory: {self.models_dir}")
        print(f"Absolute images directory: {self.images_dir}")
        print(f"Absolute output directory: {self.output_dir}")
        
        # Create output directory and subdirectories if they don't exist
        self._create_output_directories()
        
        self.camera_model = CameraModel(self.models_dir, self.images_dir)
        self.processed_images = self.camera_model.processed_images
        self.K = self.camera_model.K
        self.trajectory = []
    
    def _create_output_directories(self):
        """
        Create output directory structure.
        """
        # Main output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
        
        # Matches subdirectory
        self.matches_dir = os.path.join(self.output_dir, "matches")
        if not os.path.exists(self.matches_dir):
            os.makedirs(self.matches_dir)
            print(f"Created matches directory: {self.matches_dir}")
    
    def track_motion(self, max_frames=None):
        """
        Track camera motion through the image sequence.
        
        Args:
            max_frames: Maximum number of frames to process
            
        Returns:
            trajectory: Camera trajectory as a list of poses
        """
        if self.processed_images is None or len(self.processed_images) == 0:
            print("No images to process")
            return []
        
        num_images = len(self.processed_images)
        if max_frames is not None:
            num_images = min(num_images, max_frames)
        
        # Starting pose (identity)
        curr_pose = np.eye(4)
        self.trajectory = [curr_pose[:3, 3]]  # Store camera positions
        
        # Store camera positions for plotting
        xs, ys, zs = [], [], []
        
        # Process each pair of consecutive frames
        for i in range(num_images - 1):
            print(f"Processing frames {i} and {i+1}")
            frame1, frame2 = self.processed_images[i], self.processed_images[i+1]
            
            # Find feature matches
            matcher = FeatureMatcher(frame1, frame2)
            correspondences = matcher.get_correspondences()
            x1, y1, x2, y2 = matcher.get_point_arrays()
            
            if len(correspondences) < 8:
                print(f"Not enough matches found between frames {i} and {i+1}")
                continue
            
            # Display matches
            match_img = matcher.display_matches()
            matches_path = os.path.join(self.matches_dir, f"matches_{i}_{i+1}.jpg")
            cv2.imwrite(matches_path, match_img)
            print(f"Saved matches to {matches_path}")
            
            # Estimate fundamental matrix
            ransac_F = RANSACFundamentalMatrix(x1, y1, x2, y2)
            F, frame1_inliers, frame2_inliers = ransac_F.estimate_fundamental_matrix()
            
            if F is None:
                print(f"Failed to estimate fundamental matrix for frames {i} and {i+1}")
                continue
            
            # Compute essential matrix
            E = self.compute_essential_matrix(F)
            
            # Recover pose from essential matrix
            success, R, t = self.recover_pose(E, frame1_inliers, frame2_inliers)
            
            if not success:
                print(f"Failed to recover pose for frames {i} and {i+1}")
                continue
            
            # Create pose matrix
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.flatten()
            
            # Update current pose
            curr_pose = curr_pose @ pose
            
            # Store trajectory point
            cam_position = curr_pose[:3, 3]
            self.trajectory.append(cam_position)
            
            # Store for plotting
            xs.append(cam_position[0])
            ys.append(cam_position[1])
            zs.append(cam_position[2])
            
            print(f"Camera position: {cam_position}")
        
        # Plot trajectory
        self.plot_trajectory(xs, ys, zs)
        
        return self.trajectory
    
    def compute_essential_matrix(self, F):
        """
        Compute the essential matrix from the fundamental matrix.
        
        Args:
            F: Fundamental matrix
            
        Returns:
            E: Essential matrix
        """
        E = self.K.T @ F @ self.K
        
        # Enforce the constraint that 2 singular values are equal and 1 is zero
        U, S, Vt = np.linalg.svd(E)
        S = np.array([1, 1, 0])  # Force singular values to [1, 1, 0]
        E = U @ np.diag(S) @ Vt
        
        return E
    
    def decompose_essential_matrix(self, E):
        """
        Decompose the essential matrix into rotation and translation.
        
        Args:
            E: Essential matrix
            
        Returns:
            all_poses: List of possible pose configurations [R|t]
        """
        U, _, Vt = np.linalg.svd(E)
        
        # Ensure proper rotation matrix (det(R) = 1)
        if np.linalg.det(U @ Vt) < 0:
            Vt = -Vt
        
        # Two possible rotations
        W = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        
        # Two possible translations
        t1 = U[:, 2].reshape(3, 1)
        t2 = -t1
        
        # Four possible pose configurations
        poses = [
            np.hstack((R1, t1)),
            np.hstack((R1, t2)),
            np.hstack((R2, t1)),
            np.hstack((R2, t2))
        ]
        
        return poses
    
    def triangulate_point(self, P1, P2, point1, point2):
        """
        Triangulate a 3D point from two 2D points and camera projection matrices.
        
        Args:
            P1, P2: Camera projection matrices
            point1, point2: 2D points in homogeneous coordinates
            
        Returns:
            point_3d: Triangulated 3D point
        """
        # Create the system of equations
        A = np.zeros((4, 4))
        A[0] = point1[0] * P1[2] - P1[0]
        A[1] = point1[1] * P1[2] - P1[1]
        A[2] = point2[0] * P2[2] - P2[0]
        A[3] = point2[1] * P2[2] - P2[1]
        
        # Solve the system using SVD
        _, _, Vt = np.linalg.svd(A)
        point_3d = Vt[-1]
        
        # Convert to homogeneous coordinates
        point_3d = point_3d / point_3d[3]
        
        return point_3d[:3]
    
    def check_point_in_front_of_camera(self, P, point_3d):
        """
        Check if a 3D point is in front of the camera.
        
        Args:
            P: Camera projection matrix
            point_3d: 3D point
            
        Returns:
            in_front: True if point is in front of camera
        """
        # Extract rotation matrix and camera center
        R = P[:, :3]
        t = P[:, 3]
        
        # Camera center is -R^T * t
        C = -np.linalg.inv(R) @ t
        
        # Vector from camera center to 3D point
        vector = point_3d - C
        
        # Check if point is in front of camera (positive z)
        # Third row of R is the camera's optical axis
        z_component = np.dot(R[2], vector)
        
        return z_component > 0
    
    def recover_pose(self, E, points1, points2):
        """
        Recover the camera pose from the essential matrix.
        
        Args:
            E: Essential matrix
            points1, points2: Corresponding points in homogeneous coordinates
            
        Returns:
            success: True if pose recovery succeeded
            R: Rotation matrix
            t: Translation vector
        """
        # Decompose essential matrix into possible rotations and translations
        poses = self.decompose_essential_matrix(E)
        
        # Reference projection matrix (identity pose)
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        
        best_pose = None
        max_points_in_front = 0
        
        # Check all possible poses
        for pose in poses:
            R, t = pose[:, :3], pose[:, 3]
            P2 = np.hstack((R, t.reshape(3, 1)))
            
            # Count points in front of both cameras
            points_in_front = 0
            
            for i in range(len(points1)):
                # Normalize points using calibration matrix
                pt1 = np.linalg.inv(self.K) @ points1[i]
                pt2 = np.linalg.inv(self.K) @ points2[i]
                
                # Triangulate the 3D point
                point_3d = self.triangulate_point(P1, P2, pt1, pt2)
                
                # Check if point is in front of both cameras
                if self.check_point_in_front_of_camera(P1, point_3d) and \
                   self.check_point_in_front_of_camera(P2, point_3d):
                    points_in_front += 1
            
            # Update best pose if more points are in front
            if points_in_front > max_points_in_front:
                max_points_in_front = points_in_front
                best_pose = pose
        
        # If no points are in front of both cameras, pose recovery failed
        if max_points_in_front == 0:
            return False, None, None
        
        # Extract rotation and translation from best pose
        R = best_pose[:, :3]
        t = best_pose[:, 3]
        
        return True, R, t
    
    def plot_trajectory(self, xs, ys, zs):
        """
        Plot the camera trajectory.
        
        Args:
            xs, ys, zs: Camera position coordinates
        """
        plt.figure(figsize=(10, 8))
        
        # Plot camera trajectory
        plt.plot(xs, zs, 'b-', linewidth=2)
        plt.scatter(xs, zs, c='red', s=50)
        
        # Mark start and end points
        if len(xs) > 0:
            plt.scatter(xs[0], zs[0], c='green', s=100, marker='^', label='Start')
            plt.scatter(xs[-1], zs[-1], c='black', s=100, marker='o', label='End')
        
        plt.xlabel('X Position')
        plt.ylabel('Z Position')
        plt.title('Camera Trajectory (Top View)')
        plt.grid(True)
        plt.legend()
        
        # Save to output directory (main results folder, not in matches)
        trajectory_path = os.path.join(self.output_dir, 'camera_trajectory.png')
        plt.savefig(trajectory_path)
        print(f"Saved trajectory plot to {trajectory_path}")
        plt.show()
        
        # Save trajectory data
        trajectory_data_path = os.path.join(self.output_dir, 'trajectory.npy')
        np.save(trajectory_data_path, np.column_stack((xs, ys, zs)))
        print(f"Saved trajectory data to {trajectory_data_path}")