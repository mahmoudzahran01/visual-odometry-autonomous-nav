"""
Fundamental matrix estimation using RANSAC.

This module implements the normalized 8-point algorithm with RANSAC
for robust fundamental matrix estimation from point correspondences.
"""

import numpy as np
import random

class RANSACFundamentalMatrix:
    """
    Fundamental matrix estimation using RANSAC.
    """
    
    def __init__(self, x1, y1, x2, y2, confidence=0.99, outlier_ratio=0.5):
        """
        Initialize the RANSAC fundamental matrix estimator.
        
        Args:
            x1, y1: Coordinates in the first image
            x2, y2: Coordinates in the second image
            confidence: RANSAC confidence level (typically 0.95-0.99)
            outlier_ratio: Expected ratio of outliers in the data
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
        self.num_points = len(x1)
        
        # Compute number of RANSAC iterations
        points_needed = 8  # 8-point algorithm
        self.num_iterations = self._compute_ransac_iterations(confidence, outlier_ratio, points_needed)
    
    def _compute_ransac_iterations(self, confidence, outlier_ratio, sample_size):
        """
        Compute the number of RANSAC iterations needed.
        
        Args:
            confidence: Desired probability of success
            outlier_ratio: Expected ratio of outliers
            sample_size: Number of points needed for model estimation
            
        Returns:
            Number of iterations
        """
        inlier_ratio = 1.0 - outlier_ratio
        # Formula: log(1-confidence) / log(1-(inlier_ratio^sample_size))
        num_iter = np.log(1.0 - confidence) / np.log(1.0 - (inlier_ratio ** sample_size))
        return int(np.ceil(num_iter))
    
    @staticmethod
    def normalize_points(points):
        """
        Normalize points to improve numerical stability.
        
        Args:
            points: Points to normalize [x, y, 1]
            
        Returns:
            normalized_points: Normalized points
            T: Transformation matrix
        """
        # Compute centroid
        centroid = np.mean(points[:, :2], axis=0)
        
        # Compute average distance from centroid
        centered_points = points[:, :2] - centroid
        avg_distance = np.mean(np.sqrt(np.sum(centered_points**2, axis=1)))
        
        # Scale factor to make average distance sqrt(2)
        scale = np.sqrt(2) / (avg_distance + 1e-10)
        
        # Create transformation matrix
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        
        # Apply transformation
        normalized_points = np.dot(T, points.T).T
        
        return normalized_points, T
    
    def compute_fundamental_matrix(self, point_indices):
        """
        Compute fundamental matrix using the 8-point algorithm.
        
        Args:
            point_indices: Indices of points to use
            
        Returns:
            F: Fundamental matrix
        """
        # Extract points
        selected_x1 = self.x1[point_indices]
        selected_y1 = self.y1[point_indices]
        selected_x2 = self.x2[point_indices]
        selected_y2 = self.y2[point_indices]
        
        # Create homogeneous coordinates
        points1 = np.hstack((selected_x1, selected_y1, np.ones_like(selected_x1)))
        points2 = np.hstack((selected_x2, selected_y2, np.ones_like(selected_x2)))
        
        # Normalize points
        normalized_points1, T1 = self.normalize_points(points1)
        normalized_points2, T2 = self.normalize_points(points2)
        
        # Build constraint matrix A
        A = np.zeros((len(point_indices), 9))
        
        for i in range(len(point_indices)):
            x1, y1, _ = normalized_points1[i]
            x2, y2, _ = normalized_points2[i]
            
            A[i] = [
                x1*x2, x1*y2, x1,
                y1*x2, y1*y2, y1,
                x2, y2, 1
            ]
        
        # Solve for F using SVD
        _, _, V = np.linalg.svd(A)
        F = V[-1].reshape(3, 3)
        
        # Enforce rank-2 constraint
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ V
        
        # Denormalize
        F = T2.T @ F @ T1
        
        return F
    
    def compute_epipolar_error(self, F, x1, y1, x2, y2):
        """
        Compute epipolar line distance error.
        
        Args:
            F: Fundamental matrix
            x1, y1, x2, y2: Point coordinates
            
        Returns:
            error: Point-to-line distances
        """
        # Create homogeneous coordinates
        points1 = np.hstack((x1, y1, np.ones_like(x1)))
        points2 = np.hstack((x2, y2, np.ones_like(x2)))
        
        # Compute epipolar lines
        lines2 = F @ points1.T  # Lines in image 2
        lines1 = F.T @ points2.T  # Lines in image 1
        
        # Compute point-to-line distances
        # Distance = |ax + by + c| / sqrt(a^2 + b^2)
        dist1 = np.abs(np.sum(points1 * (lines1.T), axis=1)) / np.sqrt(lines1[0]**2 + lines1[1]**2 + 1e-10)
        dist2 = np.abs(np.sum(points2 * (lines2.T), axis=1)) / np.sqrt(lines2[0]**2 + lines2[1]**2 + 1e-10)
        
        # Sum of distances (Sampson error)
        total_error = dist1 + dist2
        
        return total_error
    
    def find_inliers(self, F, threshold):
        """
        Find inliers based on epipolar constraint.
        
        Args:
            F: Fundamental matrix
            threshold: Error threshold for inliers
            
        Returns:
            inlier_mask: Boolean mask of inliers
        """
        errors = self.compute_epipolar_error(F, self.x1, self.y1, self.x2, self.y2)
        inlier_mask = errors < threshold
        
        return inlier_mask
    
    def estimate_fundamental_matrix(self, threshold=0.01):
        """
        Estimate fundamental matrix using RANSAC.
        
        Args:
            threshold: Error threshold for inliers
            
        Returns:
            best_F: Best fundamental matrix
            inlier_points1: Inlier points in the first image
            inlier_points2: Inlier points in the second image
        """
        if self.num_points < 8:
            raise ValueError("Need at least 8 point correspondences")
        
        max_inliers = 0
        best_F = None
        best_inlier_mask = None
        
        for _ in range(self.num_iterations):
            # Randomly select 8 points
            sample_indices = random.sample(range(self.num_points), 8)
            
            # Compute fundamental matrix
            F = self.compute_fundamental_matrix(sample_indices)
            
            # Find inliers
            inlier_mask = self.find_inliers(F, threshold)
            inlier_count = np.sum(inlier_mask)
            
            # Update best model if more inliers found
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_F = F
                best_inlier_mask = inlier_mask
        
        # Refine model using all inliers if a good model was found
        if best_F is not None and max_inliers >= 8:
            inlier_indices = np.where(best_inlier_mask)[0]
            best_F = self.compute_fundamental_matrix(inlier_indices)
            best_inlier_mask = self.find_inliers(best_F, threshold)
        
        # Extract inlier points
        inlier_indices = np.where(best_inlier_mask)[0]
        
        inlier_points1 = np.hstack((
            self.x1[inlier_indices],
            self.y1[inlier_indices],
            np.ones((len(inlier_indices), 1))
        ))
        
        inlier_points2 = np.hstack((
            self.x2[inlier_indices],
            self.y2[inlier_indices],
            np.ones((len(inlier_indices), 1))
        ))
        
        return best_F, inlier_points1, inlier_points2