"""
Feature detection and matching for visual odometry.

This module implements feature detection and matching between consecutive frames
using SIFT features and ratio test for matching.
"""

import numpy as np
import cv2

class FeatureMatcher:
    """
    Feature matcher class using SIFT features.
    """
    
    def __init__(self, img1, img2, ratio_threshold=0.8):
        """
        Initialize the feature matcher and find matches between two images.
        
        Args:
            img1: First image
            img2: Second image
            ratio_threshold: Ratio test threshold for SIFT matching
        """
        self.img1 = img1
        self.img2 = img2
        self.ratio_threshold = ratio_threshold
        self.correspondences = self._find_correspondences()
    
    def _find_correspondences(self):
        """
        Find corresponding points between two images using SIFT features.
        
        Returns:
            Array of point correspondences [x1, y1, x2, y2]
        """
        # Convert images to grayscale if they are color
        if len(self.img1.shape) == 3:
            gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = self.img1
            gray2 = self.img2
        
        # 1. Detect SIFT keypoints and compute descriptors
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        # Convert keypoints to numpy arrays
        if len(keypoints1) == 0 or len(keypoints2) == 0:
            return np.array([])
        
        kp1_array = cv2.KeyPoint_convert(keypoints1)
        kp2_array = cv2.KeyPoint_convert(keypoints2)
        
        # 2. Match descriptors using brute force matcher
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        # 3. Apply ratio test to filter good matches
        good_matches = []
        for m, n in raw_matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
        
        # Extract point coordinates from matches
        if len(good_matches) == 0:
            return np.array([])
        
        correspondences = []
        for match in good_matches:
            queryIdx = match.queryIdx
            trainIdx = match.trainIdx
            
            # Get coordinates of matching points
            x1, y1 = kp1_array[queryIdx]
            x2, y2 = kp2_array[trainIdx]
            
            correspondences.append([x1, y1, x2, y2])
        
        return np.array(correspondences)
    
    def get_correspondences(self):
        """
        Get the computed point correspondences.
        
        Returns:
            Array of point correspondences [x1, y1, x2, y2]
        """
        return self.correspondences
    
    def get_point_arrays(self):
        """
        Get the matched points as separate arrays.
        
        Returns:
            x1, y1, x2, y2: Arrays of coordinates for matched points
        """
        if len(self.correspondences) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        x1 = self.correspondences[:, 0].reshape(-1, 1)
        y1 = self.correspondences[:, 1].reshape(-1, 1)
        x2 = self.correspondences[:, 2].reshape(-1, 1)
        y2 = self.correspondences[:, 3].reshape(-1, 1)
        
        return x1, y1, x2, y2
    
    def display_matches(self, max_points=50):
        """
        Create a visualization of the matches between two images.
        
        Args:
            max_points: Maximum number of matches to display
            
        Returns:
            Image with match lines drawn between the two input images
        """
        # Create a combined image
        h1, w1 = self.img1.shape[:2]
        h2, w2 = self.img2.shape[:2]
        
        max_height = max(h1, h2)
        combined_width = w1 + w2
        
        # Create output image
        combined_img = np.zeros((max_height, combined_width, 3), dtype=np.uint8)
        combined_img[:h1, :w1] = self.img1[:, :, :3] if self.img1.shape[2] >= 3 else cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
        combined_img[:h2, w1:w1+w2] = self.img2[:, :, :3] if self.img2.shape[2] >= 3 else cv2.cvtColor(self.img2, cv2.COLOR_GRAY2BGR)
        
        if len(self.correspondences) == 0:
            return combined_img
        
        # Draw matches
        num_matches = min(max_points, len(self.correspondences))
        indices = np.random.choice(len(self.correspondences), num_matches, replace=False)
        
        for idx in indices:
            x1, y1, x2, y2 = self.correspondences[idx]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2) + w1, int(y2))
            
            # Draw line between matches
            cv2.line(combined_img, pt1, pt2, (0, 255, 0), 1)
            
            # Draw points
            cv2.circle(combined_img, pt1, 3, (0, 0, 255), -1)
            cv2.circle(combined_img, pt2, 3, (0, 0, 255), -1)
        
        return combined_img