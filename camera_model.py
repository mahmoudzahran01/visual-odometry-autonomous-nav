"""
Camera model and image processing utilities for visual odometry.

This module provides functionality for camera calibration, 
image undistortion, and handling image sequences.
"""

import os
import numpy as np
import cv2

class CameraModel:
    """
    Camera model class that handles camera parameters and image processing.
    """
    
    def __init__(self, models_dir_path='./model', images_dir_path='./images'):
        """
        Initialize the camera model and load parameters.
        
        Args:
            models_dir_path: Path to camera model files
            images_dir_path: Path to image sequence
        """
        self.MODELS_DIR_PATH = models_dir_path
        self.IMAGES_DIR_PATH = images_dir_path
        
        # Load camera parameters
        self.fx, self.fy, self.cx, self.cy, self.G_camera_image, self.LUT = self._load_camera_model()
        
        # Initialize camera intrinsic matrix
        self.init_intrinsic()
        
        # Load and process image sequence
        self.processed_images = self.load_images()
    
    def _load_camera_model(self):
        """
        Load camera intrinsics and undistortion LUT.
        """
        # Use absolute paths
        models_dir_abs = os.path.abspath(self.MODELS_DIR_PATH)
        
        # Paths to camera model files
        intrinsics_path = os.path.join(models_dir_abs, "stereo_narrow_left.txt")
        lut_path = os.path.join(models_dir_abs, "stereo_narrow_left_distortion_lut.bin")
        
        print(f"Loading intrinsics from: {intrinsics_path}")
        print(f"File exists: {os.path.exists(intrinsics_path)}")
        
        try:
            # Try direct file reading first to verify access
            with open(intrinsics_path, 'r') as f:
                print(f"File can be opened and contains: {len(f.readlines())} lines")
            
            # Now load with numpy
            intrinsics = np.loadtxt(intrinsics_path)
            
            # Extract intrinsic parameters
            fx = intrinsics[0, 0]
            fy = intrinsics[0, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[0, 3]
            
            # Extract transformation matrix
            G_camera_image = intrinsics[1:5, 0:4]
            
            # Load undistortion lookup table
            print(f"Loading LUT from: {lut_path}")
            print(f"LUT file exists: {os.path.exists(lut_path)}")
            
            if not os.path.exists(lut_path):
                raise FileNotFoundError(f"LUT file {lut_path} not found")
            
            lut = np.fromfile(lut_path, np.double)
            lut = lut.reshape([2, lut.size//2])
            LUT = lut.transpose()
            
            return fx, fy, cx, cy, G_camera_image, LUT
            
        except Exception as e:
            print(f"Error during camera model loading: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise
        
    def init_intrinsic(self):
        """
        Initialize camera intrinsic matrix.
        """
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    def load_images(self, max_images=25):
        """
        Load and process images from the images directory.
        
        Args:
            max_images: Maximum number of images to load
            
        Returns:
            List of processed images
        """
        try:
            image_names = sorted(os.listdir(self.IMAGES_DIR_PATH))
            processed_images = []
            
            for image_name in image_names[:max_images]:
                image_path = os.path.join(self.IMAGES_DIR_PATH, image_name)
                processed_images.append(self.load_and_process_image(image_path))
            
            return np.array(processed_images)
        
        except Exception as e:
            print(f"Error loading images: {e}")
            return []
    
    def load_and_process_image(self, image_path):
        """
        Load an image and apply necessary processing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Processed image
        """
        # Load raw image
        bayer_img = cv2.imread(image_path, flags=-1)
        
        # Convert from Bayer pattern to BGR
        bgr_image = cv2.cvtColor(bayer_img, cv2.COLOR_BayerGR2BGR)
        
        # Undistort the image
        undistorted_image = self.undistort_image(bgr_image)
        
        return undistorted_image
    
    def undistort_image(self, image):
        """
        Undistort an image using the lookup table.
        
        Args:
            image: Distorted image
            
        Returns:
            Undistorted image
        """
        # Create an empty output image of the same size and type
        undistorted = np.zeros_like(image)
        
        # Get image dimensions
        height, width, channels = image.shape
        
        # Create coordinate grids for the undistorted image
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        
        # Map coordinates from undistorted to distorted image using LUT
        map_x = self.LUT[:, 0].reshape(height, width)
        map_y = self.LUT[:, 1].reshape(height, width)
        
        # Apply remapping using OpenCV
        undistorted = cv2.remap(image, map_x.astype(np.float32), 
                                 map_y.astype(np.float32), 
                                 interpolation=cv2.INTER_LINEAR)
        
        return undistorted