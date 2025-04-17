"""
Main entry point for the visual odometry system.

This script runs the visual odometry pipeline on a sequence of images.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from motion_tracker import MotionTracker

def main():
    parser = argparse.ArgumentParser(description='Visual Odometry System')
    parser.add_argument('--models', type=str, default='./model', 
                        help='Path to camera model files')
    parser.add_argument('--images', type=str, default='./images', 
                        help='Path to image sequence')
    parser.add_argument('--max_frames', type=int, default=None, 
                        help='Maximum number of frames to process')
    parser.add_argument('--output', type=str, default='./results', 
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    print(f"Processing images from: {args.images}")
    print(f"Using camera models from: {args.models}")
    print(f"Saving results to: {args.output}")
    
    # Initialize and run motion tracker
    tracker = MotionTracker(args.models, args.images)
    
    # DO NOT change directory - handle output paths explicitly instead
    # os.chdir(args.output)
    
    trajectory = tracker.track_motion(args.max_frames)
    
    print(f"Processed {len(trajectory)-1} frame pairs")
    print("Visual odometry completed successfully!")
    

if __name__ == "__main__":
    import os
    print(f"Current working directory: {os.getcwd()}")
    main()