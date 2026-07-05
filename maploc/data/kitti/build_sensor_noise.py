#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build KITTI dataset noise variants
Supports three types of noise: underexposure, overexposure, and motion blur
"""

import os
import glob
import shutil
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def create_underexposed_image(image, factor=0.5):
    """
    Create underexposed image variant
    
    Parameters:
        image: Input image
        factor: Brightness reduction factor (0.0-1.0)
    """
    # Convert to HSV color space to manipulate brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    # Reduce brightness channel (V)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    # Ensure values are in valid range
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    # Convert back to BGR
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def create_overexposed_image(image, factor=1.5):
    """
    Create overexposed image variant
    
    Parameters:
        image: Input image
        factor: Brightness increase factor (>1.0)
    """
    # Convert to HSV color space to manipulate brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    # Increase brightness channel (V)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    # Ensure values are in valid range
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    # Convert back to BGR
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def create_motion_blur(image, kernel_size=15):
    """
    Create motion blur image variant
    
    Parameters:
        image: Input image
        kernel_size: Convolution kernel size, controls blur intensity
    """
    # Create motion blur convolution kernel (horizontal direction)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = 1.0 / kernel_size
    # Apply blur
    return cv2.filter2D(image, -1, kernel)


def copy_directory_structure(src_dir, dest_dir):
    """Copy directory structure, excluding files"""
    for dirpath, dirnames, _ in os.walk(src_dir):
        # Calculate relative path
        structure = os.path.relpath(dirpath, src_dir)
        if structure == ".":
            continue
        for noise_type in ["under_exposure", "over_exposure", "motion_blur"]:
            # Create corresponding directory structure for each noise type
            new_dir = os.path.join(dest_dir, noise_type, structure)
            os.makedirs(new_dir, exist_ok=True)


def copy_non_image_files(src_dir, dest_dir):
    """Copy all non-image files (like calibration data, GPS data, etc.) to all noise variant directories"""
    non_image_files = []
    
    # Find non-image files
    for dirpath, _, filenames in os.walk(src_dir):
        for filename in filenames:
            # Skip image files
            if not filename.endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, src_dir)
                non_image_files.append(rel_path)

    # Copy each non-image file to each noise type directory
    for noise_type in ["under_exposure", "over_exposure", "motion_blur"]:
        for file_path in tqdm(non_image_files, desc=f"Copying non-image files for {noise_type}"):
            src_path = os.path.join(src_dir, file_path)
            dest_path = os.path.join(dest_dir, noise_type, file_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(src_path, dest_path)  # copy2 preserves metadata


def process_image(args):
    """Process a single image and create its noise variants"""
    image_path, src_dir, dest_dir, under_factor, over_factor, blur_size = args
    
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Get relative path to maintain directory structure
        rel_path = os.path.relpath(image_path, src_dir)
        
        # Create and save corresponding image for each noise type
        noise_variants = {
            "under_exposure": create_underexposed_image(image, under_factor),
            "over_exposure": create_overexposed_image(image, over_factor),
            "motion_blur": create_motion_blur(image, blur_size)
        }
        
        for noise_type, noisy_image in noise_variants.items():
            target_path = os.path.join(dest_dir, noise_type, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            cv2.imwrite(target_path, noisy_image)
            
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


def build_noisy_dataset(src_dir, dest_dir, num_workers=8, under_factor=0.5, over_factor=1.8, blur_size=15):
    """Build complete noise variant dataset"""
    print(f"Building noise variant dataset from {src_dir} to {dest_dir}")
    
    # Create basic directory structure
    os.makedirs(dest_dir, exist_ok=True)
    for noise_type in ["under_exposure", "over_exposure", "motion_blur"]:
        os.makedirs(os.path.join(dest_dir, noise_type), exist_ok=True)
    
    print("Copying directory structure...")
    copy_directory_structure(src_dir, dest_dir)
    
    print("Copying non-image files...")
    copy_non_image_files(src_dir, dest_dir)
    
    # Get all image files
    print("Finding image files...")
    image_files = []
    for camera_idx in [0, 1, 2, 3]:  # KITTI has 4 cameras
        camera_pattern = f"image_{camera_idx:02}/data/*.png"
        found_images = glob.glob(os.path.join(src_dir, "**", camera_pattern), recursive=True)
        image_files.extend(found_images)
    
    print(f"Found {len(image_files)} image files, starting processing...")
    
    # Prepare processing parameters
    process_args = [(img, src_dir, dest_dir, under_factor, over_factor, blur_size) 
                    for img in image_files]
    
    # Process images in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(process_image, process_args), total=len(process_args), desc="Processing images"))
    
    # Create marker files to indicate dataset is downloaded/prepared
    for noise_type in ["under_exposure", "over_exposure", "motion_blur"]:
        with open(os.path.join(dest_dir, noise_type, ".downloaded"), "w") as f:
            f.write("This directory contains a prepared noisy variant of the KITTI dataset.")
    
    print("Noise variant dataset build complete!")


def main():
    parser = argparse.ArgumentParser(description="Build noise variants of KITTI dataset")
    parser.add_argument("--src", required=True, help="Path to original KITTI dataset")
    parser.add_argument("--dest", required=True, help="Output path for noise variant dataset")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel processing workers")
    parser.add_argument("--under-factor", type=float, default=0.5, help="Underexposure factor (0-1)")
    parser.add_argument("--over-factor", type=float, default=1.5, help="Overexposure factor (>1)")
    parser.add_argument("--blur-size", type=int, default=15, help="Blur kernel size")
    
    args = parser.parse_args()
    
    build_noisy_dataset(
        args.src, 
        args.dest, 
        args.workers,
        args.under_factor,
        args.over_factor,
        args.blur_size
    )


if __name__ == "__main__":
    main()
