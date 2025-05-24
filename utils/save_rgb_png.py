#!/usr/bin/env python3
"""
Utility script to extract and save RGB images from RGBD zarr datasets as PNG files

Usage:
    python utils/save_rgb_png.py --dataset path/to/dataset.zarr --output path/to/output/dir
    python utils/save_rgb_png.py --dataset path/to/dataset.zarr --max_samples 20
    python utils/save_rgb_png.py --dataset_dir demo_data/ --all  # Process all RGBD datasets
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import zarr
from pathlib import Path

def save_rgb_images_from_zarr(dataset_path, output_dir=None, max_samples=10, create_summary=True):
    """
    Extract and save RGB images from a zarr dataset as PNG files
    
    Args:
        dataset_path (str): Path to the zarr dataset
        output_dir (str): Output directory for PNG files
        max_samples (int): Maximum number of samples to save
        create_summary (bool): Whether to create a summary grid image
    """
    try:
        # Load zarr dataset
        data = zarr.open(dataset_path, 'r')
        
        if 'data' not in data:
            print(f"❌ No data found in {dataset_path}")
            return False
        
        if 'rgb_images' not in data['data']:
            print(f"❌ No RGB images found in {dataset_path}")
            return False
        
        rgb_data = data['data']['rgb_images']
        depth_data = data['data']['depth_images'] if 'depth_images' in data['data'] else None
        
        # Get dataset info
        dataset_name = Path(dataset_path).stem
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(dataset_path), f"{dataset_name}_png_export")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample indices
        total_steps = rgb_data.shape[0]
        if total_steps == 0:
            print(f"⚠️ No RGB data to save from {dataset_path}")
            return False
        
        sample_indices = np.linspace(0, total_steps-1, min(max_samples, total_steps), dtype=int)
        
        print(f"💾 Saving {len(sample_indices)} RGB images from {dataset_name}...")
        print(f"   RGB shape: {rgb_data.shape}")
        print(f"   Output dir: {output_dir}")
        
        saved_count = 0
        
        for i, step_idx in enumerate(sample_indices):
            try:
                rgb_step = rgb_data[step_idx]  # Shape: [num_cameras, height, width, 3]
                
                if rgb_step.size == 0 or (len(rgb_step.shape) >= 2 and rgb_step.shape[0] == 0):
                    continue
                
                num_cameras = rgb_step.shape[0] if len(rgb_step.shape) > 1 else 1
                
                # Skip if this looks like dummy data (1x1 images)
                if len(rgb_step.shape) >= 3 and (rgb_step.shape[1] <= 1 or rgb_step.shape[2] <= 1):
                    continue
                
                # Create figure for this time step
                if depth_data is not None:
                    fig, axes = plt.subplots(2, num_cameras, figsize=(4*num_cameras, 8))
                    if num_cameras == 1:
                        axes = axes.reshape(2, 1)
                else:
                    fig, axes = plt.subplots(1, num_cameras, figsize=(4*num_cameras, 4))
                    if num_cameras == 1:
                        axes = [axes]
                
                for cam_idx in range(num_cameras):
                    try:
                        # Get RGB image for this camera
                        if len(rgb_step.shape) >= 4:  # [num_cameras, height, width, 3]
                            rgb_img = rgb_step[cam_idx]
                        else:  # Single camera case
                            rgb_img = rgb_step
                        
                        # Normalize RGB if needed
                        if rgb_img.dtype == np.uint8:
                            rgb_img_norm = rgb_img.astype(np.float32) / 255.0
                        else:
                            rgb_img_norm = np.clip(rgb_img, 0, 1)
                        
                        # Plot RGB image
                        ax_idx = cam_idx if depth_data is None else (0, cam_idx)
                        if isinstance(ax_idx, tuple):
                            ax = axes[ax_idx[0], ax_idx[1]] if num_cameras > 1 else axes[ax_idx[0]]
                        else:
                            ax = axes[ax_idx] if num_cameras > 1 else axes
                        
                        ax.imshow(rgb_img_norm)
                        ax.set_title(f"RGB Camera {cam_idx} (Step {step_idx})")
                        ax.axis('off')
                        
                        # Plot depth image if available
                        if depth_data is not None:
                            depth_step = depth_data[step_idx]
                            if len(depth_step.shape) >= 3:
                                depth_img = depth_step[cam_idx]
                            else:
                                depth_img = depth_step
                            
                            ax_depth = axes[1, cam_idx] if num_cameras > 1 else axes[1]
                            
                            if depth_img.size > 1:  # Not dummy data
                                im = ax_depth.imshow(depth_img, cmap='viridis')
                                plt.colorbar(im, ax=ax_depth)
                            else:
                                ax_depth.imshow(np.zeros_like(rgb_img_norm[:,:,0]), cmap='viridis')
                            ax_depth.set_title(f"Depth Camera {cam_idx} (Step {step_idx})")
                            ax_depth.axis('off')
                        
                    except Exception as e:
                        print(f"⚠️ Error processing camera {cam_idx} at step {step_idx}: {e}")
                        continue
                
                plt.tight_layout()
                png_path = os.path.join(output_dir, f"sample_{i:03d}_step_{step_idx:04d}.png")
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()
                saved_count += 1
                
            except Exception as e:
                print(f"⚠️ Error processing step {step_idx}: {e}")
                continue
        
        print(f"✅ Saved {saved_count} PNG images to {output_dir}")
        
        # Create summary grid
        if create_summary and saved_count > 0:
            create_summary_grid(rgb_data, sample_indices, output_dir, dataset_name)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {dataset_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_summary_grid(rgb_data, sample_indices, output_dir, dataset_name):
    """Create a summary grid of RGB images"""
    try:
        # Create a grid layout
        n_samples = len(sample_indices)
        cols = min(5, n_samples)  # Max 5 columns
        rows = (n_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        fig.suptitle(f"RGB Images Summary - {dataset_name}", fontsize=16)
        
        if n_samples == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, step_idx in enumerate(sample_indices):
            if i < len(axes):
                try:
                    rgb_step = rgb_data[step_idx]
                    
                    # Use first camera if multiple cameras
                    if len(rgb_step.shape) >= 4:  # [num_cameras, height, width, 3]
                        rgb_img = rgb_step[0]
                    else:
                        rgb_img = rgb_step
                    
                    # Skip dummy data
                    if rgb_img.size <= 1 or (len(rgb_img.shape) >= 2 and (rgb_img.shape[0] <= 1 or rgb_img.shape[1] <= 1)):
                        axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
                        axes[i].set_title(f"Step {step_idx}")
                        axes[i].axis('off')
                        continue
                    
                    # Normalize if needed
                    if rgb_img.dtype == np.uint8:
                        rgb_img = rgb_img.astype(np.float32) / 255.0
                    
                    axes[i].imshow(np.clip(rgb_img, 0, 1))
                    axes[i].set_title(f"Step {step_idx}")
                    axes[i].axis('off')
                    
                except Exception as e:
                    axes[i].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f"Step {step_idx}")
                    axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(sample_indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, f"{dataset_name}_summary_grid.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved summary grid to {summary_path}")
        
    except Exception as e:
        print(f"⚠️ Error creating summary grid: {e}")

def find_rgbd_datasets(directory):
    """Find all RGBD zarr datasets in a directory"""
    rgbd_datasets = []
    
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name.endswith('.zarr') and 'rgbd' in root.lower():
                dataset_path = os.path.join(root, dir_name)
                rgbd_datasets.append(dataset_path)
    
    return rgbd_datasets

def main():
    parser = argparse.ArgumentParser(description='Extract RGB images from RGBD zarr datasets as PNG files')
    parser.add_argument('--dataset', type=str, help='Path to zarr dataset file')
    parser.add_argument('--dataset_dir', type=str, help='Directory containing RGBD datasets')
    parser.add_argument('--output', type=str, help='Output directory for PNG files')
    parser.add_argument('--max_samples', type=int, default=10, help='Maximum number of samples to save')
    parser.add_argument('--all', action='store_true', help='Process all RGBD datasets in dataset_dir')
    parser.add_argument('--no_summary', action='store_true', help='Skip creating summary grid images')
    
    args = parser.parse_args()
    
    if args.all and args.dataset_dir:
        # Process all RGBD datasets in directory
        datasets = find_rgbd_datasets(args.dataset_dir)
        
        if not datasets:
            print(f"❌ No RGBD datasets found in {args.dataset_dir}")
            return 1
        
        print(f"📊 Found {len(datasets)} RGBD datasets:")
        for dataset in datasets:
            print(f"   - {dataset}")
        
        success_count = 0
        for dataset in datasets:
            print(f"\n🎯 Processing {dataset}...")
            if save_rgb_images_from_zarr(
                dataset, 
                args.output, 
                args.max_samples, 
                not args.no_summary
            ):
                success_count += 1
        
        print(f"\n📈 Summary: {success_count}/{len(datasets)} datasets processed successfully")
        return 0 if success_count == len(datasets) else 1
        
    elif args.dataset:
        # Process single dataset
        if not os.path.exists(args.dataset):
            print(f"❌ Dataset not found: {args.dataset}")
            return 1
        
        success = save_rgb_images_from_zarr(
            args.dataset, 
            args.output, 
            args.max_samples, 
            not args.no_summary
        )
        return 0 if success else 1
        
    else:
        print("❌ Please specify either --dataset or --dataset_dir with --all")
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 