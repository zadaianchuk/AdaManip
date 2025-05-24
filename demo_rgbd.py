#!/usr/bin/env python3
"""
Demonstration script for RGBD data collection in AdaManip

This script shows how to:
1. Set up an environment with RGBD collection
2. Collect a few episodes of RGBD data
3. Save and load the data
4. Visualize sample images

Usage:
    python demo_rgbd.py --task OpenBottle --headless --controller GtController --manipulation OpenBottleManipulation
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add necessary paths BEFORE any imports that might use Isaac Gym
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    BASE_DIR,
    os.path.join(BASE_DIR, "envs"),
    os.path.join(BASE_DIR, "controller"),
    os.path.join(BASE_DIR, "dataset")
])

# Import Isaac Gym dependent modules first (before torch)
from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_env
from utils.parse import *
from dataset.dataset_rgbd import RGBDExperience, RGBDEpisodeBuffer, RGBDManipDataset
from utils.rgbd_utils import add_rgbd_collection_to_env, debug_rgbd_collection, validate_rgbd_environment

# Now safe to import torch and other packages
import torch
import matplotlib.pyplot as plt
import zarr

def demo_rgbd_collection():
    """Demonstrate RGBD data collection with a minimal example"""
    
    print("\n" + "🚀 " + "="*58)
    print("RGBD DATA COLLECTION DEMONSTRATION")
    print("="*60)
    
    # Setup
    set_np_formatting()
    args = get_args()
    cfg, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg)
    set_seed(args.seed)
    
    # Create environment
    print("🏗️ Setting up environment...")
    env = parse_env(args, cfg, sim_params, logdir)
    
    # Add RGBD capability if not present
    if not hasattr(env, 'collect_rgbd_data'):
        print("➕ Adding RGBD collection capability...")
        add_rgbd_collection_to_env(env.__class__)
    
    # Validate RGBD capability
    print("🔍 Validating RGBD collection...")
    validation = validate_rgbd_environment(env)
    
    if not validation['can_collect_rgbd']:
        print("❌ RGBD collection validation failed:")
        for error in validation['errors']:
            print(f"   - {error}")
        return False
    
    print("✅ RGBD collection validated successfully")
    print(f"📊 Camera info: {validation['camera_info']}")
    
    # Collect sample data
    print("\n📸 Collecting sample RGBD episodes...")
    num_episodes = 3  # Small number for demo
    demo_buffer = RGBDExperience()
    
    for eps in range(num_episodes):
        print(f"📦 Episode {eps+1}/{num_episodes}")
        
        # Reset environment
        env.reset()
        
        # Create episode buffer
        eps_buffer = [RGBDEpisodeBuffer() for _ in range(env.num_envs)]
        
        # Collect a few steps of data
        for step in range(5):  # Short episodes for demo
            # Collect RGBD observation
            rgbd_obs = env.collect_rgbd_data(debug=False)
            
            # Generate random action (for demo purposes)
            action = env.hand_rigid_body_tensor[:, :7].clone()
            action += 0.01 * torch.randn_like(action)  # Add small random movement
            
            # Step environment
            for _ in range(5):
                env.step(action)
            env.actions = action
            
            # Store data
            for env_id in range(env.num_envs):
                eps_buffer[env_id].add_rgbd(
                    rgbd_obs['pc'][env_id],
                    rgbd_obs['proprioception'][env_id],
                    action[env_id],
                    rgbd_obs['rgb_images'][env_id],
                    rgbd_obs['depth_images'][env_id]
                )
        
        # Add episodes to demo buffer
        for env_id in range(env.num_envs):
            demo_buffer.append(eps_buffer[env_id])
    
    print(f"✅ Collected {num_episodes} episodes")
    
    # Save data
    demo_save_path = "./demo_rgbd_dataset.zarr"
    print(f"💾 Saving demo dataset to {demo_save_path}...")
    demo_buffer.save(demo_save_path)
    
    # Load and verify data
    print("📂 Loading and verifying saved data...")
    dataset = RGBDManipDataset(
        dataset_path=[demo_save_path],
        pred_horizon=3,
        obs_horizon=2,
        action_horizon=1,
        use_images=True
    )
    
    print(f"📊 Dataset loaded with {len(dataset)} sequences")
    
    # Get a sample
    sample = dataset[0]
    print(f"📋 Sample keys: {list(sample.keys())}")
    print(f"📋 Sample shapes: {[(k, v.shape) for k, v in sample.items()]}")
    
    # Visualize sample images
    visualize_sample_rgbd(sample, output_dir="demo_rgbd_output")
    
    # Cleanup
    import shutil
    if os.path.exists(demo_save_path):
        shutil.rmtree(demo_save_path)
        print("🧹 Cleaned up demo files")
    
    print("\n🎉 RGBD demonstration completed successfully!")
    return True

def visualize_sample_rgbd(sample, output_dir="demo_rgbd_output"):
    """Visualize RGBD sample data"""
    print(f"🎨 Creating visualizations in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get RGB and depth images
    rgb_images = sample['rgb_images']  # Shape: [obs_horizon, num_cameras, height, width, 3]
    depth_images = sample['depth_images']  # Shape: [obs_horizon, num_cameras, height, width]
    
    # Visualize first observation
    if rgb_images.shape[0] > 0 and rgb_images.shape[1] > 0:
        rgb_obs = rgb_images[0]  # First observation
        depth_obs = depth_images[0]  # First observation
        
        num_cameras = rgb_obs.shape[0]
        
        # Create subplot for each camera
        fig, axes = plt.subplots(2, num_cameras, figsize=(4*num_cameras, 8))
        if num_cameras == 1:
            axes = axes.reshape(2, 1)
        
        for cam_id in range(num_cameras):
            rgb_img = rgb_obs[cam_id].numpy()
            depth_img = depth_obs[cam_id].numpy()
            
            # RGB image
            axes[0, cam_id].imshow(rgb_img)
            axes[0, cam_id].set_title(f"RGB Camera {cam_id}")
            axes[0, cam_id].axis('off')
            
            # Depth image
            im = axes[1, cam_id].imshow(depth_img, cmap='viridis')
            axes[1, cam_id].set_title(f"Depth Camera {cam_id}")
            axes[1, cam_id].axis('off')
            plt.colorbar(im, ax=axes[1, cam_id])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rgbd_visualization.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ RGBD visualization saved to {output_dir}/rgbd_visualization.png")
        
        # Create point cloud visualization
        visualize_point_cloud(sample, output_dir)
    else:
        print("⚠️ No RGB/depth images to visualize")

def visualize_point_cloud(sample, output_dir):
    """Visualize point cloud data"""
    try:
        pcs = sample['pcs']  # Shape: [obs_horizon, num_points, 3]
        
        if pcs.shape[0] > 0:
            pc = pcs[0].numpy()  # First observation
            
            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Sample points for visualization (to avoid overcrowding)
            if pc.shape[0] > 1000:
                indices = np.random.choice(pc.shape[0], 1000, replace=False)
                pc_viz = pc[indices]
            else:
                pc_viz = pc
            
            # Color by Z coordinate
            scatter = ax.scatter(pc_viz[:, 0], pc_viz[:, 1], pc_viz[:, 2], 
                               c=pc_viz[:, 2], cmap='viridis', s=1)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Point Cloud Visualization')
            plt.colorbar(scatter)
            
            plt.savefig(os.path.join(output_dir, "pointcloud_visualization.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Point cloud visualization saved to {output_dir}/pointcloud_visualization.png")
    
    except Exception as e:
        print(f"⚠️ Could not create point cloud visualization: {e}")

def create_rgbd_dataset_info(dataset_path):
    """Create informational summary about an RGBD dataset"""
    try:
        data = zarr.open(dataset_path, 'r')
        
        info = {
            'dataset_path': dataset_path,
            'num_episodes': len(data['meta']['episode_ends']),
            'total_steps': data['data']['pcs'].shape[0],
            'pc_shape': data['data']['pcs'].shape,
            'action_shape': data['data']['action'].shape,
            'env_state_shape': data['data']['env_state'].shape,
        }
        
        if 'rgb_images' in data['data']:
            info['rgb_shape'] = data['data']['rgb_images'].shape
            info['depth_shape'] = data['data']['depth_images'].shape
            info['has_rgbd'] = True
        else:
            info['has_rgbd'] = False
        
        if 'rgbd_meta' in data:
            info['rgbd_meta'] = dict(data['rgbd_meta'])
        
        return info
    
    except Exception as e:
        return {'error': str(e)}

def analyze_existing_datasets():
    """Analyze existing RGBD datasets in the demo_data directory"""
    print("\n🔍 Analyzing existing datasets...")
    
    demo_data_dir = "./demo_data"
    if not os.path.exists(demo_data_dir):
        print("ℹ️ No demo_data directory found")
        return
    
    rgbd_datasets = []
    for item in os.listdir(demo_data_dir):
        if 'rgbd' in item.lower() and os.path.isdir(os.path.join(demo_data_dir, item)):
            dataset_path = os.path.join(demo_data_dir, item)
            # Look for zarr files
            for file in os.listdir(dataset_path):
                if file.endswith('.zarr'):
                    full_path = os.path.join(dataset_path, file)
                    info = create_rgbd_dataset_info(full_path)
                    if 'error' not in info:
                        rgbd_datasets.append(info)
    
    if rgbd_datasets:
        print(f"📊 Found {len(rgbd_datasets)} RGBD datasets:")
        for i, info in enumerate(rgbd_datasets):
            print(f"\n   Dataset {i+1}:")
            print(f"   📁 Path: {info['dataset_path']}")
            print(f"   📦 Episodes: {info['num_episodes']}")
            print(f"   🔢 Total steps: {info['total_steps']}")
            if info.get('has_rgbd'):
                print(f"   🖼️ RGB shape: {info['rgb_shape']}")
                print(f"   📏 Depth shape: {info['depth_shape']}")
    else:
        print("ℹ️ No RGBD datasets found in demo_data directory")

if __name__ == "__main__":
    try:
        # Check if we should just analyze existing datasets
        if '--analyze' in sys.argv:
            analyze_existing_datasets()
        else:
            # Run full demo
            success = demo_rgbd_collection()
            if not success:
                print("❌ Demo failed")
                sys.exit(1)
            
            # Also analyze any existing datasets
            analyze_existing_datasets()
            
            print("\n💡 Tips:")
            print("   - Use --analyze flag to just analyze existing datasets")
            print("   - Check the debug_rgbd_* directories for sample images")
            print("   - Use test_rgbd.py for comprehensive testing")
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 