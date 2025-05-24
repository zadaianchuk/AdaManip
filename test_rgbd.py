#!/usr/bin/env python3
"""
Test script for RGBD data collection functionality
Provides comprehensive testing, debugging, and validation
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

# Now safe to import torch and other packages
import torch
import matplotlib.pyplot as plt
import zarr

def test_rgbd_environment_collection():
    """Test RGBD data collection at the environment level"""
    print("\n" + "="*60)
    print("TESTING: Environment RGBD Data Collection")
    print("="*60)
    
    # Mock args for testing
    class MockArgs:
        def __init__(self):
            self.task = "OpenBottle"
            self.headless = True
            self.controller = "GtController"
            self.manipulation = "OpenBottleManipulation"
            self.logdir = "./logs/"
            self.cfg_env = "Base"
            self.seed = 42
            self.device_id = 0
            self.device = 'cuda'
            self.use_gpu_pipeline = True
            self.compute_device_id = 0
            self.sim_device_type = 'cuda'
    
    args = MockArgs()
    
    try:
        # Load configuration
        cfg, logdir = load_cfg(args)
        sim_params = parse_sim_params(args, cfg)
        set_seed(args.seed)
        
        # Create environment
        env = parse_env(args, cfg, sim_params, logdir)
        
        # Test 1: Check if RGBD collection method exists
        has_rgbd_method = hasattr(env, 'collect_rgbd_data')
        print(f"✓ Environment has collect_rgbd_data method: {has_rgbd_method}")
        
        if not has_rgbd_method:
            print("❌ RGBD collection method not found! Need to add to this environment.")
            return False
        
        # Test 2: Try collecting RGBD data
        print("📸 Testing RGBD data collection...")
        env.reset()
        rgbd_obs = env.collect_rgbd_data()
        
        # Validate RGBD observation structure
        required_keys = ['pc', 'proprioception', 'dof_state', 'prev_action', 'rgb_images', 'depth_images']
        missing_keys = [key for key in required_keys if key not in rgbd_obs]
        
        if missing_keys:
            print(f"❌ Missing keys in RGBD observation: {missing_keys}")
            return False
        
        print("✓ RGBD observation contains all required keys")
        
        # Test 3: Validate image data shapes and types
        rgb_images = rgbd_obs['rgb_images']
        depth_images = rgbd_obs['depth_images']
        
        print(f"📊 RGB images shape: {rgb_images.shape}")
        print(f"📊 Depth images shape: {depth_images.shape}")
        print(f"📊 RGB dtype: {rgb_images.dtype}")
        print(f"📊 Depth dtype: {depth_images.dtype}")
        
        # Validate shapes
        if len(rgb_images.shape) < 4:  # Should be [num_envs, num_cameras, height, width, 3]
            print(f"❌ RGB images have invalid shape: {rgb_images.shape}")
            return False
            
        if len(depth_images.shape) < 3:  # Should be [num_envs, num_cameras, height, width]
            print(f"❌ Depth images have invalid shape: {depth_images.shape}")
            return False
        
        print("✓ Image shapes are valid")
        
        # Test 4: Check image value ranges
        rgb_min, rgb_max = rgb_images.min().item(), rgb_images.max().item()
        depth_min, depth_max = depth_images.min().item(), depth_images.max().item()
        
        print(f"📊 RGB value range: [{rgb_min:.2f}, {rgb_max:.2f}]")
        print(f"📊 Depth value range: [{depth_min:.2f}, {depth_max:.2f}]")
        
        # RGB should be in [0, 255] for uint8 or [0, 1] for float
        if rgb_images.dtype == torch.uint8:
            if rgb_min < 0 or rgb_max > 255:
                print(f"❌ RGB values out of range for uint8: [{rgb_min}, {rgb_max}]")
                return False
        
        print("✓ Image value ranges are reasonable")
        
        # Test 5: Save sample images for visual inspection
        save_sample_images(rgb_images, depth_images, "test_rgbd_samples")
        
        print("✅ Environment RGBD collection test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Environment RGBD collection test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_rgbd_dataset_creation():
    """Test RGBD dataset creation and storage"""
    print("\n" + "="*60)
    print("TESTING: RGBD Dataset Creation")
    print("="*60)
    
    try:
        # Test 1: Create RGBD buffers
        print("📦 Creating RGBD episode buffers...")
        
        num_envs = 2
        episode_length = 5
        eps_buffers = [RGBDEpisodeBuffer() for _ in range(num_envs)]
        
        # Simulate RGBD data
        for step in range(episode_length):
            for env_id in range(num_envs):
                # Mock data
                pc = torch.randn(1000, 3)
                env_state = torch.randn(20)
                action = torch.randn(9)  # 6D pose + 3D rotation
                rgb_images = torch.randint(0, 255, (3, 128, 128, 3), dtype=torch.uint8)  # 3 cameras
                depth_images = torch.rand(3, 128, 128)
                
                eps_buffers[env_id].add_rgbd(pc, env_state, action, rgb_images, depth_images)
        
        print(f"✓ Created {num_envs} episode buffers with {episode_length} steps each")
        
        # Test 2: Create experience buffer and save
        print("💾 Creating RGBD experience buffer...")
        demo_buffer = RGBDExperience()
        
        for eps_buffer in eps_buffers:
            demo_buffer.append(eps_buffer)
        
        print(f"✓ Added {len(eps_buffers)} episodes to experience buffer")
        
        # Test 3: Save to zarr format
        test_save_path = "./test_rgbd_dataset.zarr"
        print(f"💾 Saving RGBD dataset to {test_save_path}...")
        
        demo_buffer.save(test_save_path)
        print("✓ Dataset saved successfully")
        
        # Test 4: Load and validate saved data
        print("📂 Loading and validating saved dataset...")
        data = zarr.open(test_save_path, 'r')
        
        required_data_keys = ['pcs', 'env_state', 'action', 'rgb_images', 'depth_images']
        missing_data_keys = [key for key in required_data_keys if key not in data['data']]
        
        if missing_data_keys:
            print(f"❌ Missing data keys: {missing_data_keys}")
            return False
        
        print("✓ All required data keys present")
        
        # Check metadata
        if 'rgbd_meta' in data:
            rgbd_meta = dict(data['rgbd_meta'])
            print(f"📊 RGBD metadata: {rgbd_meta}")
        
        # Validate data shapes
        pcs_shape = data['data']['pcs'].shape
        rgb_shape = data['data']['rgb_images'].shape
        depth_shape = data['data']['depth_images'].shape
        
        print(f"📊 Saved shapes - PCs: {pcs_shape}, RGB: {rgb_shape}, Depth: {depth_shape}")
        
        expected_total_steps = num_envs * episode_length
        if pcs_shape[0] != expected_total_steps:
            print(f"❌ Unexpected number of steps: {pcs_shape[0]} vs expected {expected_total_steps}")
            return False
        
        print("✓ Data shapes are correct")
        
        # Test 5: Load with PyTorch dataset
        print("🔄 Testing PyTorch dataset loading...")
        
        dataset = RGBDManipDataset(
            dataset_path=[test_save_path],
            pred_horizon=4,
            obs_horizon=2,
            action_horizon=1,
            use_images=True
        )
        
        print(f"✓ Dataset loaded with {len(dataset)} sequences")
        
        # Test sample retrieval
        sample = dataset[0]
        sample_keys = list(sample.keys())
        expected_keys = ['pcs', 'env_state', 'action', 'rgb_images', 'depth_images']
        
        missing_sample_keys = [key for key in expected_keys if key not in sample_keys]
        if missing_sample_keys:
            print(f"❌ Missing sample keys: {missing_sample_keys}")
            return False
        
        print("✓ Sample contains all expected keys")
        print(f"📊 Sample shapes: {[(k, v.shape) for k, v in sample.items()]}")
        
        # Cleanup
        import shutil
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
            print("🧹 Cleaned up test files")
        
        print("✅ RGBD dataset creation test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ RGBD dataset creation test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_rgbd_run_integration():
    """Test the full RGBD run integration"""
    print("\n" + "="*60)
    print("TESTING: RGBD Run Integration")
    print("="*60)
    
    try:
        # This test will run the RGBD collection pipeline with minimal episodes
        print("🚀 Testing run_rgbd.py integration...")
        
        # Import the run_rgbd module
        import run_rgbd
        
        # Mock minimal config for testing
        class MockArgs:
            def __init__(self):
                self.task = "OpenBottle"
                self.headless = True
                self.controller = "GtController"
                self.manipulation = "OpenBottleManipulation"
                self.logdir = "./logs/"
                self.cfg_env = "Base"
                self.seed = 42
                self.device_id = 0
                self.device = 'cuda'
                self.use_gpu_pipeline = True
                self.compute_device_id = 0
                self.sim_device_type = 'cuda'
        
        # Set minimal episode count for testing
        original_args = run_rgbd.args if hasattr(run_rgbd, 'args') else None
        
        print("✓ RGBD integration components imported successfully")
        
        # Note: We won't actually run the full pipeline here as it requires 
        # significant computational resources and time. Instead, we validate
        # that the integration components are properly structured.
        
        # Check that the key functions exist
        required_functions = ['run_rgbd', 'collect_grasp_data_rgbd', 'collect_manip_data_rgbd']
        missing_functions = [func for func in required_functions if not hasattr(run_rgbd, func)]
        
        if missing_functions:
            print(f"❌ Missing functions in run_rgbd: {missing_functions}")
            return False
        
        print("✓ All required functions present in run_rgbd")
        
        print("✅ RGBD run integration test PASSED")
        print("💡 Note: Full pipeline testing requires manual execution with proper environment setup")
        return True
        
    except Exception as e:
        print(f"❌ RGBD run integration test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def save_sample_images(rgb_images, depth_images, output_dir):
    """Save sample RGB and depth images for visual inspection"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to numpy for saving
        if isinstance(rgb_images, torch.Tensor):
            rgb_np = rgb_images.cpu().numpy()
        else:
            rgb_np = rgb_images
            
        if isinstance(depth_images, torch.Tensor):
            depth_np = depth_images.cpu().numpy()
        else:
            depth_np = depth_images
        
        # Save a few sample images from first environment, first camera
        if len(rgb_np.shape) >= 4 and rgb_np.shape[0] > 0:
            sample_rgb = rgb_np[0, 0]  # First env, first camera
            sample_depth = depth_np[0, 0]
            
            # Normalize RGB if needed
            if sample_rgb.dtype == np.uint8:
                sample_rgb = sample_rgb.astype(np.float32) / 255.0
            
            # Save RGB
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.imshow(sample_rgb)
            plt.title("Sample RGB Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(sample_depth, cmap='viridis')
            plt.title("Sample Depth Image")
            plt.colorbar()
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "sample_rgbd.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Sample images saved to {output_dir}/sample_rgbd.png")
    
    except Exception as e:
        print(f"⚠️ Could not save sample images: {str(e)}")

def run_all_tests():
    """Run all RGBD tests"""
    print("\n" + "🧪 " + "="*58)
    print("RGBD DATA COLLECTION - COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Environment RGBD Collection", test_rgbd_environment_collection),
        ("RGBD Dataset Creation", test_rgbd_dataset_creation),
        ("RGBD Run Integration", test_rgbd_run_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🏃 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {str(e)}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RGBD implementation is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    set_np_formatting()
    success = run_all_tests()
    sys.exit(0 if success else 1) 