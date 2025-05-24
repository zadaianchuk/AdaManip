#!/usr/bin/env python3
"""
Standalone test for RGBD dataset functionality
Tests the core RGBD data structures without requiring Isaac Gym
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add necessary paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    BASE_DIR,
    os.path.join(BASE_DIR, "dataset")
])

# Import dataset modules (should work without Isaac Gym)
from dataset.dataset_rgbd import RGBDExperience, RGBDEpisodeBuffer, RGBDManipDataset
import zarr

def test_rgbd_dataset_creation():
    """Test RGBD dataset creation and storage without Isaac Gym"""
    print("\n" + "="*60)
    print("TESTING: RGBD Dataset Creation (Standalone)")
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
        test_save_path = "./test_rgbd_dataset_standalone.zarr"
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
        
        # Test 6: Visualize sample data
        visualize_sample_rgbd(sample, "test_rgbd_standalone_output")
        
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

def test_rgbd_buffer_functionality():
    """Test basic RGBD buffer functionality"""
    print("\n" + "="*60)
    print("TESTING: RGBD Buffer Functionality")
    print("="*60)
    
    try:
        # Test 1: RGBDEpisodeBuffer
        print("📦 Testing RGBDEpisodeBuffer...")
        buffer = RGBDEpisodeBuffer()
        
        # Test adding regular data
        pc = torch.randn(100, 3)
        env_state = torch.randn(10)
        action = torch.randn(7)
        buffer.add(pc, env_state, action)
        
        print("✓ Added regular observation to buffer")
        
        # Test adding RGBD data
        rgb_images = torch.randint(0, 255, (2, 64, 64, 3), dtype=torch.uint8)
        depth_images = torch.rand(2, 64, 64)
        buffer.add_rgbd(pc, env_state, action, rgb_images, depth_images)
        
        print("✓ Added RGBD observation to buffer")
        
        # Verify data
        assert len(buffer.pcs) == 2, f"Expected 2 point clouds, got {len(buffer.pcs)}"
        assert len(buffer.rgb_images) == 2, f"Expected 2 RGB sets, got {len(buffer.rgb_images)}"
        
        print("✓ Buffer contains expected number of observations")
        
        # Test 2: RGBDExperience
        print("📦 Testing RGBDExperience...")
        experience = RGBDExperience()
        experience.append(buffer)
        
        print("✓ Added episode buffer to experience")
        
        # Verify episode ends
        assert len(experience.meta["episode_ends"]) == 1, "Expected 1 episode end marker"
        assert experience.meta["episode_ends"][0] == 2, f"Expected episode end at 2, got {experience.meta['episode_ends'][0]}"
        
        print("✓ Episode end markers are correct")
        
        print("✅ RGBD buffer functionality test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ RGBD buffer functionality test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def visualize_sample_rgbd(sample, output_dir):
    """Visualize RGBD sample data"""
    try:
        print(f"🎨 Creating visualizations in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get RGB and depth images
        rgb_images = sample['rgb_images']  # Shape: [obs_horizon, num_cameras, height, width, 3]
        depth_images = sample['depth_images']  # Shape: [obs_horizon, num_cameras, height, width]
        
        print(f"📊 RGB shape: {rgb_images.shape}, Depth shape: {depth_images.shape}")
        
        # Visualize first observation
        if rgb_images.shape[0] > 0 and rgb_images.shape[1] > 0:
            rgb_obs = rgb_images[0]  # First observation
            depth_obs = depth_images[0]  # First observation
            
            num_cameras = rgb_obs.shape[0]
            print(f"📹 Number of cameras: {num_cameras}")
            
            # Create subplot for each camera
            fig, axes = plt.subplots(2, num_cameras, figsize=(4*num_cameras, 8))
            if num_cameras == 1:
                axes = axes.reshape(2, 1)
            
            for cam_id in range(num_cameras):
                rgb_img = rgb_obs[cam_id].numpy()
                depth_img = depth_obs[cam_id].numpy()
                
                # RGB image
                axes[0, cam_id].imshow(rgb_img.astype(np.uint8))
                axes[0, cam_id].set_title(f"RGB Camera {cam_id}")
                axes[0, cam_id].axis('off')
                
                # Depth image
                im = axes[1, cam_id].imshow(depth_img, cmap='viridis')
                axes[1, cam_id].set_title(f"Depth Camera {cam_id}")
                axes[1, cam_id].axis('off')
                plt.colorbar(im, ax=axes[1, cam_id])
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, "rgbd_visualization.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ RGBD visualization saved to {save_path}")
        else:
            print("⚠️ No RGB/depth images to visualize")
    
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")

def run_standalone_tests():
    """Run all standalone RGBD tests"""
    print("\n" + "🧪 " + "="*58)
    print("RGBD DATASET - STANDALONE TEST SUITE")
    print("="*60)
    
    tests = [
        ("RGBD Buffer Functionality", test_rgbd_buffer_functionality),
        ("RGBD Dataset Creation", test_rgbd_dataset_creation),
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
        print("🎉 All standalone tests passed! RGBD dataset implementation is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = run_standalone_tests()
    sys.exit(0 if success else 1) 