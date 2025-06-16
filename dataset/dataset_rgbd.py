import numpy as np
import torch
import pickle
import os
import json
from PIL import Image
from pytorch3d.ops import sample_farthest_points
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
import zarr
import ipdb
import matplotlib.pyplot as plt
from .dataset import obs_wrapper, nested_dict_save, create_sample_indices, sample_sequence

class RGBDEpisodeBuffer:
    """Episode buffer that stores RGBD observations along with camera parameters"""
    def __init__(self):
        self.pcs = []
        self.env_state = []
        self.action = []
        self.rgb_images = []
        self.depth_images = []
        self.segmentation_masks = []
        # Camera parameter storage
        self.camera_intrinsics = []  # Camera intrinsic matrices [fx, fy, cx, cy], fx and fy are Focal Lengths in pixels
        self.camera_extrinsics = []  # Camera extrinsic matrices (world-to-camera transform), inv of camera_view_matrix
        self.camera_info = []        # Additional camera metadata

    def add(self, pc, env_state, action):
        """Add standard observation (for backward compatibility)"""
        self.pcs.append(pc.cpu().numpy())
        self.env_state.append(env_state.cpu().numpy())
        self.action.append(action.cpu().numpy())
        # Add empty RGB/depth placeholders for compatibility
        self.rgb_images.append(np.zeros((1, 1, 3), dtype=np.uint8))
        self.depth_images.append(np.zeros((1, 1), dtype=np.float32))
        self.segmentation_masks.append(np.zeros((1, 1), dtype=np.uint32))
        # Add empty camera parameter placeholders
        self.camera_intrinsics.append(np.zeros((1, 4), dtype=np.float32))  # [fx, fy, cx, cy]
        self.camera_extrinsics.append(np.eye(4, dtype=np.float32).reshape(1, 4, 4))
        self.camera_info.append([{"type": "dummy", "id": 0}])

    def add_rgbd(self, pc, env_state, action, rgb_images, depth_images, segmentation_masks=None, camera_intrinsics=None, camera_extrinsics=None, camera_info=None):
        """Add RGBD observation with images, segmentation masks and camera parameters"""
        self.pcs.append(pc.cpu().numpy())
        self.env_state.append(env_state.cpu().numpy())
        self.action.append(action.cpu().numpy())
        
        # Handle RGB images (ensure consistent format)
        if isinstance(rgb_images, torch.Tensor):
            rgb_images = rgb_images.cpu().numpy()
        if isinstance(rgb_images, list):
            rgb_images = np.array(rgb_images)
        self.rgb_images.append(rgb_images)
        
        # Handle depth images
        if isinstance(depth_images, torch.Tensor):
            depth_images = depth_images.cpu().numpy()
        if isinstance(depth_images, list):
            depth_images = np.array(depth_images)
        self.depth_images.append(depth_images)
        
        # Handle segmentation masks
        if segmentation_masks is not None:
            if isinstance(segmentation_masks, torch.Tensor):
                segmentation_masks = segmentation_masks.cpu().numpy()
            if isinstance(segmentation_masks, list):
                segmentation_masks = np.array(segmentation_masks)
            self.segmentation_masks.append(segmentation_masks)
        else:
            raise ValueError("segmentation_masks is not provided")
        
        # Handle camera parameters
        if camera_intrinsics is not None:
            if isinstance(camera_intrinsics, torch.Tensor):
                camera_intrinsics = camera_intrinsics.cpu().numpy()
            self.camera_intrinsics.append(camera_intrinsics)
        else:
            raise ValueError("camera_intrinsics is not provided")
        
        if camera_extrinsics is not None:
            if isinstance(camera_extrinsics, torch.Tensor):
                camera_extrinsics = camera_extrinsics.cpu().numpy()
            self.camera_extrinsics.append(camera_extrinsics)
        else:
            # Create default extrinsics (identity matrices)
            num_cameras = rgb_images.shape[0] if len(rgb_images.shape) > 2 else 1
            default_extrinsics = np.tile(np.eye(4, dtype=np.float32), (num_cameras, 1, 1))
            self.camera_extrinsics.append(default_extrinsics)
        
        
        if camera_info is not None:
            self.camera_info.append(camera_info)
        else:
            # Create default camera info
            num_cameras = rgb_images.shape[0] if len(rgb_images.shape) > 2 else 1
            default_info = [{"type": "unknown", "id": i} for i in range(num_cameras)]
            self.camera_info.append(default_info)

class RGBDExperience:
    """Experience replay buffer that handles RGBD data with camera parameters and segmentation masks"""
    def __init__(self, sample_pcs_num=1000):
        self.sample_pcs_num = sample_pcs_num
        self.data = {
            "pcs": [], 
            "env_state": [], 
            "action": [],
            "rgb_images": [],
            "depth_images": [],
            "segmentation_masks": [],
            "camera_intrinsics": [],
            "camera_extrinsics": [],
            "camera_info": []
        }
        self.meta = {"episode_ends": []}
    
    def append(self, episode: RGBDEpisodeBuffer):
        if len(episode.pcs) == 0:
            print("skip empty eps")
            return        
        
        if self.meta["episode_ends"] == []:
            self.data["pcs"] = np.array(episode.pcs)
            self.data["env_state"] = np.array(episode.env_state)
            self.data["action"] = np.array(episode.action)
            self.data["rgb_images"] = np.array(episode.rgb_images)
            self.data["depth_images"] = np.array(episode.depth_images)
            self.data["segmentation_masks"] = np.array(episode.segmentation_masks)
            self.data["camera_intrinsics"] = np.array(episode.camera_intrinsics)
            self.data["camera_extrinsics"] = np.array(episode.camera_extrinsics)
            # Camera info is stored as a list of lists (can't easily convert to numpy array)
            self.data["camera_info"] = episode.camera_info
        else:
            self.data["pcs"] = np.concatenate([self.data["pcs"], np.array(episode.pcs)])
            self.data["env_state"] = np.concatenate([self.data["env_state"], np.array(episode.env_state)])
            self.data["action"] = np.concatenate([self.data["action"], np.array(episode.action)])
            self.data["rgb_images"] = np.concatenate([self.data["rgb_images"], np.array(episode.rgb_images)])
            self.data["depth_images"] = np.concatenate([self.data["depth_images"], np.array(episode.depth_images)])
            self.data["segmentation_masks"] = np.concatenate([self.data["segmentation_masks"], np.array(episode.segmentation_masks)])
            self.data["camera_intrinsics"] = np.concatenate([self.data["camera_intrinsics"], np.array(episode.camera_intrinsics)])
            self.data["camera_extrinsics"] = np.concatenate([self.data["camera_extrinsics"], np.array(episode.camera_extrinsics)])
            self.data["camera_info"].extend(episode.camera_info)
        
        new_end = self.data["pcs"].shape[0]
        self.meta["episode_ends"].append(new_end)

    def save_png_npy(self, path, fixed_cameras_only=True):
        """Save RGBD data using d3fields structure: separate camera directories with color/depth/masks subdirs
        
        Args:
            path: Base directory path to save the dataset
            fixed_cameras_only: If True, only save data from fixed cameras (exclude hand cameras)
        """
        print(f"Saving RGBD dataset with segmentation masks in d3fields format to {path}")
        if fixed_cameras_only:
            print("📌 Using fixed cameras only (excluding hand cameras with time-varying extrinsics)")
        
        # Create base directory
        base_dir = path.replace('.zarr', '') if path.endswith('.zarr') else path
        os.makedirs(base_dir, exist_ok=True)
        
        # Get data shapes
        rgb_data = self.data["rgb_images"]
        depth_data = self.data["depth_images"]
        segmentation_data = self.data["segmentation_masks"]
        camera_info_data = self.data["camera_info"]
        
        if len(rgb_data) == 0:
            print("No RGB data to save")
            return
        
        # Determine number of cameras and filter for fixed cameras only
        num_steps = len(rgb_data)
        total_cameras = rgb_data[0].shape[0] if len(rgb_data[0].shape) > 1 else 1
        print(f"🔍 Debug: RGB data shape: {rgb_data[0].shape}, detected {total_cameras} cameras")
        
        # Filter cameras based on camera_info to identify fixed vs hand cameras
        fixed_camera_indices = []
        if fixed_cameras_only and len(camera_info_data) > 0:
            try:
                # Check first step's camera info to identify fixed cameras
                first_step_info = camera_info_data[0]
                print(f"🔍 Debug: Camera info length: {len(first_step_info)}, RGB cameras: {total_cameras}")
                
                # Only process cameras that actually exist in the RGB data
                max_cam_idx = min(len(first_step_info), total_cameras)
                for cam_idx in range(max_cam_idx):
                    cam_info = first_step_info[cam_idx]
                    # Handle case where cam_info might be a list or dict
                    if isinstance(cam_info, dict):
                        if cam_info.get('type', 'unknown') != 'hand':
                            fixed_camera_indices.append(cam_idx)
                    elif isinstance(cam_info, list):
                        # If cam_info is a list, assume it's not a hand camera
                        fixed_camera_indices.append(cam_idx)
                    else:
                        # Unknown format, assume it's not a hand camera
                        fixed_camera_indices.append(cam_idx)
                
                if not fixed_camera_indices:
                    # Fallback: assume all cameras are fixed if no hand cameras identified
                    fixed_camera_indices = list(range(total_cameras))
                    print("⚠️ No hand cameras identified, using all cameras")
                    
                print(f"🔍 Debug: Fixed camera indices: {fixed_camera_indices}")
            except Exception as e:
                print(f"⚠️ Error processing camera info: {e}, using all cameras")
                fixed_camera_indices = list(range(total_cameras))
        else:
            # Use all cameras if not filtering
            fixed_camera_indices = list(range(total_cameras))
        
        num_cameras = len(fixed_camera_indices)
        print(f"Saving {num_steps} steps with {num_cameras} fixed cameras (indices: {fixed_camera_indices})")
        
        # Create camera directories and subdirectories for fixed cameras only
        for i, cam_idx in enumerate(fixed_camera_indices):
            cam_dir = os.path.join(base_dir, f'camera_{i}')  # Use sequential numbering for output
            color_dir = os.path.join(cam_dir, 'color')
            depth_dir = os.path.join(cam_dir, 'depth')
            masks_dir = os.path.join(cam_dir, 'masks')
            os.makedirs(color_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)
        
        # Save images for each step and fixed camera
        for step_idx in range(num_steps):
            rgb_step = rgb_data[step_idx]  # Shape: [num_cameras, height, width, 3]
            depth_step = depth_data[step_idx]  # Shape: [num_cameras, height, width]
            seg_step = segmentation_data[step_idx]  # Shape: [num_cameras, height, width]
            
            if rgb_step.size == 0:
                continue
            
            for i, cam_idx in enumerate(fixed_camera_indices):
                try:
                    # Get RGB, depth and segmentation for this fixed camera
                    if len(rgb_step.shape) >= 4:  # [num_cameras, height, width, 3]
                        rgb_img = rgb_step[cam_idx]
                        depth_img = depth_step[cam_idx]
                        seg_img = seg_step[cam_idx]
                    else:  # Single camera case
                        rgb_img = rgb_step
                        depth_img = depth_step
                        seg_img = seg_step
                    
                    # Save RGB image as PNG in color directory
                    if rgb_img.size > 1:  # Not dummy data
                        rgb_filename = f"{step_idx}.png"
                        rgb_path = os.path.join(base_dir, f'camera_{i}', 'color', rgb_filename)
                        
                        # Ensure RGB is in correct format
                        if rgb_img.dtype != np.uint8:
                            if rgb_img.max() <= 1.0:
                                rgb_img = (rgb_img * 255).astype(np.uint8)
                            else:
                                rgb_img = rgb_img.astype(np.uint8)
                        
                        # Save using PIL
                        if len(rgb_img.shape) == 3 and rgb_img.shape[2] == 3:
                            Image.fromarray(rgb_img).save(rgb_path)
                    
                    # Save depth image (as 16-bit PNG to preserve precision)
                    depth_filename = f"{step_idx}.png"
                    depth_path = os.path.join(base_dir, f"camera_{i}", "depth", depth_filename)
                    # Convert to 16-bit for better precision
                    assert depth_img.dtype == np.float32
                    # Scale to 16-bit range while preserving original values
                    depth_img_16 = (np.clip(-depth_img, 0, 2.5) * 1000).astype(np.uint16)  # Convert meters to millimeters
                    Image.fromarray(depth_img_16).save(depth_path)
                    
                    # Save segmentation mask as PNG in masks directory
                    mask_filename = f"{step_idx}.png"
                    mask_path = os.path.join(base_dir, f"camera_{i}", "masks", mask_filename)
                    
                    # Convert segmentation mask to appropriate format
                    # Isaac Gym segmentation IDs are 32-bit integers, but we need to handle large IDs properly
                    if seg_img.dtype != np.uint32:
                        seg_img = seg_img.astype(np.uint32)
                    
                    # For PNG saving, we need to decide how to handle the segmentation IDs
                    # Option 1: Save as grayscale PNG (8-bit) - map unique IDs to 0-255 range
                    # Option 2: Save as RGB PNG where ID is encoded in RGB channels
                    # Option 3: Save as 16-bit grayscale PNG (supports up to 65535 unique IDs)
                    
                    # We'll use Option 3: 16-bit grayscale PNG for better ID preservation
                    seg_img_16 = np.clip(seg_img, 0, 65535).astype(np.uint16)
                    Image.fromarray(seg_img_16, mode='I;16').save(mask_path)
                
                except Exception as e:
                    print(f"Error saving images for step {step_idx}, camera {cam_idx}: {e}")
                    continue
        
        # Save camera parameters for each fixed camera
        print("Saving camera parameters for fixed cameras...")
        for i, cam_idx in enumerate(fixed_camera_indices):
            cam_dir = os.path.join(base_dir, f'camera_{i}')
            
            # Extract camera parameters for this fixed camera
            if len(self.data["camera_intrinsics"]) > 0:
                # Get intrinsics for this camera (assuming they're consistent across steps)
                intrinsics_all_steps = self.data["camera_intrinsics"]
                if len(intrinsics_all_steps[0].shape) > 1 and intrinsics_all_steps[0].shape[0] > cam_idx:
                    camera_params = intrinsics_all_steps[0][cam_idx]  # [fx, fy, cx, cy]
                else:
                    raise ValueError(f"Camera parameters not found for camera {cam_idx}")
                
                np.save(os.path.join(cam_dir, 'camera_params.npy'), camera_params)
            else:
                raise ValueError("Camera intrinsics not found")
            
            if len(self.data["camera_extrinsics"]) > 0:
                extrinsics_all_steps = self.data["camera_extrinsics"]
                if len(extrinsics_all_steps[0].shape) > 2 and extrinsics_all_steps[0].shape[0] > cam_idx:
                    camera_extrinsics = extrinsics_all_steps[0][cam_idx]  # [4, 4]
                else:
                    raise ValueError(f"Camera extrinsics not found for camera {cam_idx}")
                
                np.save(os.path.join(cam_dir, 'camera_extrinsics.npy'), camera_extrinsics)
            else:
                raise ValueError("Camera extrinsics not found")
        # Filter other data to match fixed cameras only
        filtered_camera_intrinsics = []
        filtered_camera_extrinsics = []
        filtered_camera_info = []
        
        for step_idx in range(len(self.data["camera_intrinsics"])):
            step_intrinsics = [self.data["camera_intrinsics"][step_idx][cam_idx] for cam_idx in fixed_camera_indices]
            step_extrinsics = [self.data["camera_extrinsics"][step_idx][cam_idx] for cam_idx in fixed_camera_indices]
            step_info = [self.data["camera_info"][step_idx][cam_idx] for cam_idx in fixed_camera_indices]
            
            filtered_camera_intrinsics.append(step_intrinsics)
            filtered_camera_extrinsics.append(step_extrinsics)
            filtered_camera_info.append(step_info)
        
        # Save other data as NPY files in base directory
        data_files = {
            'pcs.npy': self.data["pcs"],
            'env_state.npy': self.data["env_state"], 
            'action.npy': self.data["action"],
            'episode_ends.npy': np.array(self.meta["episode_ends"]),    
            'camera_intrinsics_all_steps.npy': np.array(filtered_camera_intrinsics),
            'camera_extrinsics_all_steps.npy': np.array(filtered_camera_extrinsics)
        }
        
        print("Saving additional data files...")
        for filename, data in data_files.items():
            np.save(os.path.join(base_dir, filename), data)
        
        # Save filtered camera info as JSON
        with open(os.path.join(base_dir, 'camera_info.json'), 'w') as f:
            json.dump(filtered_camera_info, f, indent=2)
        
        # Save dataset info file (similar to d3fields format)
        dataset_info = {
            "format": "d3fields_compatible_with_segmentation",
            "num_frames": num_steps,
            "num_cameras": num_cameras,
            "fixed_cameras_only": fixed_cameras_only,
            "fixed_camera_indices": fixed_camera_indices,
            "rgb_shape": f"({num_steps}, {num_cameras}, H, W, 3)",
            "depth_format": "PNG RGB",
            "color_format": "PNG RGB",
            "segmentation_format": "PNG 16-bit grayscale (segmentation IDs)",
            "camera_params_format": "numpy [fx, fy, cx, cy]",
            "camera_extrinsics_format": "numpy 4x4 world-to-camera (static)",
            "total_episodes": len(self.meta["episode_ends"]),
            "storage_structure": "camera_X/color/*.png, camera_X/depth/*.png, camera_X/masks/*.png",
            "note": "Hand cameras excluded due to time-varying extrinsics. Segmentation masks saved as 16-bit PNG."
        }
        
        with open(os.path.join(base_dir, 'dataset_info.txt'), 'w') as f:
            f.write("D3Fields Compatible Dataset Info (with Segmentation)\n")
            f.write("=" * 50 + "\n\n")
            for key, value in dataset_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Saved RGBD dataset with segmentation masks in d3fields format:")
        print(f"   Base directory: {base_dir}")
        print(f"   Frames: {num_steps}")
        print(f"   Fixed cameras: {num_cameras}")
        print(f"   RGB: camera_X/color/*.png")
        print(f"   Depth: camera_X/depth/*.png")
        print(f"   Segmentation: camera_X/masks/*.png (16-bit PNG)")
        print(f"   Camera params: camera_X/camera_params.npy")
        print(f"   Static extrinsics only (hand cameras excluded)")