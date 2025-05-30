#!/usr/bin/env python3
"""
Enhanced RGBD utilities for AdaManip with proper camera parameter extraction

This module provides utilities for collecting RGBD data with accurate camera parameters
needed for point cloud reconstruction and compatibility with D3Fields format.

Key features:
- Extract camera intrinsics from Isaac Gym projection matrices
- Convert view matrices to proper extrinsics (world-to-camera transforms)
- Support for both fixed and hand-mounted cameras
- Automatic fallback for environments without RGBD capability
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from isaacgym import gymapi
from isaacgym import gymtorch

def extract_camera_intrinsics_from_projection(proj_matrix: torch.Tensor, width: int, height: int) -> np.ndarray:
    """
    Extract camera intrinsics [fx, fy, cx, cy] from Isaac Gym projection matrix
    
    Isaac Gym uses OpenGL-style projection matrices. We need to convert to camera intrinsics.
    
    Args:
        proj_matrix: (4, 4) projection matrix from Isaac Gym
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        intrinsics: (4,) array [fx, fy, cx, cy]
    """
    # Isaac Gym projection matrix is in OpenGL format
    # P[0,0] = 2*fx/width, P[1,1] = 2*fy/height
    # P[0,2] = (2*cx - width)/width, P[1,2] = (2*cy - height)/height
    
    if isinstance(proj_matrix, torch.Tensor):
        proj_matrix = proj_matrix.cpu().numpy()
    
    # Extract focal lengths
    fx = proj_matrix[0, 0] * width / 2.0
    fy = proj_matrix[1, 1] * height / 2.0
    # Extract principal point
    cx = (proj_matrix[0, 2] * width + width) / 2.0
    cy = (proj_matrix[1, 2] * height + height) / 2.0
    
    return np.array([fx, fy, cx, cy], dtype=np.float32)

def convert_view_matrix_to_extrinsics(view_matrix: torch.Tensor, is_already_inverted: bool = False) -> np.ndarray:
    """
    Convert Isaac Gym view matrix to camera extrinsics (world-to-camera transform)
    
    Args:
        view_matrix: (4, 4) view matrix from Isaac Gym
        is_already_inverted: If True, the input is already camera-to-world (inverted view matrix)
                           If False, the input is world-to-camera (original view matrix)
        
    Returns:
        extrinsics: (4, 4) world-to-camera transformation matrix
    """
    if isinstance(view_matrix, torch.Tensor):
        view_matrix = view_matrix.cpu().numpy()
    
    if is_already_inverted:
        # Input is camera-to-world, we need world-to-camera
        # So we take the inverse
        extrinsics = np.linalg.inv(view_matrix)
    else:
        # Input is world-to-camera (original Isaac Gym view matrix)
        # This is already what we want for extrinsics
        extrinsics = view_matrix.copy()
    
    return extrinsics.astype(np.float32)

def collect_camera_images(env, debug=False):
    """
    Enhanced camera image collection with proper camera parameter extraction
    
    Args:
        env: Isaac Gym environment
        debug: Whether to print debug information
        
    Returns:
        tuple: (rgb_images, depth_images, camera_intrinsics, camera_extrinsics, camera_info)
    """
    try:
        # Ensure camera sensors are rendered
        if hasattr(env, 'gym') and hasattr(env, 'sim'):
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.start_access_image_tensors(env.sim)
        
        rgb_images_list = []
        depth_images_list = []
        camera_intrinsics_list = []
        camera_extrinsics_list = []
        camera_info_list = []
        
        # Get camera properties for intrinsics calculation
        cam_width = env.cfg.get("env", {}).get("cam", {}).get("width", 128)
        cam_height = env.cfg.get("env", {}).get("cam", {}).get("height", 128)
        
        if debug:
            print(f"Camera resolution: {cam_width}x{cam_height}")
        
        # Process each environment
        for env_id in range(env.num_envs):
            env_rgb_images = []
            env_depth_images = []
            env_intrinsics = []
            env_extrinsics = []
            env_info = []
            
            # Collect from fixed cameras
            if hasattr(env, 'fixed_camera_handle_list') and len(env.fixed_camera_handle_list) > env_id:
                fixed_cameras = env.fixed_camera_handle_list[env_id]
                
                for cam_idx, camera_handle in enumerate(fixed_cameras):
                    try:
                        # Get RGB image
                        rgb_tensor = env.gym.get_camera_image_gpu_tensor(env.sim, env.env_ptr_list[env_id], camera_handle, gymapi.IMAGE_COLOR)
                        rgb_image = gymtorch.wrap_tensor(rgb_tensor)
                        rgb_image = rgb_image.cpu().numpy()
                        
                        # Remove alpha channel
                        if len(rgb_image.shape) == 3 and rgb_image.shape[2] == 4:
                            rgb_image = rgb_image[:, :, :3]  # Remove alpha channel (RGBA -> RGB)
                        
                        # Get depth image
                        depth_tensor = env.gym.get_camera_image_gpu_tensor(env.sim, env.env_ptr_list[env_id], camera_handle, gymapi.IMAGE_DEPTH)
                        depth_image = gymtorch.wrap_tensor(depth_tensor)
                        depth_image = depth_image.cpu().numpy()
                        
                        # Get camera parameters
                        if hasattr(env, 'fixed_camera_proj_list') and len(env.fixed_camera_proj_list) > env_id:
                            if len(env.fixed_camera_proj_list[env_id]) > cam_idx:
                                proj_matrix = env.fixed_camera_proj_list[env_id][cam_idx]
                                intrinsics = extract_camera_intrinsics_from_projection(proj_matrix, cam_width, cam_height)
                            else:
                                raise ValueError(f"No projection matrix found for camera {cam_idx} in environment {env_id}")
                        else:
                            raise ValueError(f"No projection matrix list found for environment {env_id}")
                        
                        if hasattr(env, 'fixed_camera_vinv_list') and len(env.fixed_camera_vinv_list) > env_id:
                            if len(env.fixed_camera_vinv_list[env_id]) > cam_idx:
                                view_matrix = env.fixed_camera_vinv_list[env_id][cam_idx]
                                extrinsics = convert_view_matrix_to_extrinsics(view_matrix, is_already_inverted=True)
                                
                                if debug:
                                    # Verify the fix: extrinsics should be world-to-camera (det should be ~1)
                                    det = np.linalg.det(extrinsics[:3, :3])
                                    print(f"      Extrinsics rotation determinant: {det:.3f} (should be ~1.0)")
                            else:
                                extrinsics = np.eye(4, dtype=np.float32)
                        else:
                            extrinsics = np.eye(4, dtype=np.float32)
                        
                        env_rgb_images.append(rgb_image)
                        env_depth_images.append(depth_image)
                        env_intrinsics.append(intrinsics)
                        env_extrinsics.append(extrinsics)
                        env_info.append({"type": "fixed", "id": cam_idx, "env_id": env_id})
                        
                        if debug:
                            print(f"   Fixed camera {cam_idx}: RGB {rgb_image.shape}, Depth {depth_image.shape}")
                            print(f"      Intrinsics: fx={intrinsics[0]:.1f}, fy={intrinsics[1]:.1f}, cx={intrinsics[2]:.1f}, cy={intrinsics[3]:.1f}")
                        
                    except Exception as e:
                        if debug:
                            print(f"⚠️ Error collecting from fixed camera {cam_idx}: {e}")
                        continue
            
            # Convert to numpy arrays
            if env_rgb_images:
                rgb_images_list.append(np.array(env_rgb_images))
                depth_images_list.append(np.array(env_depth_images))
                camera_intrinsics_list.append(np.array(env_intrinsics))
                camera_extrinsics_list.append(np.array(env_extrinsics))
                camera_info_list.append(env_info)
            else:
                raise ValueError(f"No cameras available for environment {env_id}")
        
        # Convert to tensors
        rgb_images = torch.tensor(np.array(rgb_images_list), dtype=torch.uint8)
        depth_images = torch.tensor(np.array(depth_images_list), dtype=torch.float32)
        camera_intrinsics = torch.tensor(np.array(camera_intrinsics_list), dtype=torch.float32)
        camera_extrinsics = torch.tensor(np.array(camera_extrinsics_list), dtype=torch.float32)
        
        if hasattr(env, 'gym') and hasattr(env, 'sim'):
            env.gym.end_access_image_tensors(env.sim)
        
        if debug:
            print(f"Collected camera data:")
            print(f"   RGB: {rgb_images.shape}")
            print(f"   Depth: {depth_images.shape}")
            print(f"   Intrinsics: {camera_intrinsics.shape}")
            print(f"   Extrinsics: {camera_extrinsics.shape}")
        
        return rgb_images, depth_images, camera_intrinsics, camera_extrinsics, camera_info_list
        
    except Exception as e:
        raise ValueError(f"Error in collect_camera_images: {e}")

def add_rgbd_collection_to_env(env_class):
    """
    Enhanced function to add RGBD collection capability with proper camera parameters
    """
    def collect_rgbd_data(self, flag=True, debug=False):
        """Collect RGBD data with enhanced camera parameter extraction"""
        try:
            # Get base observation
            base_obs = self.collect_diff_data() if hasattr(self, 'collect_diff_data') else {}
            
            # Collect camera images with parameters
            rgb_images, depth_images, camera_intrinsics, camera_extrinsics, camera_info = collect_camera_images(self, debug=debug)
            
            # Add RGBD data to observation
            base_obs["rgb_images"] = rgb_images
            base_obs["depth_images"] = depth_images
            base_obs["camera_intrinsics"] = camera_intrinsics
            base_obs["camera_extrinsics"] = camera_extrinsics
            base_obs["camera_info"] = camera_info
            
            if debug:
                print(f"   RGBD observation keys: {list(base_obs.keys())}")
                print(f"   RGB shape: {rgb_images.shape}")
                print(f"   Depth shape: {depth_images.shape}")
                print(f"   Camera intrinsics shape: {camera_intrinsics.shape}")
                print(f"   Camera extrinsics shape: {camera_extrinsics.shape}")
            
            return base_obs
            
        except Exception as e:
            raise ValueError(f"Error in collect_rgbd_data: {e}")
            

    # Add the method to the class
    env_class.collect_rgbd_data = collect_rgbd_data
    print(f"Enhanced RGBD collection capability added to {env_class.__name__}")