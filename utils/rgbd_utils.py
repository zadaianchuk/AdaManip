"""
RGBD Utilities for AdaManip
Provides utilities to add RGBD data collection to any environment
"""

import torch
import numpy as np
import os
from isaacgym import gymapi


def add_rgbd_collection_to_env(env_class):
    """
    Decorator/function to add RGBD collection capability to any environment class
    """
    def collect_rgbd_data(self, flag=True, debug=False):
        """
        Collect RGBD observations including RGB and depth images from all cameras
        
        Args:
            flag (bool): Whether to normalize point clouds
            debug (bool): Enable debug output
            
        Returns:
            dict: Observation dictionary with RGB and depth images added
        """
        try:
            # Get existing point cloud and proprioception data (reuse existing method)
            if hasattr(self, 'collect_diff_data'):
                base_obs = self.collect_diff_data(flag)
            else:
                # Fallback: create minimal observation
                pc = self.compute_point_cloud_state(depth_bar=2.5, type="fixed") if hasattr(self, 'compute_point_cloud_state') else torch.zeros((self.num_envs, 1000, 3), device=self.device)
                if flag and hasattr(self, 'pc_normalize'):
                    pc = self.pc_normalize(pc)
                    
                # Create basic proprioception info
                joints = getattr(self, 'franka_num_dofs', 7)
                if hasattr(self, 'franka_dof_tensor') and hasattr(self, 'franka_dof_lower_limits_tensor') and hasattr(self, 'franka_dof_upper_limits_tensor'):
                    robotqpose = (2 * (self.franka_dof_tensor[:, :joints, 0] - self.franka_dof_lower_limits_tensor[:joints]) /
                                (self.franka_dof_upper_limits_tensor[:joints] - self.franka_dof_lower_limits_tensor[:joints])) - 1
                    robotqvel = self.franka_dof_tensor[:, :joints, 1]
                else:
                    robotqpose = torch.zeros(self.num_envs, joints, device=self.device)
                    robotqvel = torch.zeros(self.num_envs, joints, device=self.device)
                
                if hasattr(self, 'hand_rigid_body_tensor'):
                    hand_pos = self.hand_rigid_body_tensor[:, :3]
                    hand_rot = self.hand_rigid_body_tensor[:, 3:7]
                else:
                    hand_pos = torch.zeros(self.num_envs, 3, device=self.device)
                    hand_rot = torch.zeros(self.num_envs, 4, device=self.device)
                    hand_rot[:, 3] = 1  # Set w=1 for valid quaternion
                
                proprioception_info = torch.cat([robotqpose, robotqvel, hand_pos, hand_rot], dim=-1)
                prev_actions = getattr(self, 'actions', torch.zeros(self.num_envs, 9, device=self.device))
                
                # DOF state (environment-specific)
                if hasattr(self, 'one_dof_tensor') and hasattr(self, 'two_dof_tensor'):
                    dof_state = torch.cat([self.one_dof_tensor[:,0].unsqueeze(-1), self.two_dof_tensor[:,0].unsqueeze(-1)], dim=-1)
                else:
                    dof_state = torch.zeros(self.num_envs, 2, device=self.device)
                
                base_obs = {
                    "pc": pc,
                    "proprioception": proprioception_info,
                    "dof_state": dof_state,
                    "prev_action": prev_actions
                }

            # Collect RGB and depth images from all cameras
            rgb_images, depth_images = collect_camera_images(self, debug=debug)
            print(f"rgb_images: {rgb_images.shape}, depth_images: {depth_images.shape}")
            
            # Add RGBD data to observation
            base_obs["rgb_images"] = rgb_images
            base_obs["depth_images"] = depth_images
            
            return base_obs
            
        except Exception as e:
            print(f"❌ Error in RGBD collection: {str(e)}")
            if debug:
                import traceback
                traceback.print_exc()
            
            # Return fallback observation without RGBD data
            if hasattr(self, 'collect_diff_data'):
                return self.collect_diff_data(flag)
            else:
                raise e
    
    # Add the method to the class
    env_class.collect_rgbd_data = collect_rgbd_data
    return env_class


def collect_camera_images(env, debug=False):
    """
    Collect RGB and depth images from all available cameras in the environment
    
    Args:
        env: Environment instance
        debug (bool): Enable debug output
        
    Returns:
        tuple: (rgb_images_tensor, depth_images_tensor)
    """
    try:
        # Get camera properties
        camera_props = gymapi.CameraProperties()
        camera_props.width = env.cfg.get("env", {}).get("cam", {}).get("width", 128)
        camera_props.height = env.cfg.get("env", {}).get("cam", {}).get("height", 128)
        
        if debug:
            print(f"📷 Camera resolution: {camera_props.width}x{camera_props.height}")
        
        env.gym.start_access_image_tensors(env.sim)
        
        # Initialize lists to store images for each environment
        rgb_images = []
        depth_images = []
        
        for env_id in range(env.num_envs):
            env_ptr = env.env_ptr_list[env_id]
            env_rgb_images = []
            env_depth_images = []
            
            # Collect from fixed cameras if available
            if hasattr(env, 'fixed_camera_handle_list') and hasattr(env, 'num_cam'):
                try:
                    for cam_id in range(env.num_cam):
                        if env_id < len(env.fixed_camera_handle_list) and cam_id < len(env.fixed_camera_handle_list[env_id]):
                            fixed_camera_handle = env.fixed_camera_handle_list[env_id][cam_id]
                            
                            # Get RGB image
                            rgb_array = env.gym.get_camera_image(env.sim, env_ptr, fixed_camera_handle, gymapi.IMAGE_COLOR)
                            rgb_image = rgb_array.reshape(camera_props.height, camera_props.width, 4)[:, :, :3]  # Remove alpha
                            env_rgb_images.append(rgb_image)
                            
                            # Get depth image
                            depth_array = env.gym.get_camera_image(env.sim, env_ptr, fixed_camera_handle, gymapi.IMAGE_DEPTH)
                            depth_image = depth_array.reshape(camera_props.height, camera_props.width)
                            env_depth_images.append(depth_image)
                            
                            if debug:
                                print(f"📸 Collected fixed camera {cam_id} for env {env_id}")
                                
                except Exception as e:
                    if debug:
                        print(f"⚠️ Error collecting from fixed cameras: {str(e)}")
            
            # Collect from hand camera if available
            if hasattr(env, 'hand_camera_handle_list'):
                try:
                    if env_id < len(env.hand_camera_handle_list):
                        hand_camera_handle = env.hand_camera_handle_list[env_id]
                        
                        # Hand camera RGB
                        hand_rgb_array = env.gym.get_camera_image(env.sim, env_ptr, hand_camera_handle, gymapi.IMAGE_COLOR)
                        hand_rgb_image = hand_rgb_array.reshape(camera_props.height, camera_props.width, 4)[:, :, :3]
                        env_rgb_images.append(hand_rgb_image)
                        
                        # Hand camera depth
                        hand_depth_array = env.gym.get_camera_image(env.sim, env_ptr, hand_camera_handle, gymapi.IMAGE_DEPTH)
                        hand_depth_image = hand_depth_array.reshape(camera_props.height, camera_props.width)
                        env_depth_images.append(hand_depth_image)
                        
                        if debug:
                            print(f"📸 Collected hand camera for env {env_id}")
                            
                except Exception as e:
                    if debug:
                        print(f"⚠️ Error collecting from hand camera: {str(e)}")
            
            # If no cameras were found, create dummy images
            if len(env_rgb_images) == 0:
                if debug:
                    print(f"⚠️ No cameras found for env {env_id}, creating dummy images")
                env_rgb_images.append(np.zeros((camera_props.height, camera_props.width, 3), dtype=np.uint8))
                env_depth_images.append(np.zeros((camera_props.height, camera_props.width), dtype=np.float32))
            
            # Convert to numpy arrays and add to environment list
            rgb_images.append(np.array(env_rgb_images))
            depth_images.append(np.array(env_depth_images))
        
        env.gym.end_access_image_tensors(env.sim)
        
        # Convert to tensors
        rgb_images_tensor = torch.tensor(np.array(rgb_images), device=env.device, dtype=torch.uint8)
        depth_images_tensor = torch.tensor(np.array(depth_images), device=env.device, dtype=torch.float32)

        if debug:
            print(f"📊 Final tensor shapes - RGB: {rgb_images_tensor.shape}, Depth: {depth_images_tensor.shape}")
        
        return rgb_images_tensor, depth_images_tensor
        
    except Exception as e:
        print(f"❌ Error collecting camera images: {str(e)}")
        if debug:
            import traceback
            traceback.print_exc()
        
        # Return dummy tensors as fallback
        dummy_rgb = torch.zeros((env.num_envs, 1, camera_props.height, camera_props.width, 3), 
                               device=env.device, dtype=torch.uint8)
        dummy_depth = torch.zeros((env.num_envs, 1, camera_props.height, camera_props.width), 
                                 device=env.device, dtype=torch.float32)
        return dummy_rgb, dummy_depth


def apply_rgbd_to_all_environments():
    """
    Apply RGBD collection to all environment classes
    This should be called after importing all environment modules
    """
    try:
        # Import all environment classes
        import sys
        import os
        
        # Add envs directory to path if not already there
        envs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs")
        if envs_dir not in sys.path:
            sys.path.append(envs_dir)
        
        # List of environment modules to enhance
        env_modules = [
            'open_bottle', 'open_door', 'open_lamp', 'open_microwave',
            'open_pen', 'open_pressurecooker', 'open_window', 'open_coffeemachine',
            'open_safe'
        ]
        
        enhanced_envs = []
        
        for module_name in env_modules:
            try:
                module = __import__(module_name)
                
                # Find the environment class in the module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        hasattr(attr, '__bases__') and 
                        any('BaseEnv' in str(base) for base in attr.__bases__)):
                        
                        # Apply RGBD enhancement if not already present
                        if not hasattr(attr, 'collect_rgbd_data'):
                            add_rgbd_collection_to_env(attr)
                            enhanced_envs.append(f"{module_name}.{attr_name}")
                            
            except ImportError as e:
                print(f"⚠️ Could not import {module_name}: {e}")
            except Exception as e:
                print(f"⚠️ Error processing {module_name}: {e}")
        
        if enhanced_envs:
            print(f"✅ Added RGBD collection to: {', '.join(enhanced_envs)}")
        else:
            print("ℹ️ No environments were enhanced (possibly already enhanced)")
            
    except Exception as e:
        print(f"❌ Error applying RGBD to environments: {str(e)}")


def validate_rgbd_environment(env):
    """
    Validate that an environment has proper RGBD collection capability
    
    Args:
        env: Environment instance
        
    Returns:
        dict: Validation results
    """
    results = {
        'has_rgbd_method': False,
        'can_collect_rgbd': False,
        'camera_info': {},
        'errors': []
    }
    
    try:
        # Check if RGBD method exists
        results['has_rgbd_method'] = hasattr(env, 'collect_rgbd_data')
        
        if not results['has_rgbd_method']:
            results['errors'].append("Missing collect_rgbd_data method")
            return results
        
        # Try to collect RGBD data
        env.reset()
        rgbd_obs = env.collect_rgbd_data()
        
        # Check observation structure
        required_keys = ['pc', 'proprioception', 'dof_state', 'prev_action', 'rgb_images', 'depth_images']
        missing_keys = [key for key in required_keys if key not in rgbd_obs]
        
        if missing_keys:
            results['errors'].append(f"Missing observation keys: {missing_keys}")
        else:
            results['can_collect_rgbd'] = True
            
            # Collect camera info
            rgb_images = rgbd_obs['rgb_images']
            depth_images = rgbd_obs['depth_images']
            
            results['camera_info'] = {
                'rgb_shape': tuple(rgb_images.shape),
                'depth_shape': tuple(depth_images.shape),
                'rgb_dtype': str(rgb_images.dtype),
                'depth_dtype': str(depth_images.dtype),
                'num_cameras': rgb_images.shape[1] if len(rgb_images.shape) > 1 else 0,
                'image_resolution': (rgb_images.shape[-2], rgb_images.shape[-1]) if len(rgb_images.shape) >= 2 else (0, 0)
            }
        
    except Exception as e:
        results['errors'].append(f"Exception during RGBD collection: {str(e)}")
    
    return results


def debug_rgbd_collection(env, save_samples=True, output_dir="debug_rgbd"):
    """
    Debug RGBD collection and optionally save sample images
    
    Args:
        env: Environment instance
        save_samples (bool): Whether to save sample images
        output_dir (str): Directory to save debug outputs
    """
    print("\n" + "="*50)
    print("RGBD COLLECTION DEBUG")
    print("="*50)
    
    # Validate environment
    validation = validate_rgbd_environment(env)
    
    print(f"Has RGBD method: {validation['has_rgbd_method']}")
    print(f"Can collect RGBD: {validation['can_collect_rgbd']}")
    
    if validation['camera_info']:
        info = validation['camera_info']
        print(f"Camera info:")
        print(f"  - RGB shape: {info['rgb_shape']}")
        print(f"  - Depth shape: {info['depth_shape']}")
        print(f"  - Number of cameras: {info['num_cameras']}")
        print(f"  - Image resolution: {info['image_resolution']}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    # Save sample images if requested and possible
    if save_samples and validation['can_collect_rgbd']:
        try:
            rgbd_obs = env.collect_rgbd_data()
            rgb_images = rgbd_obs['rgb_images']
            depth_images = rgbd_obs['depth_images']
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Save using matplotlib
            import matplotlib.pyplot as plt
            
            if rgb_images.numel() > 0 and depth_images.numel() > 0:
                # Get first environment, all cameras
                sample_rgb = rgb_images[0].cpu().numpy()  # [num_cameras, height, width, 3]
                sample_depth = depth_images[0].cpu().numpy()  # [num_cameras, height, width]
                
                # Skip if dummy data
                if len(sample_rgb.shape) >= 3 and (sample_rgb.shape[1] <= 1 or sample_rgb.shape[2] <= 1):
                    print("⚠️ RGB images appear to be dummy data (1x1 size), skipping PNG save")
                else:
                    # Save individual camera images as PNG
                    save_individual_camera_pngs(sample_rgb, sample_depth, output_dir)
                    
                    # Create overview image
                    save_overview_image(sample_rgb, sample_depth, output_dir)
                
                print(f"✅ Sample images saved to {output_dir}/")
        
        except Exception as e:
            print(f"❌ Error saving sample images: {str(e)}")
    
    print("="*50)
    return validation

def save_individual_camera_pngs(rgb_images, depth_images, output_dir):
    """Save individual camera images as PNG files"""
    import matplotlib.pyplot as plt
    
    num_cameras = rgb_images.shape[0] if len(rgb_images.shape) > 1 else 1
    
    for cam_idx in range(num_cameras):
        try:
            # Get RGB and depth for this camera
            if len(rgb_images.shape) >= 4:  # [num_cameras, height, width, 3]
                rgb_img = rgb_images[cam_idx]
                depth_img = depth_images[cam_idx]
            else:  # Single camera case
                rgb_img = rgb_images
                depth_img = depth_images
            
            # Normalize RGB
            if rgb_img.dtype == np.uint8:
                rgb_img_norm = rgb_img.astype(np.float32) / 255.0
            else:
                rgb_img_norm = np.clip(rgb_img, 0, 1)
            
            # Save RGB as PNG
            rgb_path = os.path.join(output_dir, f"debug_rgb_camera_{cam_idx}.png")
            plt.figure(figsize=(8, 6))
            plt.imshow(rgb_img_norm)
            plt.title(f"RGB Camera {cam_idx}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(rgb_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save depth as PNG
            depth_path = os.path.join(output_dir, f"debug_depth_camera_{cam_idx}.png")
            plt.figure(figsize=(8, 6))
            if depth_img.size > 1:  # Not dummy data
                plt.imshow(depth_img, cmap='viridis')
                plt.colorbar()
            else:
                plt.imshow(np.zeros_like(rgb_img_norm[:,:,0]), cmap='viridis')
            plt.title(f"Depth Camera {cam_idx}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(depth_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   📸 Saved camera {cam_idx} images: {rgb_path}, {depth_path}")
            
        except Exception as e:
            print(f"⚠️ Error saving camera {cam_idx}: {e}")

def save_overview_image(rgb_images, depth_images, output_dir):
    """Save overview image with all cameras"""
    import matplotlib.pyplot as plt
    
    num_cameras = rgb_images.shape[0] if len(rgb_images.shape) > 1 else 1
    
    try:
        # Create overview figure
        fig, axes = plt.subplots(2, num_cameras, figsize=(4*num_cameras, 8))
        if num_cameras == 1:
            axes = axes.reshape(2, 1)
        
        for cam_idx in range(num_cameras):
            # Get images for this camera
            if len(rgb_images.shape) >= 4:
                rgb_img = rgb_images[cam_idx]
                depth_img = depth_images[cam_idx]
            else:
                rgb_img = rgb_images
                depth_img = depth_images
            
            # Normalize RGB
            if rgb_img.dtype == np.uint8:
                rgb_img_norm = rgb_img.astype(np.float32) / 255.0
            else:
                rgb_img_norm = np.clip(rgb_img, 0, 1)
            
            # RGB subplot
            axes[0, cam_idx].imshow(rgb_img_norm)
            axes[0, cam_idx].set_title(f"RGB Camera {cam_idx}")
            axes[0, cam_idx].axis('off')
            
            # Depth subplot
            if depth_img.size > 1:
                im = axes[1, cam_idx].imshow(depth_img, cmap='viridis')
                plt.colorbar(im, ax=axes[1, cam_idx])
            else:
                axes[1, cam_idx].imshow(np.zeros_like(rgb_img_norm[:,:,0]), cmap='viridis')
            axes[1, cam_idx].set_title(f"Depth Camera {cam_idx}")
            axes[1, cam_idx].axis('off')
        
        plt.tight_layout()
        overview_path = os.path.join(output_dir, "debug_rgbd_overview.png")
        plt.savefig(overview_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved overview image: {overview_path}")
        
    except Exception as e:
        print(f"⚠️ Error creating overview image: {e}") 