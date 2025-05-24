import numpy as np
import torch
import pickle
from pytorch3d.ops import sample_farthest_points
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
import zarr
import ipdb
import os
from .dataset import obs_wrapper, nested_dict_save, create_sample_indices, sample_sequence

class RGBDEpisodeBuffer:
    """Episode buffer that stores RGBD observations along with other data"""
    def __init__(self):
        self.pcs = []
        self.env_state = []
        self.action = []
        self.rgb_images = []
        self.depth_images = []

    def add(self, pc, env_state, action):
        """Add standard observation (for backward compatibility)"""
        self.pcs.append(pc.cpu().numpy())
        self.env_state.append(env_state.cpu().numpy())
        self.action.append(action.cpu().numpy())
        # Add empty RGB/depth placeholders for compatibility
        self.rgb_images.append(np.zeros((1, 1, 3), dtype=np.uint8))
        self.depth_images.append(np.zeros((1, 1), dtype=np.float32))

    def add_rgbd(self, pc, env_state, action, rgb_images, depth_images):
        """Add RGBD observation with images"""
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

class RGBDExperience:
    """Experience replay buffer that handles RGBD data"""
    def __init__(self, sample_pcs_num=1000):
        self.sample_pcs_num = sample_pcs_num
        self.data = {
            "pcs": [], 
            "env_state": [], 
            "action": [],
            "rgb_images": [],
            "depth_images": []
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
        else:
            self.data["pcs"] = np.concatenate([self.data["pcs"], np.array(episode.pcs)])
            self.data["env_state"] = np.concatenate([self.data["env_state"], np.array(episode.env_state)])
            self.data["action"] = np.concatenate([self.data["action"], np.array(episode.action)])
            self.data["rgb_images"] = np.concatenate([self.data["rgb_images"], np.array(episode.rgb_images)])
            self.data["depth_images"] = np.concatenate([self.data["depth_images"], np.array(episode.depth_images)])
        
        new_end = self.data["pcs"].shape[0]
        self.meta["episode_ends"].append(new_end)

    def save(self, path, save_png_samples=True, max_png_samples=10):
        """Save RGBD data and meta in zarr format, optionally save sample PNG images"""
        self.meta["episode_ends"] = np.array(self.meta["episode_ends"])
        zarr_group = zarr.open(path, 'w')
        nested_dict_save(zarr_group, self.data, 'data')
        nested_dict_save(zarr_group, self.meta, 'meta')
        
        # Save additional metadata about RGBD format
        rgbd_meta = {
            "has_rgbd": True,
            "rgb_shape": self.data["rgb_images"].shape if len(self.data["rgb_images"]) > 0 else (0,),
            "depth_shape": self.data["depth_images"].shape if len(self.data["depth_images"]) > 0 else (0,),
            "total_episodes": len(self.meta["episode_ends"]),
            "total_steps": self.data["pcs"].shape[0] if len(self.data["pcs"]) > 0 else 0
        }
        nested_dict_save(zarr_group, rgbd_meta, 'rgbd_meta')
        print(f"Saved RGBD dataset with {rgbd_meta['total_episodes']} episodes and {rgbd_meta['total_steps']} steps")
        
        # Save sample RGB images as PNG files
        if save_png_samples and len(self.data["rgb_images"]) > 0:
            self._save_png_samples(path, max_samples=max_png_samples)

    def _save_png_samples(self, base_path, max_samples=10):
        """Save sample RGB and depth images as PNG files"""
        try:
            import matplotlib.pyplot as plt
            
            # Create PNG samples directory
            png_dir = os.path.join(os.path.dirname(base_path), "png_samples")
            os.makedirs(png_dir, exist_ok=True)
            
            rgb_data = self.data["rgb_images"]
            depth_data = self.data["depth_images"]
            
            # Sample indices evenly across the dataset
            total_steps = rgb_data.shape[0]
            if total_steps == 0:
                print("⚠️ No RGB data to save as PNG")
                return
            
            sample_indices = np.linspace(0, total_steps-1, min(max_samples, total_steps), dtype=int)
            
            print(f"💾 Saving {len(sample_indices)} sample RGB images as PNG...")
            
            for i, step_idx in enumerate(sample_indices):
                rgb_step = rgb_data[step_idx]  # Shape: [num_cameras, height, width, 3]
                depth_step = depth_data[step_idx]  # Shape: [num_cameras, height, width]
                
                if rgb_step.size == 0:
                    continue
                
                num_cameras = rgb_step.shape[0] if len(rgb_step.shape) > 1 else 1
                
                # Create figure for this time step
                fig, axes = plt.subplots(2, num_cameras, figsize=(4*num_cameras, 8))
                if num_cameras == 1:
                    axes = axes.reshape(2, 1)
                elif num_cameras == 0:
                    continue
                
                for cam_idx in range(num_cameras):
                    try:
                        # Get RGB and depth for this camera
                        if len(rgb_step.shape) >= 4:  # [num_cameras, height, width, 3]
                            rgb_img = rgb_step[cam_idx]
                            depth_img = depth_step[cam_idx]
                        else:  # Fallback for different shapes
                            rgb_img = rgb_step
                            depth_img = depth_step
                            
                        # Normalize RGB if needed
                        if rgb_img.dtype == np.uint8:
                            rgb_img_norm = rgb_img.astype(np.float32) / 255.0
                        else:
                            rgb_img_norm = rgb_img
                        
                        # RGB image
                        axes[0, cam_idx].imshow(rgb_img_norm)
                        axes[0, cam_idx].set_title(f"RGB Camera {cam_idx} (Step {step_idx})")
                        axes[0, cam_idx].axis('off')
                        
                        # Depth image
                        if depth_img.size > 1:  # Not dummy data
                            im = axes[1, cam_idx].imshow(depth_img, cmap='viridis')
                            plt.colorbar(im, ax=axes[1, cam_idx])
                        else:
                            # Handle dummy depth data
                            axes[1, cam_idx].imshow(np.zeros_like(rgb_img_norm[:,:,0]), cmap='viridis')
                        axes[1, cam_idx].set_title(f"Depth Camera {cam_idx} (Step {step_idx})")
                        axes[1, cam_idx].axis('off')
                        
                    except Exception as e:
                        print(f"⚠️ Error processing camera {cam_idx}: {e}")
                        continue
                
                plt.tight_layout()
                png_path = os.path.join(png_dir, f"sample_{i:03d}_step_{step_idx:04d}.png")
                plt.savefig(png_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"✅ Saved {len(sample_indices)} PNG samples to {png_dir}/")
            
            # Also save a summary image with all RGB samples in a grid
            self._save_rgb_summary_grid(png_dir, sample_indices)
            
        except Exception as e:
            print(f"❌ Error saving PNG samples: {e}")
            import traceback
            traceback.print_exc()

    def _save_rgb_summary_grid(self, png_dir, sample_indices):
        """Save a summary grid of all RGB samples"""
        try:
            import matplotlib.pyplot as plt
            
            rgb_data = self.data["rgb_images"]
            
            # Create a grid layout
            n_samples = len(sample_indices)
            cols = min(5, n_samples)  # Max 5 columns
            rows = (n_samples + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
            if n_samples == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, step_idx in enumerate(sample_indices):
                if i < len(axes):
                    rgb_step = rgb_data[step_idx]
                    
                    # Use first camera if multiple cameras
                    if len(rgb_step.shape) >= 4:  # [num_cameras, height, width, 3]
                        rgb_img = rgb_step[0]
                    else:
                        rgb_img = rgb_step
                    
                    # Normalize if needed
                    if rgb_img.dtype == np.uint8:
                        rgb_img = rgb_img.astype(np.float32) / 255.0
                    
                    axes[i].imshow(rgb_img)
                    axes[i].set_title(f"Step {step_idx}")
                    axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(sample_indices), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            summary_path = os.path.join(png_dir, "rgb_summary_grid.png")
            plt.savefig(summary_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved RGB summary grid to {summary_path}")
            
        except Exception as e:
            print(f"⚠️ Error creating RGB summary grid: {e}")

class RGBDManipDataset(torch.utils.data.Dataset):
    """PyTorch dataset for RGBD manipulation data"""
    def __init__(self,
                 dataset_path: list,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 use_images: bool = True):
        
        print("Using RGBD Manip Dataset")
        print(f"Loading RGBD data from {dataset_path}")
        
        pcs_list = []
        action_data_list = []
        env_state_list = []
        rgb_images_list = []
        depth_images_list = []
        episode_end_list = []

        init = 0
        for data_path in dataset_path:
            data = zarr.open(data_path, 'r')
            ends = data['meta']['episode_ends'][:]
            episode_end_list.append(ends + init)
            init += ends[-1]
            
            pcs_list.append(data['data']['pcs'])
            action_data_list.append(data['data']['action'])
            env_state_list.append(data['data']['env_state'])
            
            if use_images and 'rgb_images' in data['data']:
                rgb_images_list.append(data['data']['rgb_images'])
                depth_images_list.append(data['data']['depth_images'])
            else:
                # Create dummy images if not available
                dummy_rgb = np.zeros((len(data['data']['pcs']), 1, 1, 3), dtype=np.uint8)
                dummy_depth = np.zeros((len(data['data']['pcs']), 1, 1), dtype=np.float32)
                rgb_images_list.append(dummy_rgb)
                depth_images_list.append(dummy_depth)

        # Concatenate all data
        pcs = np.concatenate(pcs_list, axis=0)
        action_data = np.concatenate(action_data_list, axis=0)
        env_state = np.concatenate(env_state_list, axis=0)
        rgb_images = np.concatenate(rgb_images_list, axis=0)
        depth_images = np.concatenate(depth_images_list, axis=0)
        
        action_train = action_data
        train_data = {
            'pcs': pcs,
            'env_state': env_state,
            'action': action_train,
            'rgb_images': rgb_images,
            'depth_images': depth_images
        }
        
        episode_ends = np.concatenate(episode_end_list, axis=0)
        
        # Compute start and end of each state-action sequence
        indices = create_sample_indices(
            episode_ends=episode_ends,
            obs_length=obs_horizon,
            action_length=pred_horizon,
            pad_after=pred_horizon
        )
        
        print(f"RGBD dataset loaded: {len(indices)} sequences")
        print(f"RGB images shape: {rgb_images.shape}")
        print(f"Depth images shape: {depth_images.shape}")
        
        self.indices = indices
        self.train_data = train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.use_images = use_images

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the start/end indices for this datapoint
        action_start_idx, action_end_idx, obs_start_idx, obs_end_idx = self.indices[idx]

        # Sample sequence from training data - pass all data at once
        sample = sample_sequence(
            train_data=self.train_data,
            obs_length=self.obs_horizon,
            action_length=self.pred_horizon,
            action_start_idx=action_start_idx, 
            action_end_idx=action_end_idx,
            obs_start_idx=obs_start_idx, 
            obs_end_idx=obs_end_idx
        )
        
        # Convert to tensors
        pcs_tensor = torch.from_numpy(sample['pcs']).float()
        env_state_tensor = torch.from_numpy(sample['env_state']).float()
        action_tensor = torch.from_numpy(sample['action']).float()
        
        # Handle RGBD data
        if self.use_images and 'rgb_images' in sample:
            rgb_tensor = torch.from_numpy(sample['rgb_images']).float() / 255.0  # Normalize to [0,1]
            depth_tensor = torch.from_numpy(sample['depth_images']).float()
        else:
            # Create dummy tensors if no RGBD data
            rgb_tensor = torch.zeros((self.obs_horizon, 1, 1, 3), dtype=torch.float32)
            depth_tensor = torch.zeros((self.obs_horizon, 1, 1), dtype=torch.float32)

        return {
            'pcs': pcs_tensor,
            'env_state': env_state_tensor,
            'action': action_tensor,
            'rgb_images': rgb_tensor,
            'depth_images': depth_tensor
        }

def load_rgbd_dataset(dataset_paths: list, **kwargs):
    """Convenience function to load RGBD dataset"""
    return RGBDManipDataset(dataset_paths, **kwargs)

def merge_rgbd_dataset(path_list, to_path):
    """Merge multiple RGBD datasets into one"""
    for i, path in enumerate(path_list):
        dataset_root = zarr.open(path, 'r')
        if i == 0:
            pcs_data = dataset_root['data']['pcs'][:]
            pose_data = dataset_root['data']['env_state'][:]
            action_data = dataset_root['data']['action'][:]
            rgb_data = dataset_root['data']['rgb_images'][:] if 'rgb_images' in dataset_root['data'] else None
            depth_data = dataset_root['data']['depth_images'][:] if 'depth_images' in dataset_root['data'] else None
            meta = dataset_root['meta']['episode_ends'][:]
            cur_len = dataset_root['meta']['episode_ends'][-1]
        else:
            pcs_data = np.concatenate([pcs_data, dataset_root['data']['pcs'][:]])
            pose_data = np.concatenate([pose_data, dataset_root['data']['env_state'][:]])
            action_data = np.concatenate([action_data, dataset_root['data']['action'][:]])
            
            if rgb_data is not None and 'rgb_images' in dataset_root['data']:
                rgb_data = np.concatenate([rgb_data, dataset_root['data']['rgb_images'][:]])
                depth_data = np.concatenate([depth_data, dataset_root['data']['depth_images'][:]])
            
            new_meta = dataset_root['meta']['episode_ends'][:] + cur_len
            cur_len += dataset_root['meta']['episode_ends'][-1]
            meta = np.concatenate([meta, new_meta])
        
        print(f"Merged {path}: pcs {pcs_data.shape}, pose {pose_data.shape}, action {action_data.shape}")
        if rgb_data is not None:
            print(f"  RGB {rgb_data.shape}, depth {depth_data.shape}")

    # Save merged data
    all_data = {
        "pcs": pcs_data, 
        "env_state": pose_data, 
        "action": action_data
    }
    if rgb_data is not None:
        all_data["rgb_images"] = rgb_data
        all_data["depth_images"] = depth_data
    
    all_meta = {"episode_ends": meta}
    zarr_group = zarr.open(to_path, 'w')
    nested_dict_save(zarr_group, all_data, 'data')
    nested_dict_save(zarr_group, all_meta, 'meta')
    
    # Add RGBD metadata
    rgbd_meta = {
        "has_rgbd": rgb_data is not None,
        "merged_from": path_list,
        "total_episodes": len(meta),
        "total_steps": pcs_data.shape[0]
    }
    nested_dict_save(zarr_group, rgbd_meta, 'rgbd_meta')
    print(f"Merged RGBD dataset saved to {to_path}") 