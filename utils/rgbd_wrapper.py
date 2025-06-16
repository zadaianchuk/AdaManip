"""
RGBD Wrapper for Data Collection Methods
========================================

This module provides a decorator that can wrap original data collection methods
(collect_manip_data, collect_grasp_data) to automatically use RGBD data structures
and save to the proper adamanip_d3fields directory structure.

The wrapper preserves all the original task-specific logic while enhancing it
with RGBD capabilities.
"""

import os
import functools
from typing import Callable, Any
from dataset.dataset_rgbd import RGBDExperience, RGBDEpisodeBuffer
from dataset.dataset import obs_wrapper


def rgbd_data_collection_wrapper(original_method: Callable) -> Callable:
    """
    Decorator that wraps original data collection methods to use RGBD data structures.
    
    This wrapper:
    1. Replaces Episode_Buffer with RGBDEpisodeBuffer
    2. Replaces Experience with RGBDExperience  
    3. Intercepts data collection calls to use RGBD versions
    4. Modifies save paths to use adamanip_d3fields structure
    5. Handles per-environment saving
    
    Args:
        original_method: The original collect_manip_data or collect_grasp_data method
        
    Returns:
        Wrapped method that uses RGBD data collection
    """
    
    @functools.wraps(original_method)
    def wrapper(self):
        import sys
        print(f"RGBD Wrapper: Enhancing {original_method.__name__} with RGBD capabilities")
        
        # Store original references
        original_collect_single = getattr(self.env, 'collect_single_diff_data', None)
        original_collect_diff = getattr(self.env, 'collect_diff_data', None)
        
        # Store original method on environment for RGBD collection to use (avoid recursion)
        self.env._original_collect_diff_data = original_collect_diff
        
        # Create RGBD data collection interceptor
        def rgbd_collect_single_diff_data(env_id):
            """Intercept single environment data collection to use RGBD"""
            try:
                rgbd_obs = self.env.collect_rgbd_data()
                # Extract data for specific environment
                if isinstance(rgbd_obs, dict) and 'pc' in rgbd_obs:
                    single_obs = {}
                    for key, value in rgbd_obs.items():
                        if hasattr(value, '__getitem__') and len(value) > env_id:
                            single_obs[key] = value[env_id]
                        else:
                            single_obs[key] = value
                    return single_obs
                else:
                    # Fallback to original if RGBD fails
                    return original_collect_single(env_id) if original_collect_single else rgbd_obs
            except Exception as e:
                print(f"WARNING: RGBD collection failed for env {env_id}: {e}, falling back to original")
                return original_collect_single(env_id) if original_collect_single else {}
        
        def rgbd_collect_diff_data():
            """Intercept general data collection to use RGBD"""
            try:
                return self.env.collect_rgbd_data()
            except Exception as e:
                print(f"WARNING: RGBD collection failed: {e}, falling back to original")
                return original_collect_diff() if original_collect_diff else {}
        
        # Temporarily replace collection methods
        self.env.collect_single_diff_data = rgbd_collect_single_diff_data
        self.env.collect_diff_data = rgbd_collect_diff_data
        
        # Create RGBD-enhanced data structures
        class RGBDEnhancedEpisodeBuffer:
            """Episode buffer that automatically handles RGBD data"""
            def __init__(self):
                self._buffer = RGBDEpisodeBuffer()
                self._env = None  # Will be set by the wrapper
                self._env_id = None  # Will be set by the wrapper
                
            def add(self, pc, env_state, action, **kwargs):
                """Add data, automatically detecting if it's RGBD or regular"""
                # Check if we have RGBD data in kwargs or if pc contains RGBD info
                if 'rgb_images' in kwargs or 'depth_images' in kwargs:
                    # Direct RGBD data
                    self._buffer.add_rgbd(
                        pc, env_state, action,
                        kwargs.get('rgb_images'),
                        kwargs.get('depth_images'),
                        kwargs.get('segmentation_masks'),
                        kwargs.get('camera_intrinsics'),
                        kwargs.get('camera_extrinsics'),
                        kwargs.get('camera_info')
                    )
                else:
                    # Try to extract RGBD from environment if available
                    if self._env is not None:
                        try:
                            rgbd_obs = self._env.collect_rgbd_data()
                            # Be more careful with tensor checks to avoid boolean ambiguity
                            has_rgb = isinstance(rgbd_obs, dict) and 'rgb_images' in rgbd_obs.keys()
                            if has_rgb and rgbd_obs['rgb_images'] is not None:
                                # We have RGBD data available
                                env_id = self._env_id if self._env_id is not None else 0
                                rgb_data = rgbd_obs['rgb_images']
                                depth_data = rgbd_obs['depth_images']
                                
                                # Extract data for this environment
                                rgb_env = rgb_data[env_id] if len(rgb_data) > env_id else rgb_data[0]
                                depth_env = depth_data[env_id] if len(depth_data) > env_id else depth_data[0]
                                
                                # Get optional data
                                seg_data = rgbd_obs.get('segmentation_masks')
                                seg_env = seg_data[env_id] if seg_data is not None and len(seg_data) > env_id else None
                                
                                intrinsics_data = rgbd_obs.get('camera_intrinsics')
                                intrinsics_env = intrinsics_data[env_id] if intrinsics_data is not None and len(intrinsics_data) > env_id else None
                                
                                extrinsics_data = rgbd_obs.get('camera_extrinsics')
                                extrinsics_env = extrinsics_data[env_id] if extrinsics_data is not None and len(extrinsics_data) > env_id else None
                                
                                self._buffer.add_rgbd(
                                    pc, env_state, action,
                                    rgb_env, depth_env, seg_env,
                                    intrinsics_env, extrinsics_env,
                                    rgbd_obs.get('camera_info', None)
                                )
                            else:
                                # Fallback to regular add
                                self._buffer.add(pc, env_state, action)
                        except Exception as e:
                            print(f"WARNING:  RGBD collection failed in episode buffer: {e}, using regular add")
                            # Final fallback
                            self._buffer.add(pc, env_state, action)
                    else:
                        # No environment available, use regular add
                        self._buffer.add(pc, env_state, action)
            
            def __getattr__(self, name):
                return getattr(self._buffer, name)
        
        class RGBDEnhancedExperience:
            """Experience that automatically handles RGBD data and saves properly"""
            def __init__(self):
                print(f"SETUP: Creating RGBDEnhancedExperience instance")
                self._experiences = []  # List of RGBDExperience objects per environment
                self._rgbd_data_dir = "./adamanip_d3fields"  # Default for backward compatibility
                
            def append(self, episode_buffer):
                """Add an episode buffer"""
                # Always create a proper RGBDExperience and append the episode buffer to it
                rgbd_exp = RGBDExperience()
                if hasattr(episode_buffer, '_buffer'):
                    # episode_buffer is RGBDEnhancedEpisodeBuffer, get the actual buffer
                    rgbd_exp.append(episode_buffer._buffer)
                else:
                    # episode_buffer is a regular buffer
                    rgbd_exp.append(episode_buffer)
                self._experiences.append(rgbd_exp)
            
            def save(self, save_path):
                """Save using RGBD format to proper directory structure - IGNORES original save_path"""
                print(f"TARGET: RGBD Wrapper: Intercepting save call (original path: {save_path})")
                print(f"TARGET: RGBD Wrapper: Redirecting to adamanip_d3fields structure")
                print(f"DEBUG: Debug: save() called with {len(self._experiences)} experiences")
                
                # Extract task name and create proper directory structure
                task_name = getattr(self, '_task_name', 'Unknown')
                
                # Map task names to directory names
                task_mapping = {
                    'OpenCoffeeMachine': 'OpenCoffeeMachine',
                    'OpenDoor': 'OpenDoor', 
                    'OpenBottle': 'OpenBottle',
                    'OpenLamp': 'OpenLamp',
                    'OpenMicroWave': 'OpenMicroWave',
                    'OpenPen': 'OpenPen',
                    'OpenSafe': 'OpenSafe',
                    'OpenWindow': 'OpenWindow',
                    'OpenPressureCooker': 'OpenPressureCooker'
                }
                
                dir_name = task_mapping.get(task_name, task_name)
                
                # Determine if this is grasp or manip data from original save path
                is_grasp = 'grasp' in save_path.lower()
                data_type = 'grasp' if is_grasp else 'manipulation'
                
                print(f"TARGET: Task: {task_name}, Data type: {data_type}")
                
                # Save each environment separately to configurable RGBD data directory
                saved_count = 0
                for env_id, experience in enumerate(self._experiences):
                    env_dir = f'{self._rgbd_data_dir}/{dir_name}/{data_type}_env_{env_id}'
                    os.makedirs(env_dir, exist_ok=True)
                    
                    try:
                        if hasattr(experience, 'save_png_npy'):
                            print(f"DEBUG: Debug: Calling save_png_npy for env {env_id} with {len(experience.data.get('rgb_images', []))} RGB images")
                            experience.save_png_npy(env_dir)
                            print(f"SUCCESS: Saved RGBD {data_type} data for env {env_id} to {env_dir}")
                            saved_count += 1
                        else:
                            print(f"WARNING:  Experience for env {env_id} doesn't support RGBD saving")
                            print(f"DEBUG: Debug: Experience type: {type(experience)}")
                            print(f"DEBUG: Debug: Experience attributes: {dir(experience)}")
                    except Exception as e:
                        print(f"ERROR: Failed to save RGBD data for env {env_id}: {e}")
                        import traceback
                        print(f"DEBUG: Debug traceback: {traceback.format_exc()}")
                
                print(f"COMPLETE: RGBD Wrapper: Successfully saved {saved_count} environments to {self._rgbd_data_dir}!")
                
                # Also create a summary file in the original location for reference
                try:
                    original_dir = os.path.dirname(save_path)
                    os.makedirs(original_dir, exist_ok=True)
                    summary_file = os.path.join(original_dir, 'rgbd_redirect_info.txt')
                    with open(summary_file, 'w') as f:
                        f.write(f"RGBD data collection completed!\n")
                        f.write(f"Original save path: {save_path}\n")
                        f.write(f"Actual save location: {self._rgbd_data_dir}/{dir_name}/\n")
                        f.write(f"Task: {task_name}\n")
                        f.write(f"Data type: {data_type}\n")
                        f.write(f"Environments saved: {saved_count}\n")
                        f.write(f"Saved directories:\n")
                        for env_id in range(saved_count):
                            f.write(f"  - {self._rgbd_data_dir}/{dir_name}/{data_type}_env_{env_id}/\n")
                    print(f"📝 Created redirect info file: {summary_file}")
                except Exception as e:
                    print(f"WARNING:  Could not create redirect info file: {e}")
        
        # Monkey patch the data structures in the method's scope
        import dataset.dataset as dataset_module
        original_Experience = dataset_module.Experience
        original_Episode_Buffer = dataset_module.Episode_Buffer
        
        # Create a factory function that sets environment reference
        # We need to track which environment ID each buffer is for
        buffer_counter = [0]  # Use list to make it mutable in closure
        
        def create_enhanced_episode_buffer():
            buffer = RGBDEnhancedEpisodeBuffer()
            buffer._env = self.env
            buffer._env_id = buffer_counter[0] % self.env.num_envs  # Cycle through env IDs
            buffer_counter[0] += 1
            return buffer
        
        # Temporarily replace with RGBD versions
        dataset_module.Experience = RGBDEnhancedExperience
        dataset_module.Episode_Buffer = create_enhanced_episode_buffer
        
        # Also replace in the manipulation object's namespace if they're imported there
        if hasattr(self, 'Experience'):
            original_self_Experience = self.Experience
            self.Experience = RGBDEnhancedExperience
        else:
            original_self_Experience = None
            
        if hasattr(self, 'Episode_Buffer'):
            original_self_Episode_Buffer = self.Episode_Buffer  
            self.Episode_Buffer = create_enhanced_episode_buffer
        else:
            original_self_Episode_Buffer = None
        
        # Also replace in the manipulation module's namespace (for direct imports)
        manipulation_module = self.__class__.__module__
        if manipulation_module in sys.modules:
            import sys
            mod = sys.modules[manipulation_module]
            original_mod_Experience = getattr(mod, 'Experience', None)
            original_mod_Episode_Buffer = getattr(mod, 'Episode_Buffer', None)
            
            if original_mod_Experience:
                setattr(mod, 'Experience', RGBDEnhancedExperience)
                print(f"SETUP: Replaced Experience in module {manipulation_module}")
            if original_mod_Episode_Buffer:
                setattr(mod, 'Episode_Buffer', create_enhanced_episode_buffer)
                print(f"SETUP: Replaced Episode_Buffer in module {manipulation_module}")
        else:
            original_mod_Experience = None
            original_mod_Episode_Buffer = None
        
        try:
            # Set task name for proper saving
            if hasattr(self, 'env') and hasattr(self.env, 'task_name'):
                task_name = self.env.task_name
            elif hasattr(self, 'cfg') and 'task' in self.cfg and 'task_name' in self.cfg['task']:
                task_name = self.cfg['task']['task_name']
            else:
                # Try to infer from class name
                class_name = self.__class__.__name__
                if 'CoffeeMachine' in class_name:
                    task_name = 'OpenCoffeeMachine'
                elif 'Door' in class_name:
                    task_name = 'OpenDoor'
                elif 'Bottle' in class_name:
                    task_name = 'OpenBottle'
                elif 'Lamp' in class_name:
                    task_name = 'OpenLamp'
                elif 'MicroWave' in class_name:
                    task_name = 'OpenMicroWave'
                elif 'Pen' in class_name:
                    task_name = 'OpenPen'
                elif 'Safe' in class_name:
                    task_name = 'OpenSafe'
                elif 'Window' in class_name:
                    task_name = 'OpenWindow'
                elif 'PressureCooker' in class_name:
                    task_name = 'OpenPressureCooker'
                else:
                    task_name = 'Unknown'
            
            # Store task name for saving - set it as a class attribute AND instance attribute
            RGBDEnhancedExperience._task_name = task_name
            # Also set it on any existing instances
            for attr_name in dir(self):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, RGBDEnhancedExperience):
                    attr_value._task_name = task_name
            
            # Call the original method - it will now use RGBD data!
            print(f"START: Executing original {original_method.__name__} with RGBD enhancements...")
            print(f"DEBUG: Debug: Experience class is now {dataset_module.Experience}")
            print(f"DEBUG: Debug: Episode_Buffer class is now {dataset_module.Episode_Buffer}")
            result = original_method()
            print(f"SUCCESS: RGBD-enhanced {original_method.__name__} completed successfully")
            
        finally:
            # Restore original methods and classes
            self.env.collect_single_diff_data = original_collect_single
            self.env.collect_diff_data = original_collect_diff
            dataset_module.Experience = original_Experience
            dataset_module.Episode_Buffer = original_Episode_Buffer
            
            if original_self_Experience is not None:
                self.Experience = original_self_Experience
            if original_self_Episode_Buffer is not None:
                self.Episode_Buffer = original_self_Episode_Buffer
            
            # Restore manipulation module namespace
            if manipulation_module in sys.modules:
                mod = sys.modules[manipulation_module]
                if original_mod_Experience:
                    setattr(mod, 'Experience', original_mod_Experience)
                if original_mod_Episode_Buffer:
                    setattr(mod, 'Episode_Buffer', original_mod_Episode_Buffer)
        
        return result
    
    return wrapper


def rgbd_data_collection_wrapper_with_config(original_method: Callable, rgbd_data_dir: str) -> Callable:
    """
    Version of the RGBD wrapper that accepts configurable data directory.
    
    Args:
        original_method: The original collect_manip_data or collect_grasp_data method
        rgbd_data_dir: Base directory for saving RGBD data
        
    Returns:
        Wrapped method that uses RGBD data collection with configurable directory
    """
    
    @functools.wraps(original_method)
    def wrapper(self):
        import sys
        print(f"VIDEO: RGBD Wrapper: Enhancing {original_method.__name__} with RGBD capabilities")
        print(f"DIR:  Using RGBD data directory: {rgbd_data_dir}")
        
        # Store original references
        original_collect_single = getattr(self.env, 'collect_single_diff_data', None)
        original_collect_diff = getattr(self.env, 'collect_diff_data', None)
        
        # Store original method on environment for RGBD collection to use (avoid recursion)
        self.env._original_collect_diff_data = original_collect_diff
        
        # Create RGBD data collection interceptor
        def rgbd_collect_single_diff_data(env_id):
            """Intercept single environment data collection to use RGBD"""
            try:
                rgbd_obs = self.env.collect_rgbd_data()
                # Extract data for specific environment
                if isinstance(rgbd_obs, dict) and 'pc' in rgbd_obs:
                    single_obs = {}
                    for key, value in rgbd_obs.items():
                        if hasattr(value, '__getitem__') and len(value) > env_id:
                            single_obs[key] = value[env_id]
                        else:
                            single_obs[key] = value
                    return single_obs
                else:
                    # Fallback to original if RGBD fails
                    return original_collect_single(env_id) if original_collect_single else rgbd_obs
            except Exception as e:
                print(f"WARNING:  RGBD collection failed for env {env_id}: {e}, falling back to original")
                return original_collect_single(env_id) if original_collect_single else {}
        
        def rgbd_collect_diff_data():
            """Intercept general data collection to use RGBD"""
            try:
                return self.env.collect_rgbd_data()
            except Exception as e:
                print(f"WARNING:  RGBD collection failed: {e}, falling back to original")
                return original_collect_diff() if original_collect_diff else {}
        
        # Temporarily replace collection methods
        self.env.collect_single_diff_data = rgbd_collect_single_diff_data
        self.env.collect_diff_data = rgbd_collect_diff_data
        
        # Store rgbd_data_dir for use in nested classes
        wrapper_config = {'rgbd_data_dir': rgbd_data_dir}
        
        # Create RGBD-enhanced data structures
        class RGBDEnhancedEpisodeBuffer:
            """Episode buffer that automatically handles RGBD data"""
            def __init__(self):
                self._buffer = RGBDEpisodeBuffer()
                self._env = None  # Will be set by the wrapper
                self._env_id = None  # Will be set by the wrapper
                
            def add(self, pc, env_state, action, **kwargs):
                """Add data, automatically detecting if it's RGBD or regular"""
                # Check if we have RGBD data in kwargs or if pc contains RGBD info
                if 'rgb_images' in kwargs or 'depth_images' in kwargs:
                    # Direct RGBD data
                    self._buffer.add_rgbd(
                        pc, env_state, action,
                        kwargs.get('rgb_images'),
                        kwargs.get('depth_images'),
                        kwargs.get('segmentation_masks'),
                        kwargs.get('camera_intrinsics'),
                        kwargs.get('camera_extrinsics'),
                        kwargs.get('camera_info')
                    )
                else:
                    # Try to extract RGBD from environment if available
                    if self._env is not None:
                        try:
                            rgbd_obs = self._env.collect_rgbd_data()
                            # Be more careful with tensor checks to avoid boolean ambiguity
                            has_rgb = isinstance(rgbd_obs, dict) and 'rgb_images' in rgbd_obs.keys()
                            if has_rgb and rgbd_obs['rgb_images'] is not None:
                                # We have RGBD data available
                                env_id = self._env_id if self._env_id is not None else 0
                                rgb_data = rgbd_obs['rgb_images']
                                depth_data = rgbd_obs['depth_images']
                                
                                # Extract data for this environment
                                rgb_env = rgb_data[env_id] if len(rgb_data) > env_id else rgb_data[0]
                                depth_env = depth_data[env_id] if len(depth_data) > env_id else depth_data[0]
                                
                                # Get optional data
                                seg_data = rgbd_obs.get('segmentation_masks')
                                seg_env = seg_data[env_id] if seg_data is not None and len(seg_data) > env_id else None
                                
                                intrinsics_data = rgbd_obs.get('camera_intrinsics')
                                intrinsics_env = intrinsics_data[env_id] if intrinsics_data is not None and len(intrinsics_data) > env_id else None
                                
                                extrinsics_data = rgbd_obs.get('camera_extrinsics')
                                extrinsics_env = extrinsics_data[env_id] if extrinsics_data is not None and len(extrinsics_data) > env_id else None
                                
                                self._buffer.add_rgbd(
                                    pc, env_state, action,
                                    rgb_env, depth_env, seg_env,
                                    intrinsics_env, extrinsics_env,
                                    rgbd_obs.get('camera_info', None)
                                )
                            else:
                                # Fallback to regular add
                                self._buffer.add(pc, env_state, action)
                        except Exception as e:
                            print(f"WARNING:  RGBD collection failed in episode buffer: {e}, using regular add")
                            # Final fallback
                            self._buffer.add(pc, env_state, action)
                    else:
                        # No environment available, use regular add
                        self._buffer.add(pc, env_state, action)
            
            def __getattr__(self, name):
                return getattr(self._buffer, name)
        
        class RGBDEnhancedExperience:
            """Experience that automatically handles RGBD data and saves properly"""
            def __init__(self):
                print(f"SETUP: Creating RGBDEnhancedExperience instance")
                self._experiences = []  # List of RGBDExperience objects per environment
                self._rgbd_data_dir = wrapper_config['rgbd_data_dir']
                
            def append(self, episode_buffer):
                """Add an episode buffer"""
                # Always create a proper RGBDExperience and append the episode buffer to it
                rgbd_exp = RGBDExperience()
                if hasattr(episode_buffer, '_buffer'):
                    # episode_buffer is RGBDEnhancedEpisodeBuffer, get the actual buffer
                    rgbd_exp.append(episode_buffer._buffer)
                else:
                    # episode_buffer is a regular buffer
                    rgbd_exp.append(episode_buffer)
                self._experiences.append(rgbd_exp)
            
            def save(self, save_path):
                """Save using RGBD format to proper directory structure - IGNORES original save_path"""
                print(f"TARGET: RGBD Wrapper: Intercepting save call (original path: {save_path})")
                print(f"TARGET: RGBD Wrapper: Redirecting to {self._rgbd_data_dir} structure")
                print(f"DEBUG: Debug: save() called with {len(self._experiences)} experiences")
                
                # Extract task name and create proper directory structure
                task_name = getattr(self, '_task_name', 'Unknown')
                
                # Map task names to directory names
                task_mapping = {
                    'OpenCoffeeMachine': 'OpenCoffeeMachine',
                    'OpenDoor': 'OpenDoor', 
                    'OpenBottle': 'OpenBottle',
                    'OpenLamp': 'OpenLamp',
                    'OpenMicroWave': 'OpenMicroWave',
                    'OpenPen': 'OpenPen',
                    'OpenSafe': 'OpenSafe',
                    'OpenWindow': 'OpenWindow',
                    'OpenPressureCooker': 'OpenPressureCooker'
                }
                
                dir_name = task_mapping.get(task_name, task_name)
                
                # Determine if this is grasp or manip data from original save path
                is_grasp = 'grasp' in save_path.lower()
                data_type = 'grasp' if is_grasp else 'manipulation'
                
                print(f"TARGET: Task: {task_name}, Data type: {data_type}")
                
                # Save each environment separately to configurable RGBD data directory
                saved_count = 0
                for env_id, experience in enumerate(self._experiences):
                    env_dir = f'{self._rgbd_data_dir}/{dir_name}/{data_type}_env_{env_id}'
                    os.makedirs(env_dir, exist_ok=True)
                    
                    try:
                        if hasattr(experience, 'save_png_npy'):
                            print(f"DEBUG: Debug: Calling save_png_npy for env {env_id} with {len(experience.data.get('rgb_images', []))} RGB images")
                            experience.save_png_npy(env_dir)
                            print(f"SUCCESS: Saved RGBD {data_type} data for env {env_id} to {env_dir}")
                            saved_count += 1
                        else:
                            print(f"WARNING:  Experience for env {env_id} doesn't support RGBD saving")
                            print(f"DEBUG: Debug: Experience type: {type(experience)}")
                            print(f"DEBUG: Debug: Experience attributes: {dir(experience)}")
                    except Exception as e:
                        print(f"ERROR: Failed to save RGBD data for env {env_id}: {e}")
                        import traceback
                        print(f"DEBUG: Debug traceback: {traceback.format_exc()}")
                
                print(f"COMPLETE: RGBD Wrapper: Successfully saved {saved_count} environments to {self._rgbd_data_dir}!")
                
                # Also create a summary file in the original location for reference
                try:
                    original_dir = os.path.dirname(save_path)
                    os.makedirs(original_dir, exist_ok=True)
                    summary_file = os.path.join(original_dir, 'rgbd_redirect_info.txt')
                    with open(summary_file, 'w') as f:
                        f.write(f"RGBD data collection completed!\n")
                        f.write(f"Original save path: {save_path}\n")
                        f.write(f"Actual save location: {self._rgbd_data_dir}/{dir_name}/\n")
                        f.write(f"Task: {task_name}\n")
                        f.write(f"Data type: {data_type}\n")
                        f.write(f"Environments saved: {saved_count}\n")
                        f.write(f"Saved directories:\n")
                        for env_id in range(saved_count):
                            f.write(f"  - {self._rgbd_data_dir}/{dir_name}/{data_type}_env_{env_id}/\n")
                    print(f"📝 Created redirect info file: {summary_file}")
                except Exception as e:
                    print(f"WARNING:  Could not create redirect info file: {e}")
        
        # Monkey patch the data structures in the method's scope
        import dataset.dataset as dataset_module
        original_Experience = dataset_module.Experience
        original_Episode_Buffer = dataset_module.Episode_Buffer
        
        # Create a factory function that sets environment reference
        # We need to track which environment ID each buffer is for
        buffer_counter = [0]  # Use list to make it mutable in closure
        
        def create_enhanced_episode_buffer():
            buffer = RGBDEnhancedEpisodeBuffer()
            buffer._env = self.env
            buffer._env_id = buffer_counter[0] % self.env.num_envs  # Cycle through env IDs
            buffer_counter[0] += 1
            return buffer
        
        # Temporarily replace with RGBD versions
        dataset_module.Experience = RGBDEnhancedExperience
        dataset_module.Episode_Buffer = create_enhanced_episode_buffer
        
        # Also replace in the manipulation object's namespace if they're imported there
        if hasattr(self, 'Experience'):
            original_self_Experience = self.Experience
            self.Experience = RGBDEnhancedExperience
        else:
            original_self_Experience = None
            
        if hasattr(self, 'Episode_Buffer'):
            original_self_Episode_Buffer = self.Episode_Buffer  
            self.Episode_Buffer = create_enhanced_episode_buffer
        else:
            original_self_Episode_Buffer = None
        
        # Also replace in the manipulation module's namespace (for direct imports)
        manipulation_module = self.__class__.__module__
        if manipulation_module in sys.modules:
            import sys
            mod = sys.modules[manipulation_module]
            original_mod_Experience = getattr(mod, 'Experience', None)
            original_mod_Episode_Buffer = getattr(mod, 'Episode_Buffer', None)
            
            if original_mod_Experience:
                setattr(mod, 'Experience', RGBDEnhancedExperience)
                print(f"SETUP: Replaced Experience in module {manipulation_module}")
            if original_mod_Episode_Buffer:
                setattr(mod, 'Episode_Buffer', create_enhanced_episode_buffer)
                print(f"SETUP: Replaced Episode_Buffer in module {manipulation_module}")
        else:
            original_mod_Experience = None
            original_mod_Episode_Buffer = None
        
        try:
            # Set task name for proper saving
            if hasattr(self, 'env') and hasattr(self.env, 'task_name'):
                task_name = self.env.task_name
            elif hasattr(self, 'cfg') and 'task' in self.cfg and 'task_name' in self.cfg['task']:
                task_name = self.cfg['task']['task_name']
            else:
                # Try to infer from class name
                class_name = self.__class__.__name__
                if 'CoffeeMachine' in class_name:
                    task_name = 'OpenCoffeeMachine'
                elif 'Door' in class_name:
                    task_name = 'OpenDoor'
                elif 'Bottle' in class_name:
                    task_name = 'OpenBottle'
                elif 'Lamp' in class_name:
                    task_name = 'OpenLamp'
                elif 'MicroWave' in class_name:
                    task_name = 'OpenMicroWave'
                elif 'Pen' in class_name:
                    task_name = 'OpenPen'
                elif 'Safe' in class_name:
                    task_name = 'OpenSafe'
                elif 'Window' in class_name:
                    task_name = 'OpenWindow'
                elif 'PressureCooker' in class_name:
                    task_name = 'OpenPressureCooker'
                else:
                    task_name = 'Unknown'
            
            # Store task name for saving - set it as a class attribute AND instance attribute
            RGBDEnhancedExperience._task_name = task_name
            # Also set it on any existing instances
            for attr_name in dir(self):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, RGBDEnhancedExperience):
                    attr_value._task_name = task_name
            
            # Call the original method - it will now use RGBD data!
            print(f"START: Executing original {original_method.__name__} with RGBD enhancements...")
            print(f"DEBUG: Debug: Experience class is now {dataset_module.Experience}")
            print(f"DEBUG: Debug: Episode_Buffer class is now {dataset_module.Episode_Buffer}")
            result = original_method()
            print(f"SUCCESS: RGBD-enhanced {original_method.__name__} completed successfully")
            
        finally:
            # Restore original methods and classes
            self.env.collect_single_diff_data = original_collect_single
            self.env.collect_diff_data = original_collect_diff
            dataset_module.Experience = original_Experience
            dataset_module.Episode_Buffer = original_Episode_Buffer
            
            if original_self_Experience is not None:
                self.Experience = original_self_Experience
            if original_self_Episode_Buffer is not None:
                self.Episode_Buffer = original_self_Episode_Buffer
            
            # Restore manipulation module namespace
            if manipulation_module in sys.modules:
                mod = sys.modules[manipulation_module]
                if original_mod_Experience:
                    setattr(mod, 'Experience', original_mod_Experience)
                if original_mod_Episode_Buffer:
                    setattr(mod, 'Episode_Buffer', original_mod_Episode_Buffer)
        
        return result
    
    return wrapper


def apply_rgbd_wrapper(manipulation_obj, method_names=['collect_manip_data', 'collect_grasp_data'], rgbd_data_dir="./adamanip_d3fields"):
    """
    Apply RGBD wrapper to specified methods of a manipulation object.
    
    Args:
        manipulation_obj: The manipulation object to enhance
        method_names: List of method names to wrap
        rgbd_data_dir: Base directory for saving RGBD data
    """
    for method_name in method_names:
        if hasattr(manipulation_obj, method_name):
            original_method = getattr(manipulation_obj, method_name)
            
            # Create a closure that captures the rgbd_data_dir
            def create_wrapper_with_config(orig_method, data_dir):
                def wrapper_with_config(self):
                    return rgbd_data_collection_wrapper_with_config(orig_method, data_dir)(self)
                return wrapper_with_config
            
            wrapped_method = create_wrapper_with_config(original_method, rgbd_data_dir)
            
            # Bind the wrapped method to the instance
            import types
            bound_wrapped_method = types.MethodType(wrapped_method, manipulation_obj)
            setattr(manipulation_obj, method_name + '_rgbd', bound_wrapped_method)
            print(f"SUCCESS: Added RGBD wrapper: {method_name}_rgbd")
        else:
            print(f"WARNING:  Method {method_name} not found on {manipulation_obj.__class__.__name__}") 