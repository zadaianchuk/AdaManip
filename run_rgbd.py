import os
import sys
from ast import arg
import numpy as np
import random
from logging import Logger

# Add paths BEFORE importing Isaac Gym dependent modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "envs"))
sys.path.append(os.path.join(BASE_DIR, "controller"))

# Import Isaac Gym dependent modules first (before torch)
from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_env
from utils.parse import *
from dataset.dataset_rgbd import RGBDExperience, RGBDEpisodeBuffer
from utils.rgbd_utils import add_rgbd_collection_to_env, debug_rgbd_collection

# Now safe to import torch
import torch

def run_rgbd():
    
    logger = Logger(name=args.task)

    env = parse_env(args, cfg, sim_params, logdir)

    # Add RGBD data collection capability if not already present
    if not hasattr(env, 'collect_rgbd_data'):
        print("Adding RGBD collection capability to environment...")
        add_rgbd_collection_to_env(env.__class__)
    else:
        print("Environment already has RGBD collection capability.")
    
    # Debug RGBD collection capability (but don't pass debug parameter)
    print("Debugging RGBD collection...")
    try:
        env.reset()
        test_obs = env.collect_rgbd_data()
        if 'rgb_images' in test_obs and 'depth_images' in test_obs:
            rgb_shape = test_obs['rgb_images'].shape
            depth_shape = test_obs['depth_images'].shape
            print(f"✅ RGBD collection working - RGB: {rgb_shape}, Depth: {depth_shape}")
        else:
            print("⚠️ RGBD collection returned data but missing image keys")
    except Exception as e:
        print(f"❌ RGBD collection test failed: {e}")
        # Fall back to adding capability
        add_rgbd_collection_to_env(env.__class__)
    
    manipulation = parse_manipulation(args, env, cfg, logger)

    # Modify manipulation to use RGBD data collection
    if hasattr(manipulation, 'collect_grasp_data'):
        manipulation.collect_grasp_data_rgbd = lambda: collect_grasp_data_rgbd(manipulation, env, cfg)
    if hasattr(manipulation, 'collect_manip_data'):
        manipulation.collect_manip_data_rgbd = lambda: collect_manip_data_rgbd(manipulation, env, cfg)

    controller = parse_controller(args, env, manipulation, cfg, logger)

    # Override controller run method to use RGBD collection
    original_run = controller.run
    def run_rgbd_controller(eval=False):
        if hasattr(controller, 'collect_grasp') and controller.collect_grasp:
            if hasattr(manipulation, 'collect_grasp_data_rgbd'):
                print("🎯 Using RGBD grasp data collection")
                manipulation.collect_grasp_data_rgbd()
            else:
                print("⚠️ RGBD grasp data collection not available, falling back to original")
                manipulation.collect_grasp_data()
        else:
            if hasattr(manipulation, 'collect_manip_data_rgbd'):
                print("🎯 Using RGBD manipulation data collection")
                manipulation.collect_manip_data_rgbd()
            else:
                print("⚠️ RGBD manip data collection not available, falling back to original")
                manipulation.collect_manip_data()
    
    controller.run = run_rgbd_controller
    controller.run()

def collect_grasp_data_rgbd(manipulation, env, cfg):
    """RGBD version of grasp data collection"""
    print("Collecting grasp data with RGBD observations...")
    eps_num = cfg["task"]["num_episode"]
    demo_buffer = RGBDExperience()
    
    for eps in range(eps_num):
        eps_buffer = [RGBDEpisodeBuffer() for _ in range(env.num_envs)]
        print(f"RGBD eps_{eps+1}/{eps_num}")
        env.reset()
        
        # Get initial hand pose and adjust
        if hasattr(env, 'adjust_hand_pose'):
            pre_pose = env.adjust_hand_pose.clone()
            pre_pose[:, 2] += env.gripper_length * 2
        else:
            # Fallback for environments without adjust_hand_pose
            pre_pose = env.hand_rigid_body_tensor[:, :7].clone()
            pre_pose[:, 2] += 0.1
        
        # First phase - approach
        for i in range(3):
            # Collect RGBD observations
            try:
                rgbd_obs = env.collect_rgbd_data()
            except Exception as e:
                print(f"⚠️ RGBD collection failed, falling back to regular collection: {e}")
                rgbd_obs = env.collect_diff_data() if hasattr(env, 'collect_diff_data') else {}
            
            # Step environment
            for j in range(10):
                env.step(pre_pose)
            
            # Process action
            gt_action = manipulation.action_process(pre_pose) if hasattr(manipulation, 'action_process') else pre_pose
            env.actions = gt_action.clone()
            
            # Store RGBD data
            for env_id in range(env.num_envs):
                try:
                    if isinstance(rgbd_obs, dict) and 'rgb_images' in rgbd_obs:
                        eps_buffer[env_id].add_rgbd(
                            rgbd_obs['pc'][env_id], 
                            rgbd_obs['proprioception'][env_id], 
                            gt_action[env_id],
                            rgbd_obs['rgb_images'][env_id],
                            rgbd_obs['depth_images'][env_id]
                        )
                    else:
                        # Fallback to regular data collection
                        from dataset.dataset import obs_wrapper
                        pc, env_state = obs_wrapper(rgbd_obs)
                        eps_buffer[env_id].add(pc[env_id], env_state[env_id], gt_action[env_id])
                except Exception as e:
                    print(f"⚠️ Error storing data for env {env_id}: {e}")
        
        # Second phase - grasp
        if hasattr(env, 'gripper_length'):
            pre_pose[:, 2] -= env.gripper_length + 0.01
        else:
            pre_pose[:, 2] -= 0.05  # Default offset
            
        for i in range(3):
            try:
                rgbd_obs = env.collect_rgbd_data()
            except Exception as e:
                print(f"⚠️ RGBD collection failed, falling back to regular collection: {e}")
                rgbd_obs = env.collect_diff_data() if hasattr(env, 'collect_diff_data') else {}
            
            for j in range(10):
                env.step(pre_pose)
            
            gt_action = manipulation.action_process(pre_pose) if hasattr(manipulation, 'action_process') else pre_pose
            env.actions = gt_action.clone()
            
            for env_id in range(env.num_envs):
                try:
                    if isinstance(rgbd_obs, dict) and 'rgb_images' in rgbd_obs:
                        eps_buffer[env_id].add_rgbd(
                            rgbd_obs['pc'][env_id], 
                            rgbd_obs['proprioception'][env_id], 
                            gt_action[env_id],
                            rgbd_obs['rgb_images'][env_id],
                            rgbd_obs['depth_images'][env_id]
                        )
                    else:
                        from dataset.dataset import obs_wrapper
                        pc, env_state = obs_wrapper(rgbd_obs)
                        eps_buffer[env_id].add(pc[env_id], env_state[env_id], gt_action[env_id])
                except Exception as e:
                    print(f"⚠️ Error storing data for env {env_id}: {e}")
        
        # Add episodes to demo buffer
        for env_id in range(env.num_envs):
            demo_buffer.append(eps_buffer[env_id])
        print(f"✅ RGBD Episode {eps+1} completed")
    
    # Save RGBD dataset
    if cfg.get('env', {}).get('collectData', False):
        task_name = args.task if hasattr(args, 'task') else 'unknown'
        asset_num = cfg.get('env', {}).get('asset', {}).get('AssetNum', 1)
        clockwise = cfg.get('env', {}).get('clockwise', 0.5)
        dataset_path = f"rgbd_grasp_{task_name}_{asset_num}_eps{eps_num}_clock{clockwise}"
        save_dir = f'./demo_data/{dataset_path}'
        save_path = f'{save_dir}/rgbd_demo_data.zarr'
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            demo_buffer.save(save_path)
            print(f"✅ RGBD grasp dataset saved to {save_path}")
        except Exception as e:
            print(f"❌ Failed to save RGBD dataset: {e}")
    else:
        print("ℹ️ Data collection is disabled in config, dataset not saved")

def collect_manip_data_rgbd(manipulation, env, cfg):
    """RGBD version of manipulation data collection"""
    print("Collecting manipulation data with RGBD observations...")
    eps_num = cfg["task"]["num_episode"]
    policy = cfg["task"].get("policy", "succ")
    max_step = cfg["task"].get("max_step", 25)
    
    print(f"📊 Episodes: {eps_num}, Policy: {policy}, Max steps: {max_step}")
    
    demo_buffer = RGBDExperience()
    
    for eps in range(eps_num):
        eps_buffer = [RGBDEpisodeBuffer() for _ in range(env.num_envs)]
        done_flag = [False] * env.num_envs
        print(f"RGBD manip eps_{eps+1}/{eps_num}")
        env.reset()
        
        # Get initial pose and setup
        if hasattr(env, 'adjust_hand_pose'):
            hand_pose = env.adjust_hand_pose.clone()
        else:
            hand_pose = env.hand_rigid_body_tensor[:, :7].clone()
        
        # Initialize actions
        init_actions = manipulation.action_process(hand_pose) if hasattr(manipulation, 'action_process') else hand_pose
        env.actions = init_actions
        
        # Main manipulation loop
        for step in range(max_step):
            # Collect RGBD observations
            try:
                rgbd_obs = env.collect_rgbd_data()
            except Exception as e:
                print(f"⚠️ RGBD collection failed at step {step}, falling back: {e}")
                rgbd_obs = env.collect_diff_data() if hasattr(env, 'collect_diff_data') else {}
            
            # Generate actions based on policy
            actions = []
            for env_id in range(env.num_envs):
                if done_flag[env_id]:
                    actions.append(hand_pose[env_id])
                    continue
                
                # Get action based on policy
                if hasattr(manipulation, 'succ_policy'):
                    try:
                        action_type = manipulation.succ_policy(env_id)
                        action = generate_action_from_policy(env, manipulation, env_id, action_type, hand_pose[env_id])
                        actions.append(action)
                    except Exception as e:
                        print(f"⚠️ Policy error for env {env_id}: {e}")
                        actions.append(hand_pose[env_id])
                elif hasattr(manipulation, 'ada_policy'):
                    try:
                        # Get current DOF state for adaptive policy
                        if hasattr(env, 'one_dof_tensor'):
                            dof_val = env.one_dof_tensor[env_id, 0]
                        else:
                            dof_val = 0.0
                        action_type = manipulation.ada_policy(env_id, step, dof_val)
                        action = generate_action_from_policy(env, manipulation, env_id, action_type, hand_pose[env_id])
                        actions.append(action)
                    except Exception as e:
                        print(f"⚠️ Adaptive policy error for env {env_id}: {e}")
                        actions.append(hand_pose[env_id])
                else:
                    # Default: stay in place
                    actions.append(hand_pose[env_id])
            
            # Convert to tensor
            action_tensor = torch.stack(actions)
            gt_action = manipulation.action_process(action_tensor) if hasattr(manipulation, 'action_process') else action_tensor
            
            # Step environment
            for _ in range(15):  # Standard stepping
                env.step(action_tensor)
            env.actions = gt_action
            
            # Store RGBD data
            for env_id in range(env.num_envs):
                if not done_flag[env_id]:
                    try:
                        if isinstance(rgbd_obs, dict) and 'rgb_images' in rgbd_obs:
                            eps_buffer[env_id].add_rgbd(
                                rgbd_obs['pc'][env_id], 
                                rgbd_obs['proprioception'][env_id], 
                                gt_action[env_id],
                                rgbd_obs['rgb_images'][env_id],
                                rgbd_obs['depth_images'][env_id]
                            )
                        else:
                            from dataset.dataset import obs_wrapper
                            pc, env_state = obs_wrapper(rgbd_obs)
                            eps_buffer[env_id].add(pc[env_id], env_state[env_id], gt_action[env_id])
                    except Exception as e:
                        print(f"⚠️ Error storing manipulation data for env {env_id}: {e}")
            
            # Check success conditions
            for env_id in range(env.num_envs):
                if not done_flag[env_id] and check_success_condition(env, env_id):
                    done_flag[env_id] = True
                    print(f"✅ Env {env_id} succeeded at step {step}")
        
        # Add episodes to demo buffer
        for env_id in range(env.num_envs):
            demo_buffer.append(eps_buffer[env_id])
        
        success_count = sum(done_flag)
        print(f"✅ RGBD Manipulation Episode {eps+1} completed - {success_count}/{env.num_envs} succeeded")
    
    # Save RGBD dataset
    if cfg.get('env', {}).get('collectData', False):
        task_name = args.task if hasattr(args, 'task') else 'unknown'
        asset_num = cfg.get('env', {}).get('asset', {}).get('AssetNum', 1)
        clockwise = cfg.get('env', {}).get('clockwise', 0.5)
        dataset_path = f"rgbd_manip_{task_name}_{policy}_{asset_num}_eps{eps_num}_clock{clockwise}"
        save_dir = f'./demo_data/{dataset_path}'
        save_path = f'{save_dir}/rgbd_demo_data.zarr'
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            demo_buffer.save(save_path)
            print(f"✅ RGBD manipulation dataset saved to {save_path}")
        except Exception as e:
            print(f"❌ Failed to save RGBD dataset: {e}")
    else:
        print("ℹ️ Data collection is disabled in config, dataset not saved")

def generate_action_from_policy(env, manipulation, env_id, action_type, current_pose):
    """Generate action based on policy type"""
    try:
        if action_type == 'z':
            # Move down
            new_pose = current_pose.clone()
            if hasattr(env, 'open_size'):
                new_pose[2] -= env.open_size
            else:
                new_pose[2] -= 0.015  # Default open size
            return new_pose
        elif action_type == 'r':
            # Rotate/move based on handle direction
            new_pose = current_pose.clone()
            if hasattr(env, 'part_rigid_body_tensor'):
                handle_quat = env.part_rigid_body_tensor[env_id, 3:7]
                # Get rotation direction (implementation specific)
                if hasattr(manipulation, 'quat_axis'):
                    rotate_dir = manipulation.quat_axis(handle_quat, axis=0)
                    step_size = getattr(env, 'step_size', 0.035)
                    new_pose[:3] -= rotate_dir * step_size
            return new_pose
        else:
            # Default: no change
            return current_pose
    except Exception as e:
        print(f"⚠️ Error generating action: {e}")
        return current_pose

def check_success_condition(env, env_id):
    """Check if the manipulation task succeeded for a given environment"""
    try:
        # Common success conditions based on DOF movement
        if hasattr(env, 'one_dof_tensor'):
            dof_val = env.one_dof_tensor[env_id, 0]
            # Generic success threshold - can be environment specific
            success_threshold = getattr(env, 'success_threshold', 0.025)
            return abs(dof_val) > success_threshold
        
        # Fallback: assume not succeeded
        return False
    except Exception as e:
        return False

if __name__ == '__main__':

    set_np_formatting()

    args = get_args()

    cfg, logdir = load_cfg(args)

    sim_params = parse_sim_params(args, cfg)

    set_seed(args.seed)

    run_rgbd() 