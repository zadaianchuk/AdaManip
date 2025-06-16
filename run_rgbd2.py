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
from utils.rgbd_utils import add_rgbd_collection_to_env
from utils.rgbd_wrapper import apply_rgbd_wrapper

# Now safe to import torch
import torch

"""
run_rgbd2.py
------------
Enhanced RGB-D data collection launcher using wrapper approach.

This script wraps the original data collection methods (collect_manip_data, 
collect_grasp_data) with RGBD capabilities while preserving all the original
task-specific logic. The wrapper automatically:

1. Intercepts data collection calls to use RGBD versions
2. Replaces data structures with RGBD-compatible ones  
3. Saves data to proper adamanip_d3fields directory structure
4. Handles per-environment saving

No code duplication - reuses all original manipulation logic!
"""

def run_rgbd2():
    logger = Logger(name=args.task)

    # Create environment and add RGBD collection capability
    env = parse_env(args, cfg, sim_params, logdir)
    add_rgbd_collection_to_env(env.__class__)
    
    # Test RGBD collection capability
    print("Testing RGBD collection capability...")
    try:
        env.reset()
        test_obs = env.collect_rgbd_data()
        if isinstance(test_obs, dict) and 'rgb_images' in test_obs and 'depth_images' in test_obs:
            rgb_shape = test_obs['rgb_images'].shape
            depth_shape = test_obs['depth_images'].shape
            print(f"RGBD collection working - RGB: {rgb_shape}, Depth: {depth_shape}")
            
            if 'segmentation_masks' in test_obs:
                seg_shape = test_obs['segmentation_masks'].shape
                print(f"Segmentation masks available - Shape: {seg_shape}")
            else:
                print("No segmentation masks found")
        else:
            print("WARNING: RGBD collection returned data but missing expected keys")
    except Exception as e:
        print(f"ERROR: RGBD collection test failed: {e}")
        return

    # Create manipulation object
    manipulation = parse_manipulation(args, env, cfg, logger)
    
    # Remove any existing broken native RGBD methods first
    if hasattr(manipulation, 'collect_manip_data_rgbd'):
        print("Removing existing broken collect_manip_data_rgbd method")
        delattr(manipulation, 'collect_manip_data_rgbd')
    
    if hasattr(manipulation, 'collect_grasp_data_rgbd'):
        print("Removing existing broken collect_grasp_data_rgbd method") 
        delattr(manipulation, 'collect_grasp_data_rgbd')
    
    # Apply RGBD wrapper to enhance original methods
    print(f"Applying RGBD wrapper to {manipulation.__class__.__name__}")
    print(f"Using RGBD data directory: {args.rgbd_data_dir}")
    apply_rgbd_wrapper(manipulation, ['collect_manip_data', 'collect_grasp_data'], args.rgbd_data_dir)
    
    # Verify wrapper was applied
    print(f"Verification: collect_manip_data_rgbd exists: {hasattr(manipulation, 'collect_manip_data_rgbd')}")
    print(f"Verification: collect_grasp_data_rgbd exists: {hasattr(manipulation, 'collect_grasp_data_rgbd')}")
    if hasattr(manipulation, 'collect_manip_data_rgbd'):
        print(f"Verification: collect_manip_data_rgbd type: {type(manipulation.collect_manip_data_rgbd)}")

    # Create controller
    controller = parse_controller(args, env, manipulation, cfg, logger)

    # Override controller run method to use RGBD versions
    def run_rgbd_controller(eval=False):
        if getattr(controller, 'collect_grasp', False):
            if hasattr(manipulation, 'collect_grasp_data_rgbd'):
                print("Using RGBD-wrapped grasp data collection")
                manipulation.collect_grasp_data_rgbd()
            else:
                print("WARNING: No RGBD grasp method available, using original")
                manipulation.collect_grasp_data()
        else:
            if hasattr(manipulation, 'collect_manip_data_rgbd'):
                print("Using RGBD-wrapped manipulation data collection")
                manipulation.collect_manip_data_rgbd()
            else:
                print("WARNING: No RGBD manipulation method available, using original")
                manipulation.collect_manip_data()
    
    controller.run = run_rgbd_controller
    
    print("Starting RGBD data collection...")
    controller.run()
    print("RGBD data collection completed!")

if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    print(f"Task: {args.task}")
    
    # Auto-set manipulation class based on task name
    task_to_manipulation = {
        'OpenBottle': 'OpenBottleManipulation',
        'OpenCoffeeMachine': 'OpenCoffeeMachineManipulation', 
        'OpenDoor': 'OpenDoorManipulation',
        'OpenLamp': 'OpenLampManipulation',
        'OpenMicroWave': 'OpenMicroWaveManipulation',
        'OpenPen': 'OpenPenManipulation',
        'OpenPressureCooker': 'OpenPressureCookerManipulation',
        'OpenSafe': 'OpenSafeManipulation',
        'OpenWindow': 'OpenWindowManipulation'
    }
    if args.cfg_env == "Base":
        task_to_manipulation_config = {
            'OpenBottle': 'cfg/bottle/collect_bottle_manip.yaml',
            'OpenCoffeeMachine': 'cfg/cm/collect_cm_manip.yaml',
            'OpenDoor': 'cfg/door/collect_door_manip.yaml',
            'OpenLamp': 'cfg/lamp/collect_lamp_manip.yaml',
            'OpenMicroWave': 'cfg/microwave/collect_microwave.yaml',
            'OpenPen': 'cfg/pen/collect_pen_manip.yaml',
            'OpenPressureCooker': 'cfg/pressure_cooker/collect_pc_manip.yaml',
            'OpenSafe': 'cfg/safe/collect_safe.yaml',
            'OpenWindow': 'cfg/window/collect_window_manip.yaml'
        }
        args.cfg_env = task_to_manipulation_config[args.task]
    if args.task in task_to_manipulation:
        args.manipulation = task_to_manipulation[args.task]
        print(f"Auto-set manipulation: {args.manipulation}")
    else:
        print(f"WARNING: Unknown task {args.task}, using default manipulation: {args.manipulation}")
    
    cfg, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg)
    set_seed(args.seed)
    run_rgbd2() 