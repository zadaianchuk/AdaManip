#!/usr/bin/env python3
"""
Comprehensive RGBD Data Collection Script for All Objects in AdaManip

This script runs RGBD data collection for all object types:
- bottle, coffee_machine, door, lamp, microwave, pen, pressure_cooker, safe, window

For each object, it collects both grasp and manipulation data using the RGBD implementation.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import argparse

# Object configurations mapping
OBJECT_CONFIGS = {
    'bottle': {
        'task': 'OpenBottle',
        'manipulation': 'OpenBottleManipulation',
        'has_grasp': True,
        'has_manip': True,
    },
    'cm': {  # coffee machine
        'task': 'OpenCoffeeMachine', 
        'manipulation': 'OpenCoffeeMachineManipulation',
        'has_grasp': True,
        'has_manip': True,
    },
    'door': {
        'task': 'OpenDoor',
        'manipulation': 'OpenDoorManipulation', 
        'has_grasp': True,
        'has_manip': True,
    },
    'lamp': {
        'task': 'OpenLamp',
        'manipulation': 'OpenLampManipulation',
        'has_grasp': True,
        'has_manip': True,
    },
    'microwave': {
        'task': 'OpenMicroWave',
        'manipulation': 'OpenMicroWaveManipulation',
        'has_grasp': False,  # Microwave typically doesn't have grasp collection
        'has_manip': True,
    },
    'pen': {
        'task': 'OpenPen',
        'manipulation': 'OpenPenManipulation',
        'has_grasp': True,
        'has_manip': True,
    },
    'pressure_cooker': {
        'task': 'OpenPressureCooker', 
        'manipulation': 'OpenPressureCookerManipulation',
        'has_grasp': True,
        'has_manip': True,
    },
    'safe': {
        'task': 'OpenSafe',
        'manipulation': 'OpenSafeManipulation',
        'has_grasp': False,  # Safe typically doesn't have grasp collection
        'has_manip': True,
    },
    'window': {
        'task': 'OpenWindow',
        'manipulation': 'OpenWindowManipulation',
        'has_grasp': True,
        'has_manip': True,
    }
}

def get_config_name(obj_name, collection_type):
    """Get the configuration file name for an object and collection type"""
    if obj_name == 'pressure_cooker':
        # Special case for pressure cooker
        if collection_type == 'grasp':
            return f"cfg/pc/collect_pc_grasp.yaml"
        else:
            return f"cfg/pc/collect_pc_manip.yaml"
    elif obj_name == 'microwave':
        # Microwave only has one collection config
        return f"cfg/microwave/collect_mv.yaml"
    elif obj_name == 'safe':
        # Safe only has one collection config  
        return f"cfg/safe/collect_safe.yaml"
    else:
        # Standard naming pattern
        return f"cfg/{obj_name}/collect_{obj_name}_{collection_type}.yaml"

def run_command(cmd, obj_name, collection_type, timeout=3600):
    """Run a command with timeout and logging"""
    print(f"\n{'='*60}")
    print(f"🚀 Starting {collection_type} collection for {obj_name}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the command
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor the process
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                output_lines.append(output.strip())
                
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"⏰ Timeout reached ({timeout}s), terminating process...")
                process.terminate()
                process.wait()
                return False, f"Timeout after {timeout}s"
        
        return_code = process.poll()
        elapsed_time = time.time() - start_time
        
        if return_code == 0:
            print(f"✅ {obj_name} {collection_type} completed successfully in {elapsed_time:.1f}s")
            return True, f"Success in {elapsed_time:.1f}s"
        else:
            print(f"❌ {obj_name} {collection_type} failed with return code {return_code}")
            return False, f"Failed with code {return_code}"
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Exception during {obj_name} {collection_type}: {str(e)}")
        return False, f"Exception: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Collect RGBD data for all objects')
    parser.add_argument('--objects', nargs='+', 
                       choices=list(OBJECT_CONFIGS.keys()) + ['all'],
                       default=['all'],
                       help='Objects to collect data for (default: all)')
    parser.add_argument('--types', nargs='+', 
                       choices=['grasp', 'manip', 'both'],
                       default=['both'],
                       help='Types of data to collect (default: both)')
    parser.add_argument('--device', default='cuda:0',
                       help='Device to use (default: cuda:0)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout per collection in seconds (default: 3600)')
    parser.add_argument('--continue_on_error', action='store_true',
                       help='Continue collecting other objects even if one fails')
    
    args = parser.parse_args()
    
    # Determine which objects to process
    if 'all' in args.objects:
        objects_to_process = list(OBJECT_CONFIGS.keys())
    else:
        objects_to_process = args.objects
    
    # Determine which collection types to run
    if 'both' in args.types:
        collection_types = ['grasp', 'manip']
    else:
        collection_types = args.types
    
    print(f"\n🎯 RGBD Data Collection Plan")
    print(f"Objects: {objects_to_process}")
    print(f"Collection types: {collection_types}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Timeout per collection: {args.timeout}s")
    print(f"Continue on error: {args.continue_on_error}")
    
    # Results tracking
    results = {}
    total_collections = 0
    successful_collections = 0
    start_time = datetime.now()
    
    # Run collections
    for obj_name in objects_to_process:
        config = OBJECT_CONFIGS[obj_name]
        obj_results = {}
        
        for collection_type in collection_types:
            # Check if this object supports this collection type
            if collection_type == 'grasp' and not config['has_grasp']:
                print(f"⏭️ Skipping grasp collection for {obj_name} (not supported)")
                obj_results[collection_type] = ('skipped', 'Not supported')
                continue
            elif collection_type == 'manip' and not config['has_manip']:
                print(f"⏭️ Skipping manip collection for {obj_name} (not supported)")
                obj_results[collection_type] = ('skipped', 'Not supported')
                continue
            
            total_collections += 1
            
            # Get configuration file
            config_file = get_config_name(obj_name, collection_type)
            
            # Check if config file exists
            if not os.path.exists(config_file):
                print(f"⚠️ Configuration file not found: {config_file}")
                obj_results[collection_type] = ('failed', 'Config file not found')
                if not args.continue_on_error:
                    print(f"❌ Stopping due to missing config file")
                    break
                continue
            
            # Build command - use run_rgbd.py instead of run.py for RGBD collection
            cmd = (
                f"python run_rgbd.py "
                f"--headless "
                f"--task={config['task']} "
                f"--controller=GtController "
                f"--manipulation={config['manipulation']} "
                f"--sim_device={args.device} "
                f"--seed={args.seed} "
                f"--pipeline=gpu "
                f"--cfg_env={config_file}"
            )
            
            # Run the command
            success, message = run_command(cmd, obj_name, collection_type, args.timeout)
            
            if success:
                successful_collections += 1
                obj_results[collection_type] = ('success', message)
            else:
                obj_results[collection_type] = ('failed', message)
                if not args.continue_on_error:
                    print(f"❌ Stopping due to failure in {obj_name} {collection_type}")
                    results[obj_name] = obj_results
                    break
        
        results[obj_name] = obj_results
        
        # Stop if we hit an error and continue_on_error is False
        if not args.continue_on_error and any(status == 'failed' for status, _ in obj_results.values()):
            break
    
    # Print final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n" + "="*80)
    print(f"📊 RGBD DATA COLLECTION SUMMARY")
    print(f"="*80)
    print(f"⏱️ Total time: {duration}")
    print(f"📈 Success rate: {successful_collections}/{total_collections} ({100*successful_collections/max(total_collections,1):.1f}%)")
    print(f"")
    
    for obj_name, obj_results in results.items():
        print(f"🔸 {obj_name.upper()}:")
        for collection_type, (status, message) in obj_results.items():
            status_emoji = {'success': '✅', 'failed': '❌', 'skipped': '⏭️'}[status]
            print(f"   {status_emoji} {collection_type}: {message}")
    
    print(f"\n💾 Collected datasets should be available in:")
    print(f"   ./demo_data/rgbd_*")
    
    if successful_collections == total_collections:
        print(f"🎉 All collections completed successfully!")
        return 0
    else:
        print(f"⚠️ Some collections failed or were skipped.")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 