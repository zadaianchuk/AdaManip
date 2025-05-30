#!/usr/bin/env python3
"""
Validation script to test that segmentation masks contain meaningful unique values.

This script:
1. Checks all collected segmentation masks 
2. Verifies they contain more than just background (ID=0)
3. Reports statistics for each environment and camera
4. Provides a comprehensive validation report
"""

import numpy as np
from PIL import Image
import os
import json
from collections import defaultdict

def validate_segmentation_masks():
    """Validate all segmentation masks in the collected datasets"""
    
    print("🔍 VALIDATING SEGMENTATION MASKS")
    print("=" * 50)
    
    # base_path = "adamanip_d3fields/OpenBottle"
    # base_path = "adamanip_d3fields/OpenCoffeeMachine"
    base_path = "adamanip_d3fields/OpenPressureCooker"
    # base_path = "adamanip_d3fields/OpenSafe"
    # base_path = "adamanip_d3fields/OpenWindow"
    
    
    
    
    if not os.path.exists(base_path):
        print(f"❌ Dataset path not found: {base_path}")
        return False
    
    validation_results = {
        'total_environments': 0,
        'valid_environments': 0,
        'total_masks': 0,
        'valid_masks': 0,
        'unique_ids_found': set(),
        'detailed_results': {}
    }
    
    # Process each environment
    env_dirs = [d for d in os.listdir(base_path) if d.startswith('grasp_env_') or d.startswith('manip_env_')]
    validation_results['total_environments'] = len(env_dirs)
    
    print(f"📂 Found {len(env_dirs)} environments to validate")
    
    for env_dir in sorted(env_dirs):
        env_path = os.path.join(base_path, env_dir)
        env_valid = True
        env_results = {
            'cameras': {},
            'total_masks': 0,
            'valid_masks': 0,
            'unique_ids': set()
        }
        
        print(f"\n📁 Validating {env_dir}:")
        
        # Process each camera in the environment
        camera_dirs = [d for d in os.listdir(env_path) if d.startswith('camera_') and os.path.isdir(os.path.join(env_path, d))]
        
        for camera_dir in sorted(camera_dirs):
            camera_path = os.path.join(env_path, camera_dir)
            masks_path = os.path.join(camera_path, 'masks')
            
            if not os.path.exists(masks_path):
                print(f"   ❌ {camera_dir}: No masks directory found")
                env_valid = False
                continue
            
            # Get all mask files
            mask_files = [f for f in os.listdir(masks_path) if f.endswith('.png')]
            mask_files.sort()
            
            camera_results = {
                'total_masks': len(mask_files),
                'valid_masks': 0,
                'unique_ids': set(),
                'mask_details': []
            }
            
            print(f"   📷 {camera_dir}: Found {len(mask_files)} mask files")
            
            # Validate each mask
            for mask_file in mask_files:
                mask_path = os.path.join(masks_path, mask_file)
                
                try:
                    # Load mask
                    mask = np.array(Image.open(mask_path))
                    unique_ids = np.unique(mask)
                    
                    # Check if mask has meaningful segmentation
                    is_valid = len(unique_ids) > 1  # More than just background
                    non_zero_count = np.count_nonzero(mask)
                    total_pixels = mask.size
                    non_zero_percentage = 100 * non_zero_count / total_pixels
                    
                    mask_detail = {
                        'file': mask_file,
                        'shape': list(mask.shape),
                        'unique_ids': [int(x) for x in unique_ids],
                        'is_valid': bool(is_valid),
                        'non_zero_pixels': int(non_zero_count),
                        'non_zero_percentage': float(non_zero_percentage)
                    }
                    
                    camera_results['mask_details'].append(mask_detail)
                    camera_results['unique_ids'].update(unique_ids)
                    
                    if is_valid:
                        camera_results['valid_masks'] += 1
                        validation_results['valid_masks'] += 1
                    
                    validation_results['total_masks'] += 1
                    validation_results['unique_ids_found'].update(unique_ids)
                    
                    # Print mask validation result
                    if is_valid:
                        print(f"      ✅ {mask_file}: IDs {unique_ids} ({non_zero_percentage:.1f}% objects)")
                    else:
                        print(f"      ❌ {mask_file}: Only background (ID=0)")
                        env_valid = False
                
                except Exception as e:
                    print(f"      ❌ {mask_file}: Error loading mask - {e}")
                    env_valid = False
            
            env_results['cameras'][camera_dir] = camera_results
            env_results['total_masks'] += camera_results['total_masks']
            env_results['valid_masks'] += camera_results['valid_masks']
            env_results['unique_ids'].update(camera_results['unique_ids'])
        
        validation_results['detailed_results'][env_dir] = env_results
        
        if env_valid and env_results['valid_masks'] > 0:
            validation_results['valid_environments'] += 1
            print(f"   ✅ {env_dir}: VALID ({env_results['valid_masks']}/{env_results['total_masks']} masks with objects)")
        else:
            print(f"   ❌ {env_dir}: INVALID (no meaningful segmentation)")
    
    # Print summary report
    print(f"\n📊 VALIDATION SUMMARY")
    print(f"=" * 30)
    print(f"Total Environments: {validation_results['total_environments']}")
    print(f"Valid Environments: {validation_results['valid_environments']}")
    print(f"Total Masks: {validation_results['total_masks']}")
    print(f"Valid Masks: {validation_results['valid_masks']}")
    print(f"Success Rate: {100 * validation_results['valid_masks'] / max(1, validation_results['total_masks']):.1f}%")
    
    # Analyze unique segmentation IDs found
    all_unique_ids = sorted(list(validation_results['unique_ids_found']))
    print(f"\n🎯 SEGMENTATION IDs FOUND: {all_unique_ids}")
    
    id_meanings = {
        0: "Background",
        1: "Franka Robot", 
        2: "Bottle Object (Type 0)",
        3: "Bottle Object (Type 1)",
        4: "Bottle Object (Type 2)",
        # Add more as needed
    }
    
    for seg_id in all_unique_ids:
        meaning = id_meanings.get(seg_id, f"Object Type {seg_id-2}" if seg_id >= 2 else "Unknown")
        print(f"   ID {seg_id}: {meaning}")
    
    found = all_unique_ids

    # Determine overall validation result
    overall_valid = (
        validation_results['valid_environments'] > 0 and
        validation_results['valid_masks'] > 0 and
        len(found) >= 2  # At least background + one object type
    )
    
    if overall_valid:
        if validation_results['valid_masks'] == validation_results['total_masks']:
            print(f"\n🎉 VALIDATION PASSED!")
        else:
            print(f"\n❌ VALIDATION ONLY PARTIALLY PASSED!")
        print(f"Segmentation masks contain meaningful object IDs")
        print(f"{validation_results['valid_masks']}/{validation_results['total_masks']} masks successfully contain non-background objects")
    else:
        print(f"\n❌ VALIDATION FAILED!")
        if validation_results['valid_masks'] == 0:
            print(f"❌ No masks contain non-background objects")
        if len(found) < 2:
            print(f"❌ Insufficient object types detected")
    

    
    return overall_valid

def quick_mask_test():
    """Quick test of a few masks to demonstrate working segmentation"""
    
    print(f"\n🔬 QUICK MASK TEST")
    print(f"=" * 20)
    
    # Test a few specific masks that should contain objects
    test_masks = [
        "adamanip_d3fields/OpenBottle/grasp_env_0/camera_0/masks/4.png",
        "adamanip_d3fields/OpenBottle/grasp_env_1/camera_0/masks/5.png",
        "adamanip_d3fields/OpenBottle/grasp_env_2/camera_1/masks/3.png"
    ]
    
    for mask_path in test_masks:
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            unique_ids = np.unique(mask)
            non_zero_pixels = np.count_nonzero(mask)
            total_pixels = mask.size
            
            print(f"📄 {os.path.basename(mask_path)}:")
            print(f"   Shape: {mask.shape}")
            print(f"   Unique IDs: {unique_ids}")
            print(f"   Non-zero pixels: {non_zero_pixels:,}/{total_pixels:,} ({100*non_zero_pixels/total_pixels:.1f}%)")
            
            if len(unique_ids) > 1:
                print(f"   ✅ VALID - Contains objects beyond background")
            else:
                print(f"   ❌ INVALID - Only background detected")
        else:
            print(f"📄 {mask_path}: File not found")

def main():
    """Main validation function"""
    
    print("🧪 SEGMENTATION MASK VALIDATION")
    print("🎯 Testing that masks contain meaningful unique values (not just zeros)")
    print("=" * 60)
    
    # Run quick test first
    quick_mask_test()
    
    # Run comprehensive validation
    validate_segmentation_masks()   

if __name__ == "__main__":
    main() 