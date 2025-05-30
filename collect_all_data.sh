#!/bin/bash

# Function to run scripts with specific GPU
run_with_gpu() {
    local gpu_id=$1
    local script=$2
    export CUDA_VISIBLE_DEVICES=$gpu_id
    echo "Running $script on GPU $gpu_id"
    sh "$script"
}

# Run all scripts in parallel on different GPUs
echo "Starting parallel data collection on multiple GPUs..."

# GPU 0 - Safe and Window grasp
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=0; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/safe/collect_safe.sh) &
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=0; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/window/collect_window_grasp.sh) &

# GPU 1 - Bottle scripts
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=1; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/bottle/collect_bottle_grasp.sh) &
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=1; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/bottle/collect_bottle_manip.sh) &

# GPU 2 - Coffee Machine scripts
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=2; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/cm/collect_cm_grasp.sh) &
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=2; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/cm/collect_cm_manip.sh) &

# GPU 3 - Door scripts
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=3; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/door/collect_door_grasp.sh) &
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=3; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/door/collect_door_manip.sh) &

# GPU 4 - Lamp scripts
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=4; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/lamp/collect_lamp_grasp.sh) &
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=4; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/lamp/collect_lamp_manip.sh) &

# GPU 5 - Microwave and Window manip
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=5; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/microwave/collect_mv.sh) &
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=5; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/window/collect_window_manip.sh) &

# GPU 6 - Pressure Cooker scripts
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=6; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/pc/collect_pc_grasp.sh) &
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=6; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/pc/collect_pc_manip.sh) &

# GPU 7 - Pen scripts
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=7; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/pen/collect_pen_grasp.sh) &
(source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh; export CUDA_VISIBLE_DEVICES=7; sh /ssdstore/azadaia/project_snellius_sync/AdaManip/scripts/pen/collect_pen_manip.sh) &

echo "All scripts launched in parallel. Waiting for completion..."

# Wait for all background processes to complete
wait

echo "All data collection scripts completed!"
