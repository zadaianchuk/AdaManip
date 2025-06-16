LIST_OF_TASKS=(OpenBottle OpenCoffeeMachine OpenDoor OpenLamp OpenMicroWave OpenPen OpenPressureCooker OpenSafe OpenWindow)
LIST_OF_CUDA_DEVICES=(0 1 2 3 4 5 6 7 0)

for i in "${!LIST_OF_TASKS[@]}"; do
    task=${LIST_OF_TASKS[$i]}
    device_id=${LIST_OF_CUDA_DEVICES[$i]}
    source /ssdstore/azadaia/project_snellius_sync/AdaManip/setup_cuda_env.sh
    python run_rgbd2.py --headless --task=$task --controller=GtController --manipulation=OpenCoffeeMachineManipulation --sim_device=cuda:$device_id --seed=0 --pipeline=gpu --rgbd_data_dir /ssdstore/azadaia/project_snellius_sync/AdaManip/adamanip_d3fields_manipulation &
done
