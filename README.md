# Installation
conda create -n adamanip python=3.8

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

Install IsaacGym

Install pointnet++

pip install numpy==1.23.5

Install pytorch3d

git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .

pip install plotly

pip install zarr, ipdb, open3d

pip install diffusers==0.11.1

pip install huggingface-hub==0.24.0

pip install einops

ToDo:
1. cuda device debug
2. argument add cuda choice
