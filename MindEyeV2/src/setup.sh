#!/bin/bash
# Commands to setup a new virtual environment and install all the necessary packages

set -e

pip install --upgrade pip

python3.11 -m venv mindeye
source mindeye/bin/activate

pip install numpy matplotlib seaborn jupyter ipykernel jupyterlab_nvdashboard jupyterlab tqdm scikit-image accelerate webdataset pandas matplotlib einops ftfy regex kornia h5py open_clip_torch torchvision torch==2.1.0 transformers xformers torchmetrics diffusers==0.13.0 deepspeed wandb nilearn nibabel dalle2-pytorch

pip install git+https://github.com/openai/CLIP.git --no-deps