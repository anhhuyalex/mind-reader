#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --mem-per-cpu=20G
#SBATCH --time=12:00:00
#SBATCH --output _mindreader-%J.log
#SBATCH --partition=della-gpu 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1




# SBATCH --mail-type=ALL
# SBATCH --mail-user=alexn@minerva.kgi.edu
 

# SBATCH --gpus=2
# SBATCH --ntasks=1
# source activate rtsynth
# SBATCH --constraint=gpu80

# sbatch --array=0-37 hipp.sh 2
# python -u download.py
# export MODEL_SAVE_FOLDER='/home/an633/project/CuriousContrast/results_alex'
# srun --pty -p della-gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash
# srun --pty -p psych_gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash
source activate neu502b  
python main.py --data_path=/scratch/gpfs/KNORMAN/mindeyev2 --model_name=test --subj=1 --batch_size=128 --max_lr=3e-4 --mixup_pct=.66 --num_epochs=12 --ckpt_interval=999 --clip_scale=1. --blur_scale=100. --depth_scale=100.  --file_prefix=activelrnOCT23 --num_sessions=$1
# python pretrain.py --data_path=/scratch/gpfs/KNORMAN/mindeyev2 --model_name=test --subj=1 --batch_size=128 --n_samples_save=0 --max_lr=3e-5 --mixup_pct=.66 --num_epochs=12 --ckpt_interval=999 --file_prefix=activlrnOCT16
# loop sbatch --array=0-37 hipp.sh $i 
# for i in {27..37} ; do  sbatch --array=0-37 hipp.sh $i;  done