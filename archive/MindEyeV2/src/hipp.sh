#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --mem-per-gpu=40G
#SBATCH --time=12:00:00
#SBATCH --output _mindreader-%J.log
#SBATCH --partition=della-gpu
#SBATCH --gres=gpu:1




# SBATCH --mail-type=ALL
# SBATCH --mail-user=alexn@minerva.kgi.edu
 

# SBATCH --gpus=2
# SBATCH --ntasks=1
# source activate rtsynth
# SBATCH --constraint=gpu80

source activate neu502b
# sbatch --array=0-37 hipp.sh 2
# python -u download.py
# export MODEL_SAVE_FOLDER='/home/an633/project/CuriousContrast/results_alex'
# srun --pty -p della-gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash
# srun --pty -p psych_gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash
python main.py --data_path=/scratch/gpfs/KNORMAN/mindeyev2 --model_name=test --subj=1 --batch_size=32 --n_samples_save=0 --max_lr=3e-5 --mixup_pct=.66 --num_epochs=12 --ckpt_interval=999 --file_prefix=sessions_1_2 --num_sessions=$1
# loop sbatch --array=0-37 hipp.sh $i 
# for i in {1..37} ; do  sbatch --array=0-37 hipp.sh $i;  done