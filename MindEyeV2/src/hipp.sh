#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --mem-per-cpu=40G
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
# python main.py --data_path=/scratch/gpfs/KNORMAN/mindeyev2 --model_name=test --subj=1 --batch_size=128 --max_lr=3e-4 --mixup_pct=.66 --num_epochs=12 --ckpt_interval=999 --clip_scale=1. --blur_scale=100. --depth_scale=100.  --file_prefix=activelrnOCT23 --num_sessions=$1
# python data_pruning.py --data_path=/scratch/gpfs/KNORMAN/mindeyev2 --subj_sample_list=cache/subj1_sample_list_len_6761_seed_42.txt --model_name=test --subj=1 --batch_size=128 --max_lr=3e-4 --mixup_pct=.66 --num_epochs=12 --ckpt_interval=999 --clip_scale=1. --blur_scale=100. --depth_scale=100.  --file_prefix=pruningNOV6  --num_samples=6761
# python pretrain.py --data_path=/scratch/gpfs/KNORMAN/mindeyev2 --model_name=stage1_noprior_subj01x --model_name=test --subj=1 --num_sessions=-1 --max_lr=3e-5 --mixup_pct=.33 --num_epochs=40 --use_image_aug --file_prefix=pretrainNOV16  
python finetune.py --data_path=/scratch/gpfs/KNORMAN/mindeyev2 --model_name=stage2_noprior_subj01x  --subj=1 --subj_list=1 --num_sessions=-1 --max_lr=3e-5 --mixup_pct=.33 --num_epochs=-1 --use_image_aug --file_prefix=pretrainNOV16 --resume_from_ckpt=pretrainNOV16_1700670984.4315398.pkl

# loop sbatch --array=0-37 hipp.sh $i 

# for i in {27..37} ; do  sbatch --array=0-37 hipp.sh $i;  done
# for i in  1    2    2    3    4    5    7    9   11   14   18   24 30   39   50   64   82  105  134  171  219  279  357  456  583  745 952 1216 1554 1986 2537 3242 4142 5292 6760; do sbatch --array=0-37 hipp.sh $i; done