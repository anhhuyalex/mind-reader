#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=a40x
#SBATCH --job-name=act
#SBATCH --gres=gpu:1
#SBATCH --time=300:00:00          # total run time limit (HH:MM:SS)
#SBATCH --comment=medarc 
#SBATCH --nodes=1              
#SBATCH -o slurms/%j.out
#SBATCH --nodes=1              


# SBATCH -o slurms/%j.out
# SBATCH --ntasks-per-node=1     # should = number of gpus
cd /weka/proj-fmri/alexnguyen/mind-reader/MindEyeV2/src
# cd /weka/home-alexnguyen/mind-reader/MindEyeV2/src

source ~/.bashrc
source ./mindeye/bin/activate
# source /weka/home-alexnguyen/mind-reader/MindEyeV2/src/mindeye/bin/activate
# jupyter nbconvert main.ipynb --to python
jupyter nbconvert finetune.ipynb --to python
jupyter nbconvert kmeans.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=28
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Set random model name (if you run same model_name without wanting to resume from checkpoint, wandb will error)
model_name=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | head -c 10)
model_name="finetune_numsamples"
echo model_name=${model_name}

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

###########

NUM_SAMPLES=$1 # define number of samples to filter
TOTAL_SAMPLES=$2 # define total number of samples
DATA_PATH=/weka/home-alexnguyen/mindeyev2_dataset # define data path
# DATA_PATH=/weka/proj-fmri/shared/mindeyev2_dataset
# wandb login --relogin --host=https://stability.wandb.io
# sbatch --array=0-49 paul.slurm 6500 9000
# accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 main.py --data_path=/fsx/proj-fmri/shared/mindeyev2_dataset --model_name=${model_name} --no-use_prior --subj=1 --batch_size=28 --num_sessions=-1 --hidden_dim=2048 --n_blocks=4 --max_lr=3e-5 --mixup_pct=.33 --num_epochs=40 --ckpt_saving --wandb_log
# for i in 750 1500 2250 3000 6000 9000 12000 15000 18000 21000 24000 27000; do sbatch --array=1 paul.slurm $i; done
# for i in 10 100 1000 10000 100000; do ./mindeye/bin/accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 \
#     finetune.py --data_path=/weka/proj-fmri/shared/mindeyev2_dataset \
#     --model_name=${model_name} --no-use_prior \
#     --subj=1 --subj_list=1 --batch_size=28 \
#     --hidden_dim=2048 --n_blocks=4 --max_lr=3e-5 --mixup_pct=.33 --num_epochs=15 --stage2 \
#     --ckpt_saving --wandb_log --wandb_group_name=test \
#     --num_samples=$NUM_SAMPLES \
#     --resume_from_ckpt=../train_logs/stage1_prior_subj01_h2048_3e5 --num_sessions=-1 --seed=$i ; done   
    # --filter_samples=/weka/proj-fmri/alexnguyen/mind-reader/MindEyeV2/src/cache/keys_${NUM_SAMPLES}_from_${TOTAL_SAMPLES}_seed_${SLURM_ARRAY_TASK_ID}.npy \
    # --val_filter_samples=/weka/proj-fmri/alexnguyen/mind-reader/MindEyeV2/src/cache/validation_set.npy; done
# 20 200 2000 20000 200000
for i in 10 100; do /weka/home-alexnguyen/mind-reader/MindEyeV2/src/mindeye/bin/accelerate launch --num_processes=1 --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 \
    kmeans.py --data_path=${DATA_PATH} \
    --model_name=${model_name} --no-use_prior \
    --subj=1 --subj_list=1 --batch_size=28 \
    --hidden_dim=2048 --n_blocks=4 --max_lr=3e-5 --mixup_pct=.33 --num_epochs=15 --stage2 \
    --ckpt_saving --wandb_log --wandb_group_name=mar5_0527_voxel_kmeans_pruning \
    --num_samples=$NUM_SAMPLES \
    --cache_dir=/weka/home-alexnguyen \
    --resume_from_ckpt=/weka/home-alexnguyen/mind-reader/MindEyeV2/train_logs/stage1_prior_subj01_h2048_3e5 --num_sessions=-1 --seed=$i    \
    --filter_samples=/weka/home-alexnguyen/mind-reader/MindEyeV2/src/cache/kmeans_pruned_ids_N_${NUM_SAMPLES}_seed_${SLURM_ARRAY_TASK_ID}.npy; done