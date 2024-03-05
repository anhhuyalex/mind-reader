#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=g40x
#SBATCH --job-name=eye
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1     # should = number of gpus
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00          # total run time limit (HH:MM:SS)
#SBATCH -o slurms/%j.out
#SBATCH --comment=fmri

source ~/.bashrc
source mindeye/bin/activate

module load cuda/11.7 # should match torch.cuda.version

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=28
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

model_name="stage2_noprior_subj01x"
echo model_name=${model_name}

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

export WANDB_DIR="/fsx/proj-fmri/paulscotti/MindEyeV2/wandb/"
export WANDB_CACHE_DIR="/fsx/home-paulscotti/.cache"
export WANDB_MODE="online"

echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

num_samples=$1
###########
# rm slurms/* && sbatch accel.slurm 2
cd /fsx/proj-fmri/alexnguyen/mind-reader/MindEyeV2/src
#CUDA_LAUNCH_BLOCKING=1 
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 \
    finetune.py --data_path=/fsx/proj-fmri/shared/mindeyev2_dataset \
    --cache_dir=/fsx/proj-fmri/shared/cache \
    --filter_samples=cache/subj06762_samples_6761.txt \
    --model_name=${model_name}  --stage2 --subj=1 \
    --subj_list=1 --num_samples=${num_samples} \
    --max_lr=3e-5 --mixup_pct=.33 --num_epochs=40  \
    --file_prefix=finetune_subj1_perf_vs_numsession_NOV24 --resume_from_ckpt=pretrainNOV16_1700670984.4315398.pkl \
    --seed=42 --debug # --use_image_aug