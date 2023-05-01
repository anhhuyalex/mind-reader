#!/bin/bash
#SBATCH --job-name=hipp_test
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=80G
#SBATCH --time=12:00:00
#SBATCH --partition=della-gpu
#SBATCH --output _mindreader-%J.log
#SBATCH --gres=gpu:1

# SBATCH --gpus=1
# SBATCH --ntasks=1



# SBATCH --mail-type=ALL
# SBATCH --mail-user=alexn@minerva.kgi.edu
# source activate rtsynth
source activate neu502b
# export MODEL_SAVE_FOLDER='/home/an633/project/CuriousContrast/results_alex'
# srun --pty -p della-gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash
# srun --pty -p psych_gpu -c 2 -t 4:00:00 --gres=gpu:1 --mem-per-cpu=10G bash
# sbatch hipp.sh 2 && sbatch hipp.sh 4 && sbatch hipp.sh 8 && sbatch hipp.sh 12 && sbatch hipp.sh 20 && sbatch hipp.sh 30 && sbatch hipp.sh 40 
# for num_inputs in {2..30..2}; do sbatch hipp.sh $num_inputs; done
# for num_inputs in {2..30..2}; do 
#     for first_layer_l1_regularize in {0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5}; do 
#         sbatch hipp.sh $num_inputs $first_layer_l1_regularize; 
#     done;
# done

# python -u exp.py --save_dir /scratch/gpfs/qanguyen/renorm --model_name attn --pixel_shuffled
# python -u exp.py --save_dir /scratch/gpfs/qanguyen/renorm_freezeconv --freeze_epoch 0 --model_name vgg11
# python -u exp.py --save_dir /gpfs/milgram/scratch60/turk-browne/an633/renorm_freezeconv --model_name cnn --freeze_conv
python -u mind_reader.py --batch_size 16 --run_dir /scratch/gpfs/qanguyen/mind_reader                                             --train_url \{/scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_avg_split/train/train_subj01_\{0..17\}.tar,/scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_avg_split/val/val_subj01_0.tar\}                                     --train_indices_batchidx_dict_url /scratch/gpfs/KNORMAN/alex_mindreader/subj01_train_indices_batchidx_dict.pth.tar                 --val_url /scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_avg_split/test/test_subj01_\{0..1\}.tar                                 --val_indices_batchidx_dict_url /scratch/gpfs/KNORMAN/alex_mindreader/subj01_test_indices_batchidx_dict.pth.tar                   --subj01_vitb32text_train_pred_clips_url /scratch/gpfs/KNORMAN/alex_mindreader/subj01_vitb32text_train_pred_clips.npy             --subj01_vitb32image_train_pred_clips_url /scratch/gpfs/KNORMAN/alex_mindreader/subj01_vitb32image_train_pred_clips.npy           --subj01_vitb32text_test_pred_clips_url /scratch/gpfs/KNORMAN/alex_mindreader/subj01_vitb32text_test_pred_clips.npy               --subj01_vitb32image_test_pred_clips_url /scratch/gpfs/KNORMAN/alex_mindreader/subj01_vitb32image_test_pred_clips.npy

 
# nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
# python -u mind_reader.py --run_dir /gpfs/milgram/scratch60/turk-browne/an633/mind_reader --batch_size 8
