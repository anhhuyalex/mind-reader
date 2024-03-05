#!/usr/bin/env python
# coding: utf-8

# This script follows MindEyeV1 training procedure (e.g., training a diffusion prior and reconstructing with Versatile Diffusion) except that it uses the newer data loading procedure being used for MindEyeV2. 

# # Import packages & functions

# In[1]:


import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm

import webdataset as wds
import gc

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

from accelerate import Accelerator, DeepSpeedPlugin
# from deepspeed import DeepSpeedEngine

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils


# In[2]:


### Multi-GPU config ###
local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

# ## UNCOMMENT BELOW SECTION AND COMMENT OUT DEEPSPEED SECTION TO AVOID USING DEEPSPEED ###
use_deepspeed = False
accelerator = Accelerator(split_batches=False, mixed_precision="fp16") # ['no', 'fp8', 'fp16', 'bf16']
global_batch_size = 28
data_type = torch.float16 # change depending on your mixed_precision

### DEEPSPEED INITIALIZATION ###
# use_deepspeed = True
# import deepspeed
# if num_devices <= 1 and utils.is_interactive():
#     global_batch_size = batch_size = 28
#     print(f"Setting batch_size to {batch_size}")
#     # can emulate a distributed environment for deepspeed to work in jupyter notebook
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = str(np.random.randint(10000)+9000)
#     os.environ["RANK"] = "0"
#     os.environ["LOCAL_RANK"] = "0"
#     os.environ["WORLD_SIZE"] = "1"
#     os.environ["GLOBAL_BATCH_SIZE"] = str(global_batch_size) # set this to your batch size!
# else:
#     global_batch_size = os.environ["GLOBAL_BATCH_SIZE"]    
#     batch_size = int(os.environ["GLOBAL_BATCH_SIZE"]) // num_devices
#     if num_devices <= 1:
#         os.environ["RANK"] = "0"
#         os.environ["LOCAL_RANK"] = "0"
#         os.environ["WORLD_SIZE"] = "1"

# # alter the deepspeed config according to your global and local batch size
# if local_rank == 0:
#     with open('deepspeed_config_stage2_cpuoffload.json', 'r') as file:
#         config = json.load(file)
#     config['train_batch_size'] = int(os.environ["GLOBAL_BATCH_SIZE"])
#     config['train_micro_batch_size_per_gpu'] = batch_size
#     config['bf16'] = {'enabled': False}
#     config['fp16'] = {'enabled': True}
#     with open('deepspeed_config_stage2_cpuoffload.json', 'w') as file:
#         json.dump(config, file)
# else:
#     # give some time for the local_rank=0 gpu to prep new deepspeed config file
#     time.sleep(10)
# deepspeed_plugin = DeepSpeedPlugin("deepspeed_config_stage2_cpuoffload.json")
# accelerator = Accelerator(split_batches=False, deepspeed_plugin=deepspeed_plugin)


# In[3]:


print("PID of this process =",os.getpid())
device = accelerator.device
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = torch.cuda.device_count()
if num_devices==0 or not distributed: num_devices = 1
num_workers = num_devices
print(accelerator.state)

if not use_deepspeed: 
    batch_size = global_batch_size // num_devices

# set data_type to match your mixed precision (automatically set based on deepspeed config)
if accelerator.mixed_precision == "bf16":
    data_type = torch.bfloat16
elif accelerator.mixed_precision == "fp16":
    data_type = torch.float16
else:
    data_type = torch.float32

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0 


# # Configurations

# ## Training model on just 1 subject (Baseline)
# Set both "subj" and "subj_list" to the subject you want to train on.
# You can set "num_sessions" to the number of sessions of training data to include for the given subject. Setting it to -1 means using all possible sessions.
# 
# ## Training model on all subjects except 1 (Stage 1)
# Do not set "subj_list" and let it be set to default.
# Set "subj" to the subject you *don't* want to train on. Model will train across all the other subjects.
# 
# ## Fine-tuning a pre-trained model on a single subject (Stage 2)
# Set both "subj" and "subj_list" to the subject you want to fine-tune on.
# Set "stage2" and "resume_from_ckpt" to True and specify "model_name" as the same model_name used for stage 1. Should be the name of folder containing pre-trained checkpoint in "train_logs".
# You can set "num_sessions" to the number of sessions of training data to include for the given subject. Setting it to -1 means using all possible sessions.

# In[4]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    model_name = "stage2_noprior_subj01x"
    print("model_name:", model_name)

    # global_batch_size and batch_size should already be defined in the above cells
    # other variables can be specified in the following string:
    
    # # Stage 1
    # jupyter_args = f"--data_path=/weka/proj-fmri/shared/mindeyev2_dataset \
    #                 --model_name={model_name} --no-use_prior \
    #                 --subj=1 --batch_size={batch_size} --num_sessions=-1 \
    #                 --max_lr=3e-5 --mixup_pct=.33 --num_epochs=40 --no-ckpt_saving"
    
    # # Stage 2
    jupyter_args = f"--data_path=/weka/proj-fmri/shared/mindeyev2_dataset \
                    --seed=42\
                    --model_name={model_name} --no-use_prior \
                    --subj=1 --subj_list=1 --batch_size={batch_size} \
                    --max_lr=3e-5 --mixup_pct=.33 --num_epochs=20 --no-ckpt_saving --stage2 \
                    --resume_from_ckpt=../train_logs/stage1_prior_subj01_h2048_3e5 \
                    --filter_samples=./cache/subj01_permuted_samples.txt"
    
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output # function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[ ]:





# In[5]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--data_path", type=str, default="/weka/proj-fmri/shared/natural-scenes-dataset",
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--num_sessions", type=int, default=-1,
    help="Number of training sessions to include (-1 = all possible sessions)",
)
parser.add_argument(
    "--num_samples", type=int, default=-1, 
    help="Number of samples to filter from training")
parser.add_argument(
    "--use_prior",action=argparse.BooleanOptionalAction,default=True,
    help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
)
parser.add_argument(
    "--batch_size", type=int, default=32,
    help="Batch size can be increased by 10x if only training v2c and not diffusion diffuser",
)
parser.add_argument(
    "--filter_samples", type=str, default=None,
    help="file path that contains samples to filter",
)
parser.add_argument(
    "--val_filter_samples", type=str, default=None,
    help="file path that contains samples from train_url to filter and use for validation",
)
parser.add_argument(
    "--when_to_preload", type=str, default="before_each_epoch",
    choices=["before_training","before_each_epoch"],
    help="when to preload the dataset (before training or before each epoch)",
)
parser.add_argument(
    "--wandb_log",action=argparse.BooleanOptionalAction,default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--wandb_project",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--wandb_group_name",type=str,default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--resume_from_ckpt",type=str,default=None,
    help="if not using wandb and want to resume from a ckpt",
)
parser.add_argument(
    "--stage2",action=argparse.BooleanOptionalAction,default=False,
    help="fine-tuning from a pre-trained model trained across subjects?",
)
parser.add_argument(
    "--mixup_pct",type=float,default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--prior_mult",type=float,default=3,
    help="multiply diffusion prior loss by this",
)
parser.add_argument(
    "--use_image_aug",action=argparse.BooleanOptionalAction,default=True,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs",type=int,default=120,
    help="number of epochs of training",
)
parser.add_argument(
    "--subj_list", type=int, nargs='+', default=[1,2,3,4,5,6,7,8],
    help="number of subjects"
)
parser.add_argument(
    "--n_blocks",type=int,default=4,
)
parser.add_argument(
    "--hidden_dim",type=int,default=2048,
)
parser.add_argument(
    "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
)
parser.add_argument(
    "--max_lr",type=float,default=3e-5,
)

parser.add_argument(
    "--cache_dir",type=str,default='/weka/proj-fmri/shared/cache',
)
parser.add_argument(
    "--offline_cache_dir",type=str,default='./activelearning_cache',
)
parser.add_argument(
    "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--ckpt_interval",type=int,default=5,
    help="save backup ckpt and reconstruct every x epochs",
)
parser.add_argument(
    "--seed",type=int,default=None,
)

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
if args.seed is None:
    args.seed = time.time()
if len(subj_list)>1:
    subj_list.remove(subj)

print("subj_list", subj_list, "num_sessions", num_sessions)

if wandb_log and local_rank==0:
    import wandb


# In[6]:


outdir = os.path.abspath(f'./train_logs/{model_name}')
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir,exist_ok=True)
if use_image_aug:
    import kornia
    from kornia.augmentation.container import AugmentationSequential
    img_augment = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((224,224), (0.6,1), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomHorizontalFlip(p=0.3),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.3),
        kornia.augmentation.RandomGrayscale(p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )


# # Prep data, models, and dataloaders

# ## Creating wds dataloader, preload betas and all 73k possible images

# In[7]:


def my_split_by_node(urls): return urls
num_voxels_list = []

nsessions_allsubj=np.array([40, 40, 32, 30, 40, 32, 40, 30])
# if num_sessions == 0: num_sessions = nsessions_allsubj[s-1]
num_samples_per_epoch = 30000 // num_devices 


print("dividing batch size by subj_list, which will then be concatenated across subj during training...") 
batch_size = batch_size // len(subj_list)
if stage2 and len(subj_list)==1: # dividing batch size by 7 to retain same batch size as stage 1, which divides across subj_list
    batch_size = batch_size // 7
    print("setting batch_size to same as stage 1...")

num_iterations_per_epoch = num_samples_per_epoch // (batch_size*len(subj_list))

print("batch_size =", batch_size, "num_iterations_per_epoch =",num_iterations_per_epoch, "num_samples_per_epoch =",num_samples_per_epoch)


# In[8]:


train_data = {}
train_dl = {}
val_data = {}
val_dl = {}
num_voxels = {}
voxels = {}
for si, s in enumerate(args.subj_list):
    if args.num_sessions == -1:
        train_url = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
        print(f"subj0{args.subj_list[si]} training with {train_url} train_url")
    else:
        print(f"subj0{args.subj_list[si]} training with {args.num_sessions} sessions")
        train_url = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{args.num_sessions-1}" + "}.tar"
    
    if args.filter_samples is not None:
        
        filter_samples_list = np.load(args.filter_samples)
        print ("filter_samples_list", len(filter_samples_list))
        def filter_samples_func(sample):
            if sample["behav.npy"][0,0] not in filter_samples_list:
                return None
            return sample
        
    else:
        filter_samples_list = None
        def filter_samples_func(sample):
            return sample
        
    if args.val_filter_samples is not None:
        val_filter_samples_list = np.load(args.val_filter_samples)
        print ("val_filter_samples_list", len(val_filter_samples_list))
        def val_filter_samples_func(sample):
            if sample["behav.npy"][0,0] not in val_filter_samples_list:
                return None
            return sample
    else:
        val_filter_samples_list = None
        def val_filter_samples_func(sample):
            return sample
    # make training dataset and data loader
    train_data[f'subj0{s}'] = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=2500, rng=random.Random(args.seed))\
                        .decode("torch").map(filter_samples_func)\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
 
    # make validation dataset and data loader
    val_data[f'subj0{s}'] = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=2500, rng=random.Random(args.seed))\
                        .decode("torch").map(val_filter_samples_func)\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    val_dl[f'subj0{s}'] = torch.utils.data.DataLoader(val_data[f'subj0{s}'], batch_size=batch_size if args.val_filter_samples is None else len(val_filter_samples_list),
                                                      shuffle=False, drop_last=True, pin_memory=True)
    
    # Load hdf5 data for betas, but don't put everything into memory
    f = h5py.File(f'{args.data_path}/betas_all_subj0{s}_fp32.hdf5', 'r')
    
    betas = f['betas'][:]
    betas = torch.Tensor(betas).to("cpu").to(data_type)
    num_voxels_list.append(betas[0].shape[-1])
    num_voxels[f'subj0{s}'] = betas[0].shape[-1]
    voxels[f'subj0{s}'] = betas
    print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

print("Loaded all subj train dls and betas!\n")

# Validate only on the subject from first index of subj_list
num_test = [2770,2770,2113,1985,2770,2113,2770,1985] # maximum possible number of test samples per subj
test_url = f"{args.data_path}/wds/subj0{subj_list[0]}/test/" + "0.tar"
print(test_url)
test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test[int(subj_list[0])-1], shuffle=False, drop_last=False, pin_memory=True)
print(f"Loaded test dl for subj{subj_list[0]}! num_test={num_test[int(subj_list[0])-1]}\n")


# In[9]:


# Preload 73k NSD images
f = h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images'][:]
images = torch.Tensor(images).to("cpu").to(data_type)
print("Loaded all 73k possible NSD images to cpu!", images.shape)


# ## Load CLIP model

# In[10]:


from diffusers import VersatileDiffusionPipeline, UniPCMultistepScheduler
# if you get an error here, make sure your diffusers package is version 0.13.0!
# vd_pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion", torch_dtype=data_type, cache_dir=cache_dir)
# vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained("shi-labs/versatile-diffusion", subfolder="scheduler", cache_dir=cache_dir)
vd_pipe = VersatileDiffusionPipeline.from_pretrained("/weka/proj-fmri/shared/cache/versatile-diffusion", torch_dtype=data_type)
vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained("/weka/proj-fmri/shared/cache/versatile-diffusion", subfolder="scheduler")
vd_pipe.to(device)#(torch.device(f"cuda:{local_rank}"))
clip_model = vd_pipe.image_encoder
clip_model.to(data_type)
clip_model.eval()
clip_model.requires_grad_(False)
clip_seq_dim = 257
clip_emb_dim = 768

## testing it out:
# utils.get_clip_embeddings(clip_model,images[:1].to(device))


# ## MindEye modules

# In[11]:


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x
        
model = MindEyeModule()
model


# In[12]:


class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_sizes, out_features): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
        self.temp = nn.Parameter(torch.Tensor([5.3]))
        self.bias = nn.Parameter(torch.Tensor([-2.]))
    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x)
        return out
        
model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)
utils.count_params(model.ridge)
utils.count_params(model)

# test on subject 1 with fake data
b = torch.randn((2,num_voxels_list[0]))
print(b.shape, model.ridge(b,0).shape)


# In[13]:


class BrainNetwork(nn.Module):
    def __init__(self, out_dim=768, in_dim=15724, h=4096, n_blocks=n_blocks, drop=.15, clip_size=768, use_projector=True):
        super().__init__()
        self.h = h
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        self.use_projector = use_projector
        
        self.mlps = nn.ModuleList([
            self.mlp(h, h, drop) for _ in range(n_blocks)
        ])
        
        self.lin1 = nn.Linear(h, out_dim, bias=True)
        self.clip_proj = self.projector(clip_size, clip_size)
            
    def projector(self, in_dim, out_dim):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, out_dim)
        )
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
        
    def forward(self, x):
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlps[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)
        if self.use_projector: return x, self.clip_proj(x.reshape(len(x), -1, self.clip_size))
        return x

model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim) 
utils.count_params(model.backbone)
utils.count_params(model)

# test that the model works on some fake data
b = torch.randn((2,hidden_dim))
print("in",b.shape)
backbone_, clip_ = model.backbone(b)
print("out",backbone_.shape, clip_.shape)


# ## Load diffusion prior + Versatile Diffusion

# In[14]:


if use_prior:
    from models import *

    # setup diffusion prior network
    out_dim = clip_emb_dim
    depth = 6
    dim_head = 64
    heads = clip_emb_dim//64 # heads * dim_head = 12 * 64 = 768
    guidance_scale = 3.5
    timesteps = 100

    prior_network = VersatileDiffusionPriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = clip_seq_dim,
            learned_query_mode="pos_emb"
        )

    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=None,
    )
    
    utils.count_params(model.diffusion_prior)
    utils.count_params(model)


# ## Setup optimizer and learning rate scheduler

# In[15]:


no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

if use_prior:
    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
else:
    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(num_epochs*num_iterations_per_epoch)),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    total_steps=int(np.floor(num_epochs*num_iterations_per_epoch))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )
    
def save_ckpt(tag):    
    if stage2:
        tag = tag + "_stage2"
    if use_deepspeed:
        deepspeed.DeepSpeedEngine.save_checkpoint(model, save_dir=outdir, tag=tag)
        ckpt_path = outdir+f'/{tag}/{tag}.npy'
        np.save(ckpt_path, {
            'epoch': epoch,
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs})
    else:
        ckpt_path = outdir+f'/{tag}.pth'
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
        del unwrapped_model
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")
    
from collections import OrderedDict
def filter_params(odict):
    return OrderedDict((k, v) for k, v in odict.items() if (not k.startswith('ridge')) and (not k.startswith('ridge.linears.0')))

def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,stage2=False): 
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    if use_deepspeed:
        state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
        if stage2: state_dict = filter_params(checkpoint['model_state_dict'])
        model.load_state_dict(state_dict, strict=strict)
        if load_epoch:
            np_ckpt = np.load(outdir+f'/{tag}/{tag}.npy', allow_pickle=True).tolist()
            globals()["epoch"] = np_ckpt['epoch']
            print("Epoch",epoch)
    else:
        print (f"loading from {tag}/last.pth")
        checkpoint = torch.load(tag+'/last.pth', map_location='cpu')
        if stage2: 
            state_dict = filter_params(checkpoint['model_state_dict'])
        else:
            state_dict = checkpoint['model_state_dict']
        if load_epoch:
            globals()["epoch"] = checkpoint['epoch']
            print("Epoch",epoch)
        if load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if load_lr:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        model.load_state_dict(state_dict, strict=strict)
        del checkpoint
        
print("\nDone with model preparations!")
num_params = utils.count_params(model)


# # Main

# In[16]:


epoch = 0
losses, test_losses, val_losses, lrs = [], [], [], []
best_test_loss = 1e9
torch.cuda.empty_cache()


# In[ ]:





# In[17]:


# setup weights and biases (optional)
if local_rank==0 and wandb_log: # only use main process for wandb logging
    wandb_project = 'eye_int'
    print(f"wandb {wandb_project} run {model_name}")
    wandb.login(host='https://stability.wandb.io') # need to configure wandb environment beforehand
    if args.filter_samples is not None:
        wandb_model_name = model_name + f"{args.filter_samples.split('/')[-1].split('.')[0]}"
    else:
        wandb_model_name = model_name
    offline_logfile = f"{wandb_model_name}.log"
    wandb_config = {
      "model_name": wandb_model_name,
      "global_batch_size": global_batch_size,
      "batch_size": batch_size,
      "num_epochs": num_epochs,
      "num_params": num_params,
      "use_image_aug": use_image_aug,
      "max_lr": max_lr,
      "mixup_pct": mixup_pct,
      "num_samples_per_epoch": num_samples_per_epoch,
      "num_test": num_test,
      "ckpt_interval": ckpt_interval,
      "ckpt_saving": ckpt_saving,
      "seed": seed,
      "distributed": distributed,
      "num_devices": num_devices,
      "world_size": world_size,
      "train_url": train_url,
      "test_url": test_url,
      "num_sessions": args.num_sessions,
    #   "samples_list": np.array2string(filter_samples_list, separator=','),
      "offline_logfile": f"{args.offline_cache_dir}/{offline_logfile}",
      "filter_samples": args.filter_samples
    }
    print("wandb_config:\n",wandb_config, args)   
    print("wandb_id:",wandb_model_name)
    wandb.init(
        project=wandb_project,
        name=wandb_model_name,
        config=wandb_config,
        resume="allow",
        group=args.wandb_group_name
    )
    utils.save_file_pickle(f"{args.offline_cache_dir}/{offline_logfile}", {
            "log": "dataset",
            "filter_samples": np.load(args.filter_samples) if args.filter_samples is not None else None,
        })
else:
    wandb_log = False

# load saved ckpt model weights into current model
if not stage2:
    if resume_from_ckpt:
        load_ckpt("last",load_lr=True,load_optimizer=True,load_epoch=True,strict=True)
    elif wandb_log:
        if wandb.run.resumed:
            load_ckpt("last",load_lr=True,load_optimizer=True,load_epoch=True,strict=True)
else:
    assert resume_from_ckpt is not None 
    if resume_from_ckpt:
        load_ckpt(resume_from_ckpt,load_lr=False,load_optimizer=False,load_epoch=False,strict=False,stage2=True)
    # if wandb_log:
    #     if wandb.run.resumed:
    #         load_ckpt("last",load_lr=True,load_optimizer=True,load_epoch=True,strict=True,stage2=False)
        
    model_name = model_name + "_stage2"
    print("new model_name:", model_name)


# In[18]:


train_dls = [train_dl[f'subj0{s}'] for s in subj_list]
model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)
# leaving out test_dl since we will only have local_rank 0 device do evals


# In[19]:


print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
val_image, val_voxel = None, None
test_image, test_voxel = None, None
mse = nn.MSELoss()
l1 = nn.L1Loss()
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
utils.seed_everything(seed=args.seed, cudnn_deterministic=True)
skip_train = True if epoch>=(num_epochs-1) else False # skip training if you are resuming from a fully trained model

def pre_load_all_batches(epoch, num_iterations_per_epoch, dls, subj_list, images, voxels, batch_size, mixup_pct, num_epochs, data_type):
    voxel_iters = {} # empty dict because diff subjects have differing # of voxels
    image_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), 3, 224, 224).float()
    annot_iters = {}
    perm_iters, betas_iters, select_iters = {}, {}, {}
    for s, dl in enumerate(dls):
        im_sizes = []
        with torch.cuda.amp.autocast(dtype=data_type):
            for iter, (behav0, past_behav0, future_behav0, old_behav0) in enumerate(dl):    
                image0 = images[behav0[:,0,0].cpu().long()].float()
                # assert image0.shape[0] == 28, "image0 shape is not 28"
                im_sizes.append(image0.shape[0])
                image_iters[iter,s*batch_size:s*batch_size+batch_size] = image0
                
                voxel0 = voxels[f'subj0{subj_list[s]}'][behav0[:,0,5].cpu().long()]
                voxel0 = torch.Tensor(voxel0).to(data_type)

                if epoch < int(mixup_pct * num_epochs):
                    voxel0, perm, betas, select = utils.mixco(voxel0)
                    perm_iters[f"subj0{subj_list[s]}_iter{iter}"] = perm
                    betas_iters[f"subj0{subj_list[s]}_iter{iter}"] = betas
                    select_iters[f"subj0{subj_list[s]}_iter{iter}"] = select

                voxel_iters[f"subj0{subj_list[s]}_iter{iter}"] = voxel0

                if iter >= num_iterations_per_epoch-1:
                    break

    return voxel_iters, image_iters, perm_iters, betas_iters, select_iters
if args.when_to_preload == "before_training":
    voxel_iters, image_iters, perm_iters, betas_iters, select_iters = pre_load_all_batches(0, num_iterations_per_epoch, train_dls, subj_list, images, voxels, batch_size, mixup_pct, num_epochs, data_type)
    print ("pre-loaded all batches for training")
    w
for epoch in progress_bar:
    model.train()

    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    test_fwd_percent_correct = 0.
    test_bwd_percent_correct = 0.
    val_fwd_percent_correct = 0.
    val_bwd_percent_correct = 0.

    loss_clip_total = 0.
    test_loss_clip_total = 0.
    loss_prior_total = 0.
    test_loss_prior_total = 0.
    val_loss_clip_total = 0.
    val_loss_prior_total = 0.

    # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
    if args.when_to_preload == "before_each_epoch":
        voxel_iters, image_iters, perm_iters, betas_iters, select_iters = pre_load_all_batches(epoch, num_iterations_per_epoch, train_dls, subj_list, images, voxels, batch_size, mixup_pct, num_epochs, data_type)
    
    # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
    if skip_train is False:
        for train_i in np.random.permutation(num_iterations_per_epoch): # randomize the order of batches (important if we pre-load all batches)
            with torch.cuda.amp.autocast(dtype=data_type):
                optimizer.zero_grad()

                voxel_list = [voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                image = image_iters[train_i].detach().to(device)

                if use_image_aug: 
                    image = img_augment(image)

                clip_target = utils.get_clip_embeddings(clip_model,image)
                assert not torch.any(torch.isnan(clip_target))

                if epoch < int(mixup_pct * num_epochs):
                    perm_list = [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                    perm = torch.cat(perm_list, dim=0)
                    betas_list = [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                    betas = torch.cat(betas_list, dim=0)
                    select_list = [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                    select = torch.cat(select_list, dim=0)

                voxel_ridge_list = [model.ridge(voxel_list[si],si) for si,s in enumerate(subj_list)]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

                clip_voxels, clip_voxels_proj = model.backbone(voxel_ridge)

                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                
                if epoch < int(mixup_pct * num_epochs):                
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006,
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)
                    
                loss_clip_total += loss_clip.item()
                loss = loss_clip 
                print ("loss_clip", loss_clip.item())
                
                if use_prior:
                    loss_prior, aligned_clip_voxels = model.diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)
                    aligned_clip_voxels /= model.diffusion_prior.image_embed_scale
                    loss_prior_total += loss_prior.item()
                    loss_prior *= prior_mult
                    loss += loss_prior

                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
                
                utils.check_loss(loss)
                accelerator.backward(loss)
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                if lr_scheduler_type is not None:
                    lr_scheduler.step()

    model.eval()
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            for val_i, (behav, past_behav, future_behav, old_behav) in enumerate(val_dl[f'subj0{subj_list[0]}']):
                if val_image is None:
                    voxel = voxels[f'subj0{subj_list[0]}'][behav[:,0,5].cpu().long()]
                    image = behav[:,0,0].cpu().long()

                    unique_image, sort_indices = torch.unique(image, return_inverse=True)
                    for im in unique_image:
                        locs = torch.where(im == image)[0]
                        if val_image is None:
                            val_image = images[im][None]
                            val_voxel = torch.mean(voxel[locs],axis=0)[None]
                        else:
                            val_image = torch.vstack((val_image, images[im][None]))
                            val_voxel = torch.vstack((val_voxel, torch.mean(voxel[locs],axis=0)[None]))

                val_indices = torch.arange(len(val_voxel))[:300]
                voxel = val_voxel[val_indices].to(device)
                image = val_image[val_indices].to(device)
                assert len(image) == 300

                clip_target = utils.get_clip_embeddings(clip_model,image.float())

                voxel_ridge = model.ridge(voxel,0)

                clip_voxels, clip_voxels_proj = model.backbone(voxel_ridge)

                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                loss_clip = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006)
                
                val_loss_clip_total += loss_clip.item()
                loss = loss_clip

                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                val_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()    
                val_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                utils.check_loss(loss)
                val_losses.append(loss.item())

            for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):  
                ## Average same-image repeats ##
                if test_image is None:
                    voxel = voxels[f'subj0{subj_list[0]}'][behav[:,0,5].cpu().long()]
                    image = behav[:,0,0].cpu().long()

                    unique_image, sort_indices = torch.unique(image, return_inverse=True)
                    for im in unique_image:
                        locs = torch.where(im == image)[0]
                        if test_image is None:
                            test_image = images[im][None]
                            test_voxel = torch.mean(voxel[locs],axis=0)[None]
                        else:
                            test_image = torch.vstack((test_image, images[im][None]))
                            test_voxel = torch.vstack((test_voxel, torch.mean(voxel[locs],axis=0)[None]))

                test_indices = torch.arange(len(test_voxel))[:300]
                voxel = test_voxel[test_indices].to(device)
                image = test_image[test_indices].to(device)
                print ("device", device)
                assert len(image) == 300

                clip_target = utils.get_clip_embeddings(clip_model,image.float())

                voxel_ridge = model.ridge(voxel,0) # using ridge from 0th index of subj_list

                clip_voxels, clip_voxels_proj = model.backbone(voxel_ridge)
                
                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                
                loss_clip = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006)
                
                test_loss_clip_total += loss_clip.item()
                loss = loss_clip
                
                if use_prior:
                    loss_prior, aligned_clip_voxels = model.diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)
                    aligned_clip_voxels /= model.diffusion_prior.image_embed_scale
                    test_loss_prior_total += loss_prior.item()
                    loss_prior *= prior_mult
                    loss += loss_prior

                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
                
                utils.check_loss(loss)                
                test_losses.append(loss.item())
                break

            # if utils.is_interactive(): clear_output(wait=True)
            if skip_train: break
            print("---")

            assert (test_i+1) == 1
            logs = {
                "epoch": epoch,
                "train/loss": np.mean(losses[-(train_i+1):]),
                "test/loss": np.mean(test_losses[-(test_i+1):]),
                "val/loss": np.mean(val_losses[-(val_i+1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "test/num_steps": len(test_losses),
                "val/num_steps": len(val_losses),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                "val/val_fwd_pct_correct": val_fwd_percent_correct / (val_i + 1),
                "val/val_bwd_pct_correct": val_bwd_percent_correct / (val_i + 1),
                "train/loss_clip_total": loss_clip_total / (train_i + 1),
                "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                "train/loss_prior_total": loss_prior_total / (train_i + 1),
                "test/loss_prior_total": test_loss_prior_total / (test_i + 1),
                }
            
            if use_prior: # output recons every ckpt
                if (epoch == num_epochs-1) or (epoch % ckpt_interval == 0):
                    print("reconstructing...")
                    voxel_ridge = model.ridge(voxel[:1],0)
                    clip_voxels, clip_voxels_proj = model.backbone(voxel_ridge)
                    clip_model.to(torch.float32).to("cpu")
                    grid, _, _, _ = utils.reconstruction(
                        image, voxel, clip_voxels.reshape(-1,257,768), clip_voxels_proj,
                        clip_model,
                        vd_pipe.image_unet, vd_pipe.vae, vd_pipe.scheduler,
                        diffusion_priors = model.diffusion_prior,
                        num_inference_steps = 20,
                        n_samples_save = 1,
                        guidance_scale = guidance_scale,
                        timesteps_prior = timesteps,
                        seed = seed,
                        retrieve = False,
                        plotting = True,
                        img_variations = False,
                        verbose = False,
                    )
                    clip_model.to(torch.device(f"cuda:{local_rank}")).to(data_type)
                    if wandb_log:
                        logs[f"test/recons"] = wandb.Image(grid, caption=f"epoch{epoch:03d}")
                        plt.close()
                    else:
                        plt.show()

            progress_bar.set_postfix(**logs)

            # Save model checkpoint and reconstruct
            if (ckpt_saving) and (epoch % ckpt_interval == 0):
                save_ckpt(f'last')

            if wandb_log: wandb.log(logs)

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    gc.collect()

print("\n===Finished!===\n")
if ckpt_saving:
    save_ckpt(f'last')
if not utils.is_interactive():
    sys.exit(0)


# In[ ]:


plt.plot(losses)
plt.show()
plt.plot(test_losses)
plt.show()

