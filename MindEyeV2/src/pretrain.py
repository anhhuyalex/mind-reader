import os
import sys
import json
import argparse
import numpy as np
import math 
from einops import rearrange
import time
import random
import h5py
from tqdm import tqdm
from functools import partial
from collections import defaultdict
import string

import webdataset as wds
import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

import utils
from accelerate import Accelerator, DeepSpeedPlugin
from deepspeed import DeepSpeedEngine
from diffusers import VersatileDiffusionPipeline, UniPCMultistepScheduler


from models import Clipper

def get_local_rank():
    local_rank = os.getenv('RANK')
    if local_rank is None: 
        local_rank = 0
    else:
        local_rank = int(local_rank)
    print("LOCAL RANK ", local_rank)  
    return local_rank

def setup_gpu_configs():
    torch.backends.cuda.matmul.allow_tf32 = True

    num_devices = torch.cuda.device_count()

    # get num_devices
    # if num_devices==0: 
    #     num_devices = 1
    # if num_devices <= 1 and utils.is_interactive():
    #     # can emulate a distributed environment for deepspeed to work in jupyter notebook
    #     os.environ["MASTER_ADDR"] = "localhost"
    #     os.environ["MASTER_PORT"] = str(np.random.randint(10000)+9000)
    #     os.environ["RANK"] = "0"
    #     os.environ["LOCAL_RANK"] = "0"
    #     os.environ["WORLD_SIZE"] = "1"
    #     os.environ["GLOBAL_BATCH_SIZE"] = "28" # set this to your batch size!
    #     global_batch_size = os.environ["GLOBAL_BATCH_SIZE"]

    # # set GLOBAL_BATCH_SIZE if not already set
    # try: 
    #     print("GLOBAL_BATCH_SIZE", os.environ["GLOBAL_BATCH_SIZE"])
    # except:
    #     os.environ["GLOBAL_BATCH_SIZE"] = "28" # set this to your batch size!
    return num_devices

def get_deepspeed_accelerator(args, num_devices):
    # alter the deepspeed config according to your global and local batch size
    # if local_rank == 0:
    # with open('deepspeed_config_stage2.json', 'r') as file:
    #     config = json.load(file)
    # config['train_batch_size'] = int(os.environ["GLOBAL_BATCH_SIZE"])
    # config['train_micro_batch_size_per_gpu'] = args.batch_size
    # config['bf16'] = {'enabled': False}
    # config['fp16'] = {'enabled': True}
    # with open('deepspeed_config_stage2.json', 'w') as file:
    #     json.dump(config, file)
    # else:
    #     # give some time for the local_rank=0 gpu to prep new deepspeed config file
    #     time.sleep(10)

    accelerator = Accelerator(split_batches=False, mixed_precision="fp16") # ['no', 'fp8', 'fp16', 'bf16']
    # change depending on your mixed_precision
    # deepspeed_plugin = DeepSpeedPlugin("deepspeed_config_stage2.json")
    # accelerator = Accelerator(split_batches=False, deepspeed_plugin=deepspeed_plugin)
    # print("\033[94m" + f"INFO: Using deepspeed_plugin {deepspeed_plugin} accelerator {accelerator}" + "\033[0m") 
    args.num_devices = num_devices
    return accelerator

def get_training_params(accelerator):
    print = accelerator.print
    print("PID of this process =",os.getpid())
    args.global_batch_size = 28
    device = accelerator.device
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == 'NO'
    print("device:",device)
    
    
    if args.num_devices==0 or not distributed: 
        args.num_devices = 1
    num_workers = args.num_devices
    args.batch_size = args.global_batch_size // num_devices

    print(accelerator.state)
   
    print("distributed =",distributed, "num_devices =", args.num_devices, "local rank =", local_rank, "world size =", world_size, "args.batch_size", args.batch_size)
    
    # set data_type to match your mixed precision (automatically set based on deepspeed config)
    if accelerator.mixed_precision == "bf16":
        data_type = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        data_type = torch.float16
    else:
        data_type = torch.float32
 
    
    return device, num_workers, world_size, distributed, print, data_type
 

def get_parser_args(num_devices):
    parser = argparse.ArgumentParser(description="Model Training Configuration")
    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="name of model, used for ckpt saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--data_path", type=str, default="/fsx/proj-fmri/shared/natural-scenes-dataset",
        help="Path to where NSD data is stored / where to download it to",
    )
    parser.add_argument(
        "--cache_dir",type=str,default='./cache',
    )
    parser.add_argument(
        "--subj_list", type=int, nargs='+', default=[1,2,3,4,5,6,7,8],
        help="number of subjects"
    )
    parser.add_argument(
        "--subj",type=int, default=1, choices=[1,2,5,7],
    )
    parser.add_argument(
        "--num_sessions", type=int, default=-1,
        help="Number of training sessions to include (-1 = all possible sessions)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=28,
        help="Batch size can be increased by 10x if only training v2c and not diffusion prior",
    )
   
    parser.add_argument(
        "--resume_from_ckpt",
        # action=argparse.BooleanOptionalAction,
        action='store_true',
        default=False,
        help="if not using wandb and want to resume from a ckpt",
    )
    
    parser.add_argument(
        "--stage2",
        # action=argparse.BooleanOptionalAction,
        action='store_true',
        default=False,
        help="fine-tuning from a pre-trained model trained across subjects?",
    )
    parser.add_argument(
        "--mixup_pct",type=float,default=.33,
        help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
    )
    parser.add_argument(
        "--blurry_recon",action='store_true',
        help="whether to output blurry reconstructions",
    )
    parser.add_argument(
        "--depth_recon",action='store_true',
        help="whether to output depth reconstructions",
    )
    parser.add_argument(
        "--blur_scale",type=float,default=100.,
        help="multiply loss from blurry recons by this number",
    )
    parser.add_argument(
        "--depth_scale",type=float,default=100.,
        help="multiply loss from depth recons by this number",
    )
    parser.add_argument(
        "--clip_scale",type=float,default=1.,
        help="multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--use_image_aug",
        action='store_true',
        # action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to use image augmentation",
    )
     
    parser.add_argument(
        "--num_epochs",type=int,default=240,
        help="number of epochs of training",
    )
    # optimizer / scheduler params 
    parser.add_argument(
        "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
    )
    parser.add_argument(
        "--n_blocks",type=int,default=4,
    )
    parser.add_argument(
        "--hidden_dim",type=int,default=2048,
    )

    parser.add_argument(
        "--ckpt_saving",
        action='store_true',
        # action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ckpt_interval",type=int,default=5,
        help="save backup ckpt and reconstruct every x epochs",
    )
    parser.add_argument(
        "--seed",type=int,default=0,
    )
    parser.add_argument(
        "--max_lr",type=float,default=3e-4,
    )
    
    parser.add_argument("--file_prefix", type=str, default="",
        action = "store",
        help="prefix for ckpt files")
    parser.add_argument("--debug", '-d',
        action='store_true', 
        help = "debug mode", 
        default=False)
    

    if utils.is_interactive():
        args = parser.parse_args(jupyter_args)
    else:
        args = parser.parse_args()

    if len(args.subj_list)>1:
        args.subj_list.remove(args.subj)

    print("subj_list", args.subj_list, "num_sessions", args.num_sessions)

    args.batch_size = int(args.batch_size / num_devices)
    print("batch_size", args.batch_size )

    # get output directory
    args.outdir = os.path.abspath(f'./train_logs')
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir,exist_ok=True)

   
    # get model params
    args.clip_seq_dim = 257
    args.clip_emb_dim = 768

    # save params
    args.exp_name = f"{args.file_prefix}_{time.time()}"

    # seed
    args.seed = args.seed if args.seed else int(time.time())
    return args

def get_img_augmentations(args):
    img_augment = None 
    if args.use_image_aug:
        print ("\033[91m" + "WARNING: Using image augmentation" + "\033[0m")
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
    else:
        # print in red
        print ("\033[91m" + "WARNING: not using image augmentation" + "\033[0m")
    return img_augment 
 

def get_data_loaders(args):
    def my_split_by_node(urls): return urls
    num_voxels_list = []
    nsessions_allsubj=np.array([40, 40, 32, 30, 40, 32, 40, 30])
    # if args.num_sessions == 0: args.num_sessions = nsessions_allsubj[s-1]
    num_samples_per_epoch = 30000 // args.num_devices 

    print("dividing batch size by subj_list, which will then be concatenated across subj during training...") 

    args.batch_size = args.batch_size // len(args.subj_list)
    

    if args.stage2 and len(args.subj_list)==1: # dividing batch size by 7 to retain same batch size as stage 1, which divides across subj_list
        args.batch_size = args.batch_size // 7
        print("setting batch_size to same as stage 1...")

    args.num_iterations_per_epoch = num_samples_per_epoch // (args.batch_size*len(args.subj_list))

    train_data = {}
    train_dl = {}
    num_voxels = {}
    voxels = {}
    for si, s in enumerate(args.subj_list):
        print ("subj", s, si)
        if args.num_sessions == -1:
            print(f"subj0{args.subj_list[si]} training with {nsessions_allsubj[s-1]} sessions")
            train_url = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
        else:
            print(f"subj0{args.subj_list[si]} training with {args.num_sessions} sessions")
            train_url = f"{args.data_path}/wds/subj0{s}/train/" + "{0.." + f"{args.num_sessions-1}" + "}.tar"
        print(train_url)
        train_data[f'subj0{s}'] = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
        train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(train_data[f'subj0{s}'], batch_size=args.batch_size, shuffle=False, drop_last=True, pin_memory=True)
        # Load hdf5 data for betas, but don't put everything into memory
        f = h5py.File(f'{args.data_path}/betas_all_subj0{s}_fp32.hdf5', 'r')
        betas = f['betas'][:]
        betas = torch.Tensor(betas).to("cpu").to(args.data_type)
        num_voxels_list.append(betas[0].shape[-1])
        num_voxels[f'subj0{s}'] = betas[0].shape[-1]
        voxels[f'subj0{s}'] = betas
        print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")
    print("Loaded all subj train dls and betas!\n")
    train_dls = [train_dl[f'subj0{s}'] for s in args.subj_list]

    num_test = [2770,2770,2113,1985,2770,2113,2770,1985] # maximum possible number of test samples per subj
    test_url = f"{args.data_path}/wds/subj0{args.subj_list[0]}/test/" + "0.tar"
    print("test_url", test_url)
    test_data = wds.WebDataset(test_url,resampled=False,nodesplitter=my_split_by_node)\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test[0], shuffle=False, drop_last=False, pin_memory=True)
    print(f"Loaded test dl for subj{args.subj_list[0]}! num_test={num_test[0]}\n")

 
    # Preload 73k NSD images
    f = h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r')
    images = f['images'][:]
    images = torch.Tensor(images).to("cpu").to(args.data_type)
    print("Loaded all 73k possible NSD images to cpu!", images.shape)
    args.num_voxels_list = num_voxels_list
    return train_dls, test_dl, voxels, images
 
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x

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


class BrainNetwork(nn.Module):
    def __init__(self, out_dim=768, in_dim=15724, h=4096, n_blocks=None, drop=.15, clip_size=768, use_projector=True):
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

def get_models_and_optimizers(args, local_rank, num_devices, device):
    vd_pipe = VersatileDiffusionPipeline.from_pretrained("./versatile-diffusion", torch_dtype=args.data_type, cache_dir=args.cache_dir)
    vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained("./versatile-diffusion", subfolder="scheduler", cache_dir=args.cache_dir)

    clip_model = vd_pipe.image_encoder
    clip_model.to(args.data_type)
    clip_model.eval()
    clip_model.requires_grad_(False)
    clip_model = clip_model.to(device)

    model = MindEyeModule()
    model.ridge = RidgeRegression(args.num_voxels_list, out_features=args.hidden_dim).to(device)
    utils.count_params(model.ridge)
    utils.count_params(model)

    # test shape of model ridge regression output
    b = torch.randn((2,args.num_voxels_list[0])).to(device)
    print(b.shape, model.ridge(b,0).shape)

    model.backbone = BrainNetwork(h=args.hidden_dim, 
                                in_dim=args.hidden_dim, clip_size=args.clip_emb_dim, 
                                out_dim=args.clip_emb_dim*args.clip_seq_dim,
                                n_blocks = args.n_blocks
                                ).to(device)

    utils.count_params(model.backbone)
    utils.count_params(model)
    # test that the model works on some fake data
    b = torch.randn((2,args.hidden_dim)).to(device)
    print("in",b.shape)
    backbone_, clip_ = model.backbone(b)
    print("out",backbone_.shape, clip_.shape)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ] 
    
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)

    # Compute num_epochs to balance the number of gradient updates 
    # args.num_epochs = int(args.num_epochs / (args.num_train / 24958)) 

    if args.lr_scheduler_type == 'linear':
         
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(args.num_epochs*(args.num_train/args.num_devices/args.batch_size))),
            last_epoch=-1
        )
    elif args.lr_scheduler_type == 'cycle':
        total_steps=int(np.floor(args.num_epochs*args.num_iterations_per_epoch)) 

        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/args.num_epochs
        )
 
 
 
    return clip_model, model, optimizer, lr_scheduler 

def get_record(args):
    record = utils.dotdict(
        args = args,
        metrics = utils.dotdict(
            train_losses = defaultdict(float),
            test_losses = defaultdict(float),
            fwd_percent_correct = defaultdict(float),
            bwd_percent_correct = defaultdict(float),
            train_fwd_pct_correct = defaultdict(float),
            train_bwd_pct_correct = defaultdict(float),
            test_fwd_percent_correct = defaultdict(float),
            test_bwd_percent_correct = defaultdict(float),
            avg_test_bwd_pct_correct = defaultdict(float),
            avg_test_fwd_pct_correct = defaultdict(float),
            lrs = defaultdict(float),
            loss_clip_total = defaultdict(float),
            loss_blurry_total = defaultdict(float),
            test_loss_clip_total = defaultdict(float),
            test_loss_blurry_total = defaultdict(float),
        ) ,
        model_state_dict = None,
        optimizer_state_dict = None,
        lr_scheduler = None,
    )
    return record

def train(args, local_rank, clip_model , model,
        optimizer, lr_scheduler,  
        train_dls, test_dl, voxels, images, img_augment, record, device):
    epoch = 0
    losses, test_losses, lrs = [], [], []
    best_test_loss = 1e9
    torch.cuda.empty_cache()

    test_image, test_voxel = None, None
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
    utils.seed_everything(seed=args.seed, cudnn_deterministic=True)
    
    print(f"{args.model_name} starting with epoch {epoch} / {args.num_epochs}")
     
    for epoch in range(epoch,args.num_epochs):
        model.train()

        # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
        voxel_iters = {} # empty dict because diff subjects have differing # of voxels
        image_iters = torch.zeros(args.num_iterations_per_epoch, args.batch_size*len(args.subj_list), 3, 224, 224).float()
        annot_iters = {}
        perm_iters, betas_iters, select_iters = {}, {}, {}
        for s, train_dl in enumerate(train_dls):
            with torch.cuda.amp.autocast(dtype=args.data_type):
                for iter, (behav0, past_behav0, future_behav0, old_behav0) in enumerate(train_dl):
                    image0 = images[behav0[:,0,0].cpu().long()].float()
                    image_iters[iter,s*args.batch_size:s*args.batch_size+args.batch_size] = image0
                    voxel0 = voxels[f'subj0{args.subj_list[s]}'][behav0[:,0,5].cpu().long()]
                    voxel0 = torch.Tensor(voxel0).to(args.data_type)
                    if epoch < int(args.mixup_pct * args.num_epochs):
                        voxel0, perm, betas, select = utils.mixco(voxel0)
                        perm_iters[f"subj0{args.subj_list[s]}_iter{iter}"] = perm
                        betas_iters[f"subj0{args.subj_list[s]}_iter{iter}"] = betas
                        select_iters[f"subj0{args.subj_list[s]}_iter{iter}"] = select

                    voxel_iters[f"subj0{args.subj_list[s]}_iter{iter}"] = voxel0

                    if iter >= args.num_iterations_per_epoch-1:
                        break

        # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
        for train_i in range(args.num_iterations_per_epoch):
            with torch.cuda.amp.autocast(dtype=args.data_type):
                optimizer.zero_grad()
                voxel_list = [voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in args.subj_list]
                image = image_iters[train_i].detach().to(device)

                if args.use_image_aug: 
                    image = img_augment(image)

                clip_target = utils.get_clip_embeddings(clip_model,image)
                assert not torch.any(torch.isnan(clip_target))

                if epoch < int(args.mixup_pct * args.num_epochs):
                    perm_list = [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in args.subj_list]
                    perm = torch.cat(perm_list, dim=0)
                    betas_list = [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in args.subj_list]
                    betas = torch.cat(betas_list, dim=0)
                    select_list = [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in args.subj_list]
                    select = torch.cat(select_list, dim=0)

                voxel_ridge_list = [model.ridge(voxel_list[si],si) for si,s in enumerate(args.subj_list)]
                voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

                clip_voxels, clip_voxels_proj = model.backbone(voxel_ridge)

                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                if epoch < int(args.mixup_pct * args.num_epochs):
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006, 
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(args.mixup_pct*args.num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm, 
                        temp=epoch_temp)

                loss = loss_clip
                labels = torch.arange(len(clip_target_norm)).to(clip_voxels_norm.device) 
                record.metrics.fwd_percent_correct[epoch] += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1) 
                record.metrics.bwd_percent_correct[epoch] += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1) 

                utils.check_loss(loss)
                accelerator.backward(loss)
                optimizer.step()

                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])
        
                record.metrics.loss_clip_total[epoch] += loss_clip.item()
 
                if args.lr_scheduler_type is not None:
                    lr_scheduler.step()

                if args.debug and train_i > 1: break 

        record, test_i, test_losses_this_epoch, test_image, test_voxel = compute_validation_metrics (epoch, args, local_rank,
                test_image, test_voxel,
                clip_model , model, optimizer, lr_scheduler, 
                test_dl, voxels, images, img_augment, record)

        record = compute_metrics (record, epoch, args, local_rank, 
                    train_i, test_i, losses, test_losses_this_epoch, lrs,
                    model, optimizer, lr_scheduler)
    return record, train_i, losses, lrs

def compute_validation_metrics(epoch, args, local_rank,
                test_image, test_voxel,
                clip_model, model, optimizer, lr_scheduler, 
                test_dl, voxels, images, img_augment, record):
    test_losses_this_epoch = []
    model.eval()
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=args.data_type): 
            for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):  
                ## Average same-image repeats ##
                if test_image is None:
                    voxel = voxels[f'subj0{args.subj_list[0]}'][behav[:,0,5].cpu().long()]
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
                assert len(image) == 300
                
                clip_target = utils.get_clip_embeddings(clip_model,image.float())
                voxel_ridge = model.ridge(voxel, 0)

                clip_voxels, clip_voxels_proj = model.backbone(voxel_ridge)
                
                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                loss_clip = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006)

                loss = loss_clip
  
                record.metrics.test_loss_clip_total[epoch] += loss_clip.item() 
                utils.check_loss(loss)
        
                test_losses_this_epoch.append(loss.item())
        
                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_target_norm)).to(clip_voxels_norm.device) 
                record.metrics.test_fwd_percent_correct[epoch] += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1)
                record.metrics.test_bwd_percent_correct[epoch] += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1)
 

    return record, test_i, test_losses_this_epoch, test_image, test_voxel

def compute_metrics(record, epoch, args, local_rank, 
                    train_i, test_i, losses, test_losses_this_epoch, lrs,
                    model, optimizer, lr_scheduler
):
    if local_rank==0:      
        assert (test_i+1) == 1
        record.metrics.train_losses_raw = losses
        record.metrics.train_i = train_i
        record.metrics.test_i = test_i
        record.metrics.train_losses[epoch] = np.mean(losses[-(train_i+1):])
        record.metrics.test_losses[epoch] = np.mean(test_losses_this_epoch)
        record.metrics.lrs[epoch] = lrs[-1]
        # record.metrics.train_num_steps = len(losses)
        # record.metrics.test_num_steps = len(test_losses)
        record.metrics.train_fwd_pct_correct[epoch] = record.metrics.fwd_percent_correct[epoch].item() / (train_i + 1)
        record.metrics.train_bwd_pct_correct[epoch] = record.metrics.bwd_percent_correct[epoch].item() / (train_i + 1)
        record.metrics.avg_test_fwd_pct_correct[epoch] = record.metrics.test_fwd_percent_correct[epoch].item() / (test_i + 1)
        record.metrics.avg_test_bwd_pct_correct[epoch] = record.metrics.test_bwd_percent_correct[epoch].item() / (test_i + 1)
        record.args = args # save args to record
        # Save model checkpoint and reconstruct
        unwrapped_model = accelerator.unwrap_model(model)
        record.model_state_dict = unwrapped_model.state_dict()
        record.optimizer_state_dict = optimizer.state_dict()
        record.lr_scheduler = lr_scheduler.state_dict()
        utils.save_file_pickle (f'{args.outdir}/{args.exp_name}.pkl', record)
        print (
            f"epoch {epoch} / {args.num_epochs} | ", 
            f"train loss {record.metrics.train_losses[epoch]:.4f} | ",
            f"test loss {record.metrics.test_losses[epoch]:.4f} | ",
            f"train fwd {record.metrics.train_fwd_pct_correct[epoch]:.4f} | ",
            f"train bwd {record.metrics.train_bwd_pct_correct[epoch]:.4f} | ",
            f"test fwd {record.metrics.avg_test_fwd_pct_correct[epoch]:.4f} | ",
            f"test bwd {record.metrics.avg_test_bwd_pct_correct[epoch]:.4f} | ",
            flush = True
        )

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
    gc.collect()

    return record

if __name__ == "__main__":
    print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())])
    local_rank = get_local_rank()

    num_devices = setup_gpu_configs()
    
    
    args = get_parser_args(num_devices)
    accelerator = get_deepspeed_accelerator(args, num_devices)
    device, num_workers, world_size, distributed, print, args.data_type = get_training_params(accelerator)  
    

    img_augment = get_img_augmentations(args)
    train_dls, test_dl, voxels, images = get_data_loaders(args)
    clip_model , model, optimizer, lr_scheduler = get_models_and_optimizers(args, local_rank, num_devices, device)

    torch.cuda.empty_cache()
    model, optimizer, train_dls, test_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dls, test_dl, lr_scheduler
    )
    model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)


    record = get_record(args)
    record, train_i, losses, lrs = train(args, local_rank, clip_model, model,
        optimizer, lr_scheduler,  
        train_dls, test_dl, voxels, images, img_augment, record, device)
     
    
    print ("Finished successfully with args", args) 