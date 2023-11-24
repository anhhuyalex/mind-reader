import os
import sys
import json
import argparse
import numpy as np
import time
import random
import h5py
from tqdm import tqdm
from functools import partial
from collections import defaultdict

import webdataset as wds
import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

import utils
from accelerate import Accelerator, DeepSpeedPlugin
# from diffusers import AutoencoderKL
from diffusers.models.vae import Decoder

from models import Clipper

def get_local_rank():
    local_rank = os.getenv('RANK')
    if local_rank is None: 
        local_rank = 0
    else:
        local_rank = int(local_rank)
    return local_rank

def setup_gpu_configs():
    torch.backends.cuda.matmul.allow_tf32 = True

    num_devices = torch.cuda.device_count()

    # get num_devices
    if num_devices==0: 
        num_devices = 1
    if num_devices <= 1 and utils.is_interactive():
        # can emulate a distributed environment for deepspeed to work in jupyter notebook
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(np.random.randint(10000)+9000)
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["GLOBAL_BATCH_SIZE"] = "128" # set this to your batch size!
        global_batch_size = os.environ["GLOBAL_BATCH_SIZE"]

    # set GLOBAL_BATCH_SIZE if not already set
    try: 
        print("GLOBAL_BATCH_SIZE", os.environ["GLOBAL_BATCH_SIZE"])
    except:
        os.environ["GLOBAL_BATCH_SIZE"] = "128" # set this to your batch size!
    return num_devices

def get_deepspeed_accelerator(args, num_devices):
    # alter the deepspeed config according to your global and local batch size
    if local_rank == 0:
        with open('deepspeed_config_stage2.json', 'r') as file:
            config = json.load(file)
        config['train_batch_size'] = int(os.environ["GLOBAL_BATCH_SIZE"])
        config['train_micro_batch_size_per_gpu'] = args.batch_size
        config['bf16'] = {'enabled': False}
        config['fp16'] = {'enabled': True}
        with open('deepspeed_config_stage2.json', 'w') as file:
            json.dump(config, file)
    else:
        # give some time for the local_rank=0 gpu to prep new deepspeed config file
        time.sleep(10)

    
    deepspeed_plugin = DeepSpeedPlugin("deepspeed_config_stage2.json")
    accelerator = Accelerator(split_batches=False, deepspeed_plugin=deepspeed_plugin)
    print("\033[94m" + f"INFO: Using deepspeed_plugin {deepspeed_plugin} accelerator {accelerator}" + "\033[0m") 
    args.num_devices = num_devices
    return accelerator

def get_training_params(accelerator):
    print = accelerator.print
    print("PID of this process =",os.getpid())
    device = accelerator.device
    print("device:",device)
    num_workers = args.num_devices
    print(accelerator.state)
    world_size = accelerator.state.num_processes
    distributed = not accelerator.state.distributed_type == 'NO'
    print("distributed =",distributed, "num_devices =", args.num_devices, "local rank =", local_rank, "world size =", world_size)
    
    # set data_type to match your mixed precision (automatically set based on deepspeed config)
    if accelerator.mixed_precision == "bf16":
        data_type = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        data_type = torch.float16
    else:
        data_type = torch.float32

    
    return device, num_workers, world_size, distributed, print, data_type

def optionally_specify_jupyter_args():
    if utils.is_interactive():
        # Example use
        jupyter_args = f"--data_path=/fsx/proj-fmri/shared/mindeyev2_dataset \
                        --model_name=test \
                        --subj=1 --batch_size={global_batch_size} \
                        --no-blurry_recon --no-depth_recon \
                        --clip_scale=1. --blur_scale=100. --depth_scale=100. \
                        --max_lr=3e-4 --mixup_pct=.66 --num_epochs=12 --ckpt_interval=999 --no-use_image_aug --no-ckpt_saving"

        jupyter_args = jupyter_args.split()
        print(jupyter_args)
        
        from IPython.display import clear_output # function to clear print outputs in cell
        get_ipython().run_line_magic('load_ext', 'autoreload')
        # this allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
        get_ipython().run_line_magic('autoreload', '2')

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
        "--subj_sample_list", type=str, default="cache/subj1_sample_list.txt",
        help="Path to where the list of samples for the subject is stored, to be used for data-pruning",
    )
    parser.add_argument(
        "--subj",type=int, default=1, choices=[1,2,5,7],
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
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
    parser.add_argument(
        "--n_samples_save",type=int,default=0,choices=[0,1],
        help="Number of reconstructions for monitoring progress, 0 will speed up training",
    )
    parser.add_argument("--file_prefix", type=str, default="",
        action = "store",
        help="prefix for ckpt files")
    parser.add_argument("--debug", '-d',
        action='store_true', 
        help = "debug mode", 
        default=False)
    parser.add_argument("--num_samples_to_prune", type=int, 
        default=0,
        action = "store",
        help="number of sessions to train on")

    if utils.is_interactive():
        args = parser.parse_args(jupyter_args)
    else:
        args = parser.parse_args()

    
    args.batch_size = int(args.batch_size / num_devices)
    print("global batch_size", args.batch_size )

    # get output directory
    args.outdir = os.path.abspath(f'./train_logs')
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir,exist_ok=True)

    # get train parameters
    if args.subj==1:
        # args.num_train = 24958
        args.num_test = 2770
    args.test_batch_size = args.num_test # test batch size is the same as the number of test samples

    # get data paths
    args.train_url = f"{args.data_path}/wds/subj0{args.subj}/train/" + "{0..36}.tar"
    args.test_url = f"{args.data_path}/wds/subj0{args.subj}/test/" + "0.tar"

    # sesses = np.random.choice(np.arange(37), size=args.num_sessions, replace=False)
    # sess_str = ",".join([str(s) for s in sesses]) 
    # print ("sess_str", sess_str)

    # if args.num_sessions > 1:
    #     args.train_url = f"{args.data_path}/wds/subj0{args.subj}/train/" + "{{{s}}}.tar".format(s=sess_str)
    #     args.test_url = f"{args.data_path}/wds/subj0{args.subj}/test/" + "0.tar"
    # else: 
    #     args.train_url = f"{args.data_path}/wds/subj0{args.subj}/train/" + f"{sess_str}.tar"
    #     args.test_url = f"{args.data_path}/wds/subj0{args.subj}/test/" + "0.tar"

    

    # if args.file_prefix=="sessions_1_2" :
    #     args.train_url = f"{args.data_path}/wds/subj0{args.subj}/train/" + "{0..1}.tar"
    #     args.test_url = f"{args.data_path}/wds/subj0{args.subj}/test/" + "0.tar"

    # get model params
    args.clip_seq_dim = 257
    args.clip_emb_dim = 768
    args.hidden_dim = 4096

    # save params
    args.exp_name = f"{args.file_prefix}_{time.time()}"

    # seed
    args.seed = args.seed if args.seed else int(time.time())
    return args

def get_img_augmentations(args):
    img_augment = None 
    if args.use_image_aug:
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

class H5Dataset(torch.utils.data.Dataset):
    """
    Dataset for loading h5 files
    """ 

    def __init__(self, h5_path, transform=None):
        """
        Args:
            h5_path (string): Path to the h5 file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(H5Dataset, self).__init__()
        self.h5_path = h5_path
        self.transform = transform

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as f:
            return f['images'].shape[0]

    def get_image(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            image = f['images'][idx]
            image = torch.Tensor(image).to("cpu").half()
            if self.transform:
                image = self.transform(image)
        return image

    def __getitem__(self, idxs):
        if isinstance(idxs, torch.Tensor):
            images = torch.stack([self.get_image(int(idx)) for idx in idxs], dim=0)
            return images 
        else:
            return self.get_image(idxs) 
        

def get_data_loaders(args, data_type):
    def my_split_by_node(urls): return urls

    # read from args.subj_sample_list to get list of samples to use for training
    with open(args.subj_sample_list, 'r') as f:
        args.subj_sample_list = f.read().splitlines()
        samples_to_prune = np.random.choice(args.subj_sample_list, size=args.num_samples_to_prune, replace=False)
        args.subj_sample_list = [s for s in args.subj_sample_list if s not in samples_to_prune]
        print ("samples_to_prune", samples_to_prune[:100])
        # args.subj_sample_list = np.random.choice(f.read().splitlines(), size=args.num_samples, replace=False)
    # write args.subj_sample_list to keylist.txt
    #with open('cache/subj1_sample_list_len_6761_seed_42.txt', 'w') as f:
    #    f.write("\n".join(args.subj_sample_list))
    print (np.logspace(0, np.log10(6761), 37).astype(int))
   
    def filtered_samples(sample):
        return sample["__key__"] in args.subj_sample_list
    train_data = wds.WebDataset(args.train_url,resampled=False,nodesplitter=my_split_by_node)\
                    .select(filtered_samples) \
                    .shuffle(750, initial=1500, rng=random.Random(42))\
                    .decode("torch")\
                    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True)
    
    # compute how many training samples 
    num_train = 0
    for train_i, (behav, past_behav, future_behav, old_behav) in enumerate(train_dl):
        num_train += len(behav)
    args.num_train = num_train
    print ("num_train", num_train)


    test_data = wds.WebDataset(args.test_url,resampled=False,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=1500, rng=random.Random(42))\
                        .decode("torch")\
                        .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                        .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, drop_last=False, pin_memory=True)

    f = h5py.File(f'{args.data_path}/betas_all_subj0{args.subj}.hdf5', 'r')
    voxels = f['betas'][:]
    print(f"subj0{args.subj} betas loaded into memory")
    voxels = torch.Tensor(voxels).to("cpu").to(data_type)
    if args.subj==1:
        voxels = torch.hstack((voxels, torch.zeros((len(voxels), 5))))
    print("voxels", voxels.shape)
    num_voxels = voxels.shape[-1]

    f = h5py.File(f'{args.data_path}/coco_images_224_float16.hdf5', 'r')
    images = f['images'][:]
    images = torch.Tensor(images).to("cpu").to(data_type) 
    return train_dl, test_dl, voxels, images

class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x

class RidgeRegression(torch.nn.Module):
    # make sure to add weight_decay when initializing optimizer
    def __init__(self, input_size, out_features): 
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linear = torch.nn.Linear(input_size, out_features)
    def forward(self, x):
        return self.linear(x)

class BrainNetwork(nn.Module):
    def __init__(self, out_dim=768, in_dim=15724, clip_size=768, h=4096, n_blocks=4, norm_type='ln', act_first=False, drop=.15, blurry_dim=16,
        blurry_recon = False, depth_recon = False):
        super().__init__()
        self.blurry_dim = blurry_dim
        norm_func = partial(nn.BatchNorm1d, num_features=h) if norm_type == 'bn' else partial(nn.LayerNorm, normalized_shape=h)
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == 'bn' else nn.GELU
        act_and_norm = (act_fn, norm_func) if act_first else (norm_func, act_fn)
        self.lin0 = nn.Linear(in_dim, h)
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h),
                *[item() for item in act_and_norm],
                nn.Dropout(drop)
            ) for _ in range(n_blocks)
        ])

        if blurry_recon:
            # self.blin1 = nn.Sequential(
            #     nn.Linear(out_dim, 4096, bias=True),
            #     nn.LayerNorm(4096),
            #     nn.GELU(),
            #     nn.Linear(4096, 4096))
            self.blin1 = nn.Linear(h, 4096)
            self.bgroupnorm = nn.GroupNorm(1, 256)
            self.bupsampler = Decoder(
                in_channels=256,
                out_channels=128,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[32, 64, 128],
                layers_per_block=1,
            )

        if depth_recon:
            # self.dlin1 = nn.Sequential(
            #         nn.Linear(h, midas_emb_size),
            #         nn.Sigmoid(),
            #     )
            self.dlin1 = nn.Linear(h, 4096)
            self.dgroupnorm = nn.GroupNorm(1, 256)
            self.dupsampler = Decoder(
                in_channels=256,
                out_channels=1,#128,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[32, 64, 128, 256],
                layers_per_block=1,
            )
        
        self.n_blocks = n_blocks
        self.clip_size = clip_size
        self.clin1 = nn.Linear(h, out_dim, bias=True)
        self.blurry_recon = blurry_recon
        self.depth_recon = depth_recon

        # low-rank matrices
        # self.rank = 1000
        # self.U = nn.Parameter(torch.randn(self.rank, out_dim))
        # self.V = nn.Parameter(torch.randn(h, self.rank))
        
        self.clip_proj = nn.Sequential(
            nn.LayerNorm(clip_size),
            nn.GELU(),
            nn.Linear(clip_size, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, clip_size)
        )
        
    def forward(self, x):
        b, d = torch.Tensor([0.]), torch.Tensor([0.])
        data_type = x.dtype
        x = self.lin0(x)
        residual = x
        for res_block in range(self.n_blocks):
            x = self.mlp[res_block](x)
            x += residual
            residual = x
        x = x.reshape(len(x), -1)

        # linear mapping to out_dim
        c = self.clin1(x)

        # low rank linear to out dim cuts # params by nearly half compared to full linear mapping
        # c = x @ (self.V/100) @ (self.U/100)

        c = self.clip_proj(c.reshape(len(c), -1, self.clip_size))
        if self.blurry_recon:
            b = self.blin1(x)
            b = b.reshape(len(b), 256, 4, 4)
            b = self.bgroupnorm(b)
            b = self.bupsampler(b)
            
        if self.depth_recon:
            d = self.dlin1(x)#.reshape(len(x), 1, 32, 32)

            d = d.reshape(len(d), 256, 4, 4)
            d = self.dgroupnorm(d)
            d = self.dupsampler(d)
            
        return c, b, d

def get_models_and_optimizers(args, local_rank, num_devices):
    clip_model  = Clipper("ViT-L/14", device=torch.device(f"cuda:{local_rank}"), hidden_state=True, norm_embs=True)
    
    model = MindEyeModule()
    model.ridge = RidgeRegression(voxels.shape[1], out_features=args.hidden_dim)
    utils.count_params(model.ridge)
    utils.count_params(model)

    # test shape of model ridge regression output
    b = torch.randn((2,1,voxels.shape[1]))
    print(b.shape, model.ridge(b).shape)

    model.backbone = BrainNetwork(h=args.hidden_dim, 
                    in_dim=args.hidden_dim, clip_size=args.clip_emb_dim, 
                    out_dim=args.clip_emb_dim*args.clip_seq_dim) 

    utils.count_params(model.backbone)
    utils.count_params(model)

    # test shape of model backbone output
    b = torch.randn((2,args.hidden_dim))
    print(b.shape)
    clip_, blur_, depth_  = model.backbone(b)
    print(clip_.shape, blur_.shape, depth_.shape)


   
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)

    # Compute num_epochs to balance the number of gradient updates 
    args.num_epochs = int(args.num_epochs / (args.num_train / 24958)) 

    if args.lr_scheduler_type == 'linear':
         
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=int(np.floor(args.num_epochs*(args.num_train/args.num_devices/args.batch_size))),
            last_epoch=-1
        )
    elif args.lr_scheduler_type == 'cycle':
        total_steps = int(np.floor(args.num_epochs*(args.num_train/args.num_devices/args.batch_size))) 
        
        print("total_steps", total_steps)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/args.num_epochs
        )
 

    mse = nn.MSELoss()
    l1 = nn.L1Loss()


    # using the same preprocessing as was used in MindEye + BrainDiffuser
    pixcorr_preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    def pixcorr(images,brains):
        # Flatten images while keeping the batch dimension
        all_images_flattened = pixcorr_preprocess(images).reshape(len(images), -1)
        all_brain_recons_flattened = pixcorr_preprocess(brains).view(len(brains), -1)
        corrmean = torch.diag(utils.batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)).mean()
        return corrmean
    return clip_model , model, optimizer, lr_scheduler, mse, l1, pixcorr

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

            train_blurry_pixcorr_total = defaultdict(float),
            test_blurry_pixcorr_total = defaultdict(float), # needs >.456 to beat low-level subj01 results in mindeye v1
        ) 
    )
    return record

def train(args, data_type, local_rank, clip_model , model,
        optimizer, lr_scheduler, mse, l1, pixcorr,
        train_dl, test_dl, voxels, images, img_augment, record):
    epoch = 0
    losses, lrs = [], [] 
    test_image, test_voxel = None, None
    best_test_loss = 1e9
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
 
    print(f"{args.model_name} starting with epoch {epoch} / {args.num_epochs}")
    test_image, test_voxel = None, None
    
    for epoch in range(epoch,args.num_epochs):
        model.train()
         
        batches_proc = 0
        for train_i, (behav, past_behav, future_behav, old_behav) in enumerate(train_dl):
             
            # behav: behavioral data (image index, run, trial, time, voxel)
            with torch.cuda.amp.autocast():
                optimizer.zero_grad()
                
                voxel = voxels[behav[:,0,5].cpu().long()].to(device)
            
                image = images[behav[:,0,0].cpu().long()].to(device).float()

                if args.use_image_aug: image = img_augment(image)
                 
                clip_target = clip_model.embed_image(image)

                assert not torch.any(torch.isnan(clip_target))

                if epoch < int(args.mixup_pct * args.num_epochs):
                    voxel, perm, betas, select = utils.mixco(voxel)

                voxel_ridge = model.ridge(voxel)
    
                clip_voxels, blurry_image_enc_, depth_image_enc_ = model.backbone(voxel_ridge)
                
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
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
 
                loss_clip *= args.clip_scale
                loss = loss_clip
                
                utils.check_loss(loss)

                accelerator.backward(loss)
                optimizer.step()
        
                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])
        
                record.metrics.loss_clip_total[epoch] += loss_clip.item()
                
                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_target_norm)).to(clip_voxels_norm.device) 
                record.metrics.fwd_percent_correct[epoch] += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1) 
                record.metrics.bwd_percent_correct[epoch] += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1) 

             
    
                if args.lr_scheduler_type is not None:
                    lr_scheduler.step()

                if args.debug and train_i > 1: break 

        record, test_i, test_losses_this_epoch, test_image, test_voxel = compute_validation_metrics (epoch, args, data_type, local_rank,
                test_image, test_voxel,
                clip_model , model, optimizer, lr_scheduler, 
                mse, pixcorr,
                train_dl, test_dl, voxels, images, img_augment, record)

        record = compute_metrics (record, epoch, args, local_rank, train_i, test_i, losses, test_losses_this_epoch, lrs)
    return record, train_i, losses, lrs

def compute_validation_metrics (epoch, args, data_type, local_rank,
            test_image, test_voxel,
            clip_model , model, optimizer, lr_scheduler, 
            mse, pixcorr,
            train_dl, test_dl, voxels, images, img_augment, record):
    test_losses_this_epoch = []
    model.eval()
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):  
                # all test samples should be loaded per batch such that test_i should never exceed 0
                assert len(behav) == args.num_test

                ## Average same-image repeats ##
                if test_image is None:
                    voxel = voxels[behav[:,0,5].cpu().long()]
                    image = behav[:,0,0].cpu().long()
                    # image = images[behav[:,0,0].cpu().long()].to(device)
                    unique_image, sort_indices = torch.unique(image, return_inverse=True)
                    print ("image.size()", image.size(), unique_image.size())
                    for im in unique_image:
                        locs = torch.where(im == image)[0]
                        if test_image is None:
                            test_image = images[im][None]
                            # test_image = images[[im]]
                                
                            test_voxel = torch.mean(voxel[locs],axis=0)[None]
                        else:
                            # test_image = torch.vstack((test_image, images[[im]] ))
                            test_image = torch.vstack((test_image, images[im][None]))
                            test_voxel = torch.vstack((test_voxel, torch.mean(voxel[locs],axis=0)[None]))
                            

                # random sample of 300
                random_indices = torch.randperm(len(test_voxel))[:300]
                voxel = test_voxel[random_indices].to(device)
                image = test_image[random_indices].to(device)
                assert len(image) == 300
                
                clip_target = clip_model.embed_image(image.float())
                
                voxel_ridge = model.ridge(voxel)
                
                clip_voxels, blurry_image_enc_, depth_image_enc_ = model.backbone(voxel_ridge)
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
        
                loss_clip = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006)
                
                loss_clip = loss_clip * args.clip_scale
                loss = loss_clip
  
                record.metrics.test_loss_clip_total[epoch] += loss_clip.item() 
                utils.check_loss(loss)
        
                test_losses_this_epoch.append(loss.item())
        
                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_target_norm)).to(clip_voxels_norm.device) 
                record.metrics.test_fwd_percent_correct[epoch] += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1)
                record.metrics.test_bwd_percent_correct[epoch] += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1)
 

    return record, test_i, test_losses_this_epoch, test_image, test_voxel

def compute_metrics(record, epoch, args, local_rank, train_i, test_i, losses, test_losses_this_epoch, lrs):
    if local_rank==0:      
        if utils.is_interactive():
            # clear_output(wait=True)
            print("---")

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
    
    
    optionally_specify_jupyter_args()
    args = get_parser_args(num_devices)
    utils.seed_everything(seed=args.seed, cudnn_deterministic=True)
    accelerator = get_deepspeed_accelerator(args, num_devices)
    device, num_workers, world_size, distributed, print, data_type = get_training_params(accelerator)  

    img_augment = get_img_augmentations(args)
    train_dl, test_dl, voxels, images = get_data_loaders(args, data_type)
    clip_model , model, optimizer, lr_scheduler, mse, l1, pixcorr = get_models_and_optimizers(args, local_rank, num_devices)

    torch.cuda.empty_cache()
    model, optimizer, train_dl, test_dl, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dl, test_dl, lr_scheduler
    )

    record = get_record(args)
    record, train_i, losses, lrs = train(args, data_type, local_rank, 
            clip_model , model, optimizer, lr_scheduler, 
            mse, l1, pixcorr,
            train_dl, test_dl, voxels, images, img_augment, record)
    
    
    print ("Finished successfully !") 