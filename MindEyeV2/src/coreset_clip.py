import webdataset as wds 
import random
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
import utils
import sys
import argparse
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder # bigG embedder
from diffusers import VersatileDiffusionPipeline, UniPCMultistepScheduler

parser = argparse.ArgumentParser(description="Dataset Coreset config")
parser.add_argument("--normalize_output", 
    action=argparse.BooleanOptionalAction,default=False,
    help="Normalize output") 
args = parser.parse_args()

# if you get an error here, make sure your diffusers package is version 0.13.0!
data_type = torch.float32
cache_dir = "/weka/home-alexnguyen"
device = torch.device("cuda:0")
vd_pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion", torch_dtype=data_type, cache_dir=cache_dir)
vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained("shi-labs/versatile-diffusion", subfolder="scheduler", cache_dir=cache_dir)
# vd_pipe = VersatileDiffusionPipeline.from_pretrained("/weka/proj-fmri/shared/cache/versatile-diffusion")#, torch_dtype=data_type)
# vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained("/weka/proj-fmri/shared/cache/versatile-diffusion", subfolder="scheduler")
# vd_pipe = VersatileDiffusionPipeline.from_pretrained("/weka/home-alexnguyen/models--shi-labs--versatile-diffusion")#, torch_dtype=data_type)
# vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained("/weka/home-alexnguyen/models--shi-labs--versatile-diffusion", subfolder="scheduler")

vd_pipe.to(device)#(torch.device(f"cuda:{local_rank}"))
clip_model = vd_pipe.image_encoder
clip_model.to(data_type)
clip_model.eval()
clip_model.requires_grad_(False)
clip_seq_dim = 257
clip_emb_dim = 768
 

def my_split_by_node(urls): return urls

f = h5py.File(f'/weka/home-alexnguyen/mindeyev2_dataset/coco_images_224_float16.hdf5', 'r')
images = f['images'][:]
images = torch.Tensor(images).to("cpu") 
del f 
print (images.shape)

train_url = "/weka/home-alexnguyen/mindeyev2_dataset/wds/subj01/new_train/{0..39}.tar"
train_data = wds.WebDataset(train_url,resampled=True,nodesplitter=my_split_by_node)\
                        .shuffle(750, initial=2500, rng=random.Random(1))\
                        .decode("torch")\
                       .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
                       .to_tuple(*["__key__", "behav", "past_behav", "future_behav", "olds_behav"]) 
train_dl = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=False, drop_last=True, pin_memory=True) 
keys = set() 
for iter, (key, behav0, past_behav0, future_behav0, old_behav0) in enumerate(train_dl): 
    
#     print (behav0[:,0,0].cpu().long())
#     print (past_behav0[:,0,0].cpu().long())
    [keys.add( i.item() ) for i in behav0[:,0,0].cpu().long()]
#     print ((keys))
    if iter % 10 == 1: print (len(keys))
    if len(keys) >= 9000: break

# get clips using keys
keys = list(keys)
_images = images[keys]
clip_embds = []

with torch.no_grad():
    for batch_ids in range(0, 9000, 300):
        img_batch = _images[batch_ids:batch_ids+300].to(device)
        clip_embds.append(utils.get_clip_outputs(clip_model,img_batch, normalize_output=args.normalize_output))
clip_embds = torch.cat(clip_embds, dim=0)

print ("clip_embds", clip_embds.shape)

# convert to numpy

def make_coreset(clip_embds,seed):
    idxs = []
    clip_ids = []  
    clip_embds = torch.Tensor(clip_embds).to('cuda:0')
    # compute norm of each embd 
    norm_clip_embds = torch.norm(clip_embds, dim=1, keepdim=True) 
    print ("norm_clip_embds", norm_clip_embds)
    pdist = torch.nn.PairwiseDistance(p=2,keepdim=True)
    # fill with infinity
    for i in range(clip_embds.shape[0]):
        if len(idxs) == 0:
            idxs.append(np.random.choice(len(clip_embds), 1).item())
            clip_ids.append(keys[i])
            already_selected_clips = clip_embds[idxs].unsqueeze(0) # get selected clips
            dist = torch.cdist(clip_embds.unsqueeze(0), already_selected_clips).squeeze(0) 
            min_dist = torch.min(dist, dim=1).values
            print (dist.shape, min_dist.shape, min_dist, "chose", len(idxs), "items")
            continue
        argmax = torch.argmax(min_dist)
        assert argmax not in idxs, "Error, already selected"
        print (min_dist[argmax], min_dist.shape )
        idxs.append(argmax.item()) # add to selected
        clip_ids.append(keys[argmax]) # add to selected

        selected_clips = clip_embds[argmax].unsqueeze(0).unsqueeze(1) # get selected clips
        dist = torch.cdist(clip_embds.unsqueeze(0), selected_clips).flatten()
        # print ("d",dist.shape,min_dist.shape)
        min_dist = torch.minimum(dist, min_dist)
     
        if len(clip_ids) in [750, 1500, 2250, 3000, 3750, 4500, 5250, 6000]:
            print (len(idxs))
            np.save(f"./cache/clipembds_coreset_normalize_output_{args.normalize_output}_pruned_ids_N_{int(len(clip_ids))}_seed_{seed}.npy", clip_ids)
         
for seed in range(0, 10):
    make_coreset(clip_embds,seed=seed)
