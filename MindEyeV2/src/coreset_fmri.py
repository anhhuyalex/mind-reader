import webdataset as wds 
import random
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

def my_split_by_node(urls): return urls

f = h5py.File(f'/weka/home-alexnguyen/mindeyev2_dataset/betas_all_subj01_fp32.hdf5', 'r')
voxels = f['betas'][:]
del f 
print (voxels.shape)
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
    [keys.add( i.item() ) for i in behav0[:,0,5].cpu().long()]
#     print ((keys))
    if iter % 10 == 1: print (len(keys))
    if len(keys) >= 27000: break

# get voxels using keys
keys = list(keys)
_voxels = voxels[keys]
# convert to numpy
print (_voxels.shape)

def make_coreset(_voxels,seed):
    idxs = []
    voxel_ids = []  
    _voxels = torch.Tensor(_voxels).to('cuda:0')
    pdist = torch.nn.PairwiseDistance(p=2,keepdim=True)
    # fill with infinity
    
    for i in range(len(_voxels)):
        if len(idxs) == 0:
            idxs.append(np.random.choice(len(_voxels), 1).item())
            voxel_ids.append(keys[idxs[0]])
            already_selected_voxels = _voxels[idxs].unsqueeze(0) # get selected voxels 
            dist = torch.cdist(_voxels.unsqueeze(0), already_selected_voxels).squeeze(0) 
            min_dist = torch.min(dist, dim=1).values
            print (dist.shape, min_dist.shape, min_dist, "chose", len(idxs), "items")
            continue 
        argmax = torch.argmax(min_dist)
        assert argmax not in idxs, "Error, already selected"
        # print (argmax, min_dist.shape, min_dist[argmax])
        idxs.append(argmax.item()) # add to selected 
        voxel_ids.append(keys[argmax]) # add to selected
        
        selected_voxels = _voxels[argmax].unsqueeze(0).unsqueeze(1) # get selected voxels 
        dist = torch.cdist(_voxels.unsqueeze(0), selected_voxels).flatten()
        # print ("d",dist.shape,min_dist.shape)
        min_dist = torch.minimum(dist, min_dist)
        if len(voxel_ids) in [750, 1500, 2250, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000]:
            print (len(idxs))
            np.save(f"./cache/fmriraw_coreset_pruned_ids_N_{int(len(voxel_ids))}_seed_{seed}.npy", voxel_ids)
        
make_coreset(_voxels,seed=1)
make_coreset(_voxels,seed=2)
make_coreset(_voxels,seed=3)
make_coreset(_voxels,seed=4)
