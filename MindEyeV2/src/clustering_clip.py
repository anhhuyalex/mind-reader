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
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser(description="Dataset Coreset config")
parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters")
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

# Calculate dintra and dinter for each cluster. 
# dintra is computed as the average cosine distance between the items of a cluster and its centroid
# dinter is computed for each cluster j as the average cosine distance between a cluster centroid and its l nearest
# neighbor centroids
def compute_dintra_dinter(kmeans, clip_embds, n_neighbors = 20, n_clusters = 100): 
    """
    kmeans: kmeans object  
    n_neighbors: number of neighbors to use for computing dinter
    n_clusters: number of clusters
    """
    d_intra = np.zeros(n_clusters)
    d_inter = np.zeros(n_clusters)
    clip_embds = clip_embds.cpu().detach().numpy() 
    for cluster_id in range(n_clusters):
        cluster = clip_embds[kmeans.labels_ == cluster_id]
        # compute distance from each item in cluster to cluster_centers_[cluster_id]
        a_dot_b = np.dot(cluster, np.expand_dims(kmeans.cluster_centers_[cluster_id], axis=1))
        a_norm = np.linalg.norm(cluster, axis=1) 
        b_norm = np.linalg.norm(kmeans.cluster_centers_[cluster_id])
        # print (a_dot_b.shape, a_norm.shape, b_norm, cluster.shape, np.expand_dims(kmeans.cluster_centers_[cluster_id], axis=1).shape)
        d_intra[cluster_id] = 1-(a_dot_b.squeeze(1) / (a_norm * b_norm)).mean()

        # compute distance from this cluster to all other clusters
        a_dot_b = np.dot(np.expand_dims(kmeans.cluster_centers_[cluster_id], axis=0), kmeans.cluster_centers_.T)
        a_norm = np.linalg.norm(np.expand_dims(kmeans.cluster_centers_[cluster_id], axis=0), axis=1)
        b_norm = np.linalg.norm(kmeans.cluster_centers_, axis=1)
        # print (kmeans.cluster_centers_[cluster_id].shape, kmeans.cluster_centers_.T.shape, a_dot_b.shape, a_norm.shape, b_norm.shape)
        distance_to_other_clusters = 1-(a_dot_b / (a_norm * b_norm))  
        d_inter[cluster_id] = np.sort(distance_to_other_clusters)[0,1:(n_neighbors+1)].mean() # 0 is the distance to itself

    return d_intra, d_inter

def pruning(kmeans, clip_embds, d_intra, d_inter, n_clusters = 100, tau = 0.1,  N = 27000): 
    """
    kmeans: kmeans object  
    d_intra: dintra for each cluster
    d_inter: dinter for each cluster
    n_clusters: number of clusters
    tau: temperature for softmax
    N: target dataset size
    """
    # print ("distance_to_other_clusters", distance_to_other_clusters, "distance_closest_neighbors", distance_closest_neighbors) 
    # Calculate the number of examples per cluster Nj. 
    Cj = d_intra * d_inter
    # softmax
    Pj = np.exp(Cj/tau) / np.exp(Cj/tau).sum()
    Nj = Pj * N 

    # solve quadratic program to fix Nj
    probs = torch.softmax(torch.tensor(Cj/tau),dim=0)
    # print(probs,probs.shape,Pj)
    P = np.eye(n_clusters)
    q = - Nj
    A = np.array([1.0] * n_clusters)
    b = np.array([N])
    # Define the lower and upper bounds
    min_samples = 1
    num_items_in_each_cluster = np.bincount(kmeans.labels_)
    bounds = np.array([ ( min_samples, num_items_in_each_cluster[i] ) for i in range(n_clusters) ])
    # print (bounds.shape,bounds[:,[0]].shape,bounds[:,[1]].shape,P.shape, q.shape)
    # print ("P, q,A,b,bounds",P.shape, q.shape, A.shape, b.shape, bounds.shape)
    X = solve_qp(P=P, q=q, A=A, b=b, lb=bounds[:,0].reshape(1,n_clusters), ub=bounds[:,1].reshape(1,n_clusters), solver="osqp")
    # print(X)
    Nj = np.rint(X).astype(int)
    # print(Nj)
    pruned_ids = []
    clip_embds = clip_embds.cpu().detach().numpy()
    for cluster_id in range(n_clusters):
        
        cluster_ids = np.where( kmeans.labels_ == cluster_id)[0]
        
        cluster = clip_embds[cluster_ids]
        nsamples_cluster = Nj[cluster_id]
        # compute cosine similarity of cluster to cluster_centers_[cluster_id]
        a_dot_b = np.dot(cluster, np.expand_dims(kmeans.cluster_centers_[cluster_id], axis=1))
        a_norm = np.linalg.norm(cluster, axis=1)
        b_norm = np.linalg.norm(kmeans.cluster_centers_[cluster_id])
        distance_to_centroid = 1-(a_dot_b.squeeze(1) / (a_norm * b_norm))
#         print("distance_to_centroid",distance_to_centroid)
        distanceid_to_centroid = np.argsort(distance_to_centroid)[-nsamples_cluster:]
        print("near",np.sort(distance_to_centroid)[:10], "far", distance_to_centroid[distanceid_to_centroid])
        pruned_ids.extend([keys[i] for i in cluster_ids[distanceid_to_centroid]]) 
        
        # select nsamples_cluster from cluster that are furthest away from cluster centroid
        # print (cluster.shape, distance_to_centroid.shape)

    return pruned_ids, Nj, Pj
def get_d_intra_d_inter(seed, clip_embds, n_clusters = 100):
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto",verbose=0).fit(clip_embds.cpu().detach().numpy())

    d_intra, d_inter = compute_dintra_dinter(kmeans, clip_embds, n_neighbors = 20, n_clusters = n_clusters)
    return d_intra, d_inter, kmeans 

for seed in range(0, 10):
    d_intra, d_inter, kmeans = get_d_intra_d_inter(seed, clip_embds, n_clusters = args.n_clusters)
    
    for N in [750, 1500, 2250, 3000, 3750, 4500, 5250, 6000]:
        pruned_ids, _, _ = pruning(kmeans, clip_embds, d_intra, d_inter, N=N, n_clusters = args.n_clusters)
        np.save(f"./cache/clipembds_kmeans_pruned_ids_nclusters_{args.n_clusters}_normalized_{args.normalize_output}_N_{int(N)}_seed_{seed}.npy", pruned_ids)