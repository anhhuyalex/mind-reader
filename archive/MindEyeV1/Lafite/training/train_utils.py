import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
# see https://github.com/zijin-gu/meshconv-decoding/issues/3
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

def voxel_select(voxels):
    if voxels.ndim == 2:
        return voxels
    choice = torch.rand(1)
    # random combine
    if choice <= 0.5:
        weights = torch.rand(voxels.shape[0], voxels.shape[1])[:,:,None].to(voxels.device)
        return (weights * voxels).sum(1)/weights.sum(1)
    # mean
    if choice <= 0.8:
        return voxels.mean(1)
    # random select
    randints = torch.randint(0, voxels.shape[1], (voxels.shape[0],))
    return voxels[torch.arange(voxels.shape[0]), randints]

def normalize():
    return T.Compose([
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
def full_preprocess(img, mode='bicubic', ratio=0.5, fix_size=224):
    full_size = img.shape[-2]

    if full_size < fix_size:
        pad_1 = torch.randint(0, fix_size-full_size, ())
        pad_2 = torch.randint(0, fix_size-full_size, ())
        m = torch.nn.ConstantPad2d((pad_1, fix_size-full_size-pad_1, pad_2, fix_size-full_size-pad_2), 1.)
        reshaped_img = m(img)
    else:
        if ratio == 1:
            cropped_img = img
        else:
            cut_size = torch.randint(int(ratio*full_size), full_size, ())
            left = torch.randint(0, full_size-cut_size, ())
            top = torch.randint(0, full_size-cut_size, ())
            cropped_img = img[:, :, top:top+cut_size, left:left+cut_size]
        reshaped_img = F.interpolate(cropped_img, (fix_size, fix_size), mode=mode, align_corners=False)
    # print(f'reshaped_img range before {reshaped_img.min()}, {reshaped_img.max()}')
    reshaped_img = (reshaped_img + 1.)*0.5 # range in [0., 1.] now
    # print(f'reshaped_img range after {reshaped_img.min()}, {reshaped_img.max()}')
    # TODO double check this for structure==4 after training
    reshaped_img = normalize()(reshaped_img)
    return  reshaped_img

def contra_loss(temp, mat1, mat2, lam):
    sim = torch.cosine_similarity(mat1.unsqueeze(1), mat2.unsqueeze(0), dim=-1)
    if temp > 0.:
        # sim = torch.exp(sim/temp) # TODO: This implementation is incorrect, it should be sim=sim/temp. change hp
        sim = sim/temp
        # However, this incorrect implementation can reproduce our results with provided hyper-parameters.
        # If you want to use the correct implementation, please manually revise it.
        # The correct implementation should lead to better results, but don't use our provided hyper-parameters, you need to carefully tune lam, temp, itd, itc and other hyper-parameters
        sim1 = torch.diagonal(F.softmax(sim, dim=1))*temp
        sim2 = torch.diagonal(F.softmax(sim, dim=0))*temp
        if 0.<lam < 1.:
            return lam*torch.log(sim1) + (1.-lam)*torch.log(sim2)
        elif lam == 0:
            return torch.log(sim2)
        else:
            return torch.log(sim1)
    else:
        return torch.diagonal(sim)
    
def pairwise_cosine_similarity(A, B, dim=1, eps=1e-8):
    #https://stackoverflow.com/questions/67199317/pytorch-cosine-similarity-nxn-elements
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

def l2norm(x):
    return nn.functional.normalize(x, dim=-1)

def two_way_identification(all_brain_recons, all_images, model, preprocess, device):
    per_correct = []
    l2dist_list = []
    for irecon, recon in enumerate(all_brain_recons):
        with torch.no_grad():   
            if torch.all(recon==0) or torch.all(recon==1):
                print("skip")
                continue
            real = model(preprocess(all_images[irecon].to(device)).unsqueeze(0)).float().reshape(1, -1)
            # fake = model(preprocess(transforms.functional.hflip(all_images[irecon]).unsqueeze(0))).float() 
            fake = model(preprocess(recon).unsqueeze(0)).float().reshape(1, -1)
            #print("real", real.shape)
            #print("fake", fake.shape)
            rand_idx = np.random.randint(len(all_brain_recons))
            while irecon == rand_idx:
                rand_idx = np.random.randint(len(all_brain_recons))
            rand = model(preprocess(all_brain_recons[rand_idx]).unsqueeze(0)).float().reshape(1, -1)

            # l2dist_fake = torch.mean(torch.sqrt((l2norm(real) - l2norm(fake))**2))
            # l2dist_rand = torch.mean(torch.sqrt((l2norm(real) - l2norm(rand))**2))

            # cosine similarity is faster and gives same results
            l2dist_fake = pairwise_cosine_similarity(real,fake).item()
            l2dist_rand = pairwise_cosine_similarity(real,rand).item()

            if l2dist_fake > l2dist_rand:
                per_correct.append(1)
            else:
                per_correct.append(0)
            l2dist_list.append(l2dist_fake)
    return per_correct, l2dist_list

def two_way_identification_clip(all_brain_recons, all_images, clip_extractor, device):
    per_correct = []
    l2dist_list = []
    for irecon, recon in enumerate(all_brain_recons):
        with torch.no_grad():       
            if torch.all(recon==0) or torch.all(recon==1):
                print("skip")
                continue

            real = clip_extractor.embed_image(all_images[irecon].to(device).unsqueeze(0)).float()
            fake = clip_extractor.embed_image(recon.unsqueeze(0)).float()
            rand_idx = np.random.randint(len(all_brain_recons))
            while irecon == rand_idx:
                rand_idx = np.random.randint(len(all_brain_recons))
            rand = clip_extractor.embed_image(all_brain_recons[rand_idx].unsqueeze(0)).float()

            # l2dist_fake = torch.mean(torch.sqrt((l2norm(real) - l2norm(fake))**2))
            # l2dist_rand = torch.mean(torch.sqrt((l2norm(real) - l2norm(rand))**2))

            # cosine similarity is faster and gives same results
            l2dist_fake = pairwise_cosine_similarity(real,fake).item()
            l2dist_rand = pairwise_cosine_similarity(real,rand).item()

            if l2dist_fake > l2dist_rand:
                per_correct.append(1)
            else:
                per_correct.append(0)
            l2dist_list.append(l2dist_fake)
            
    return per_correct, l2dist_list

def compute_ssim_metric(all_brain_recons, all_images):
    ssim_list = []
    for irecon, recon in enumerate(all_brain_recons):
        # convert image to grayscale with rgb2grey
        img_gray = rgb2gray(all_images[irecon].permute((1,2,0)))
        recon_gray = rgb2gray(recon.permute((1,2,0)).detach().cpu().numpy())

        ssim_score = ssim(img_gray, recon_gray, multichannel=True, gaussian_weights=True, 
                              sigma=1.5, use_sample_covariance=False, data_range=1.0)
        ssim_list.append(ssim_score)
    return ssim_list

def compute_pixcorr_metric(all_brain_recons, all_images):
    pix_dist_list = []
    for irecon, recon in enumerate(all_brain_recons):
        #recon = recon.permute((1,2,0)).detach().cpu().numpy().flatten()
        recon = recon.detach().cpu().numpy().flatten()
        real = all_images[irecon].numpy().flatten()
        pix_dist =  np.corrcoef(recon,real )[0][1]
        pix_dist_list.append(pix_dist)
    return pix_dist_list
## ResNET
def resnet50(layer=2, pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    emb = {1: 256, 2: 512, 3: 1024, 4: 2048}
    if pretrained:
        misc.load_trained_model('/home/sikun/vicreg/resnet50.pth', model)
    return model, emb[layer]

## Save file
def save_checkpoint(state, save_dir = "./", filename='checkpoint.pth.tar'):
    torch.save(state, f"{save_dir}/{filename}")
    
    