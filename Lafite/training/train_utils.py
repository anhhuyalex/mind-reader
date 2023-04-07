import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

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
    
## Save file
def save_checkpoint(state, save_dir = "./", filename='checkpoint.pth.tar'):
    torch.save(state, f"{save_dir}/{filename}")
    