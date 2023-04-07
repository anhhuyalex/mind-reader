import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

from Lafite import dnnlib
from Lafite import training
# from torch_utils import misc
# from torch_utils import training_stats
from Lafite import torch_utils
# from torch_utils.ops import conv2d_gradfix
# from torch_utils.ops import grid_sample_gradfix
import clip

from .. import legacy
from .. import metrics
from . import networks
from . import train_utils

import time
import webdataset as wds
import math
import tqdm
import h5py
import io
import pickle
import copy
    
def load_file_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "torch_utils.persistence":
            renamed_module = "Lafite.torch_utils.persistence"

        return super(RenameUnpickler, self).find_class(renamed_module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

def load_file_pickle(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
    
class BaseTrainer:
    def __init__(self, data_params = dict(),
                       train_params = dict(),
                       model_optim_params = dict(),
                       save_params = dict()):
        
        self.data_params = data_params
        self.train_params = train_params
        self.model_optim_params = model_optim_params
        self.save_params = save_params
        self.setup() 
        
    def setup(self):
        # Initialize.
        start_time = time.time()
        self.device = torch.device('cuda', self.train_params.rank)
        
        # random seed
        np.random.seed(self.train_params.random_seed * self.train_params.num_gpus + self.train_params.rank)
        torch.manual_seed(self.train_params.random_seed * self.train_params.num_gpus + self.train_params.rank)
        torch.backends.cudnn.benchmark = self.train_params.cudnn_benchmark    # Improves training speed.
        torch.backends.cuda.matmul.allow_tf32 = self.train_params.allow_tf32  # Allow PyTorch to internally use tf32 for matmul
        torch.backends.cudnn.allow_tf32 = self.train_params.allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
        torch_utils.ops.conv2d_gradfix.enabled = True                       # Improves training speed.
        torch_utils.ops.grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.


        self.build_data_loader()
        self.build_model_optimizer()
        self.initialize_record()
        
    def save_record(self):
        # Save record
        fname = f"{self.save_params['save_dir']}/{self.save_params['exp_name']}"
        save_file_pickle(fname, self.record)
        
class MindReader_Trainer(BaseTrainer):
    def initialize_record(self):
        self.record = dict(
            metrics = dict(
                train_loss_prog = [],
                train_acc_prog = [],
                test_loss_prog = [],
                test_acc_prog = [],
                test_loss = 0.0,
                test_accuracy = 0.0
            ),    
            data = dict(
                inputs = None,
                labels = None
            ),
            print_interval = 5000 // self.train_params['batch_size'],
            data_params = self.data_params,
            train_params = self.train_params,
            model_optim_params = self.model_optim_params,
            save_params = self.save_params,
            model = None,
            success = False
        )
        
    def build_data_loader(self):
        # Load training set.
        num_samples = 24983
        num_workers = self.train_params.num_gpus
        global_batch_size = self.train_params.batch_size * self.train_params.num_gpus
        print("global_batch_size",global_batch_size)
        num_batches = math.floor(num_samples / global_batch_size)
        num_worker_batches = math.floor(num_batches / num_workers)
        

        if self.train_params.rank == 0:
            print('Loading training set...')
        
        train_data = wds.DataPipeline([wds.ResampledShards(self.data_params.train_url),
                    wds.tarfile_to_samples(),
                    wds.shuffle(500,initial=500),
                    wds.decode("torch"),
                    #wds.rename(images="jpg;png", voxels="nsdgeneral.npy", embs="sgxl_emb.npy", trial="trial.npy"),
                    wds.rename(images="jpg;png", voxels="nsdgeneral.npy", trial="trial.npy"),
                    wds.to_tuple("voxels", 'images', 'trial'),
                    wds.batched(self.train_params.batch_size, partial=True),
                ]).with_epoch(num_worker_batches)
        self.train_dl = wds.WebLoader(train_data, num_workers=num_workers,
                         batch_size=None, shuffle=False, persistent_workers=True)

        # Load all text annotations and select the annotations for subject 1
        f = h5py.File('/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_subj_indices.hdf5', 'r')
        subj01_order = f['subj01'][:]
        f.close()
        annots = np.load('/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_annots_curated.npy',allow_pickle=True)
        self.subj01_annots = annots[subj01_order]
        
        if self.train_params.rank == 0:
            print()
            batch = next(iter(self.train_dl))
            print('fMRI shape:', batch[0].shape) 
            print(f'Image shape:', batch[1].shape)
            print(f'Text shape:', batch[2].shape) # note that we only load the text label annotations, not the text itself !
            print()
            
        # Build data augmentation pipeline
        self.augment_pipeline = training.augment.AugmentPipe(**self.data_params.augment_kwargs).to(self.device)
            
    def build_model_optimizer(self):
        # Construct networks.
        if self.train_params.rank == 0:
            print('Constructing networks...')
            
        # fMRI mappers are frozen at this stage
        self.fMRI_to_image_mapper = networks.BrainNetwork(self.model_optim_params.emb_shape).requires_grad_(False).to(self.device)
        ckpt_path = 'checkpoints/clip_image_vitB_conv_subj01_epoch35.pth'
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.fMRI_to_image_mapper.load_state_dict(checkpoint['model_state_dict'])
        self.fMRI_to_image_mapper.eval()
        
        self.fMRI_to_text_mapper = networks.BrainNetwork(self.model_optim_params.emb_shape).requires_grad_(False).to(self.device) 
        ckpt_path_txt = 'checkpoints/clip_text_vitB_conv_subj01_epoch25.pth'
        checkpoint_txt = torch.load(ckpt_path_txt, map_location=self.device)
        self.fMRI_to_text_mapper.load_state_dict(checkpoint_txt['model_state_dict'])
        self.fMRI_to_text_mapper.eval()
        
        # CLIP model
        self.clip_extractor = networks.Clipper(self.model_optim_params.clip_model_name, self.device)

        # Load in Lafite Generator
        self.Generator = networks.Generator(**self.model_optim_params.G_kwargs).to(self.device)
        ckpt_path_lafite = 'checkpoints/COCO2014_Language-free_Gaussian.pkl'
        checkpoint_lafite = load_file_pickle(ckpt_path_lafite)
        self.Generator.load_state_dict(checkpoint_lafite["G"].state_dict(), strict=False)

        # Load in Lafite Discriminator
        self.Discriminator = networks.Discriminator(**self.model_optim_params.D_kwargs).to(self.device)
        self.Discriminator.load_state_dict(checkpoint_lafite["D"].state_dict(), strict=False)
        # common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
        # G = networks.Generator()
        # dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        # D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        # G_ema = copy.deepcopy(G).eval()
#         if use_fmri:
#             fmri_vec = dnnlib.util.construct_class_by_name(**mapper_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
#             # first time only, remove afterwards
#             # misc.load_trained_model('/home/sikun/bold5k/data/weights/fmri_clipcapnorm_mse_cos_thr_noBN_0.19923493794099553.pth', fmri_vec) # cap
#             # misc.load_trained_model('/home/sikun/bold5k/data/weights/fmri_clipcapnorm_mse_cos_contra_thr_noBN_2.3636860251426697.pth', fmri_vec) # cap with contra
#             # misc.load_trained_model('/home/sikun/bold5k/data/weights/fmri_clipnorm_mse_cos_aug_thr_noBN_0.19915693834072024.pth', fmri_vec) # img
#             misc.load_trained_model('/home/sikun/bold5k/data/weights/fmri_clipnorm_mse_cos_contra_aug_thr_noBN_2.3462039679288864.pth', fmri_vec) # img with contra
#             fmri_vec.to(torch.double)
#             if structure in [4, 5]:
#                 fmri_vec2 = dnnlib.util.construct_class_by_name(**mapper2_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
#                 # first time only, remove afterwards
#                 # misc.load_trained_model('/home/sikun/bold5k/data/weights/fmri_dnn_noBN_0.010493216838130332.pth', fmri_vec2)
#                 # misc.load_trained_model('/home/sikun/bold5k/data/weights/fmri_clipcapnorm_mse_cos_contra_thr_noBN_2.3636860251426697.pth', fmri_vec2) # cap with contra
#                 fmri_vec2.to(torch.double)

    def get_perturbed_embeddings(self, clipaligned_img_emb, clipaligned_text_emb):
        
        # augmentation levels
        aug_level_1 = 0.1
        aug_level_2 = 0.75 # not really used ?
        txt_random_noise = torch.randn(clipaligned_text_emb.shape).to(clipaligned_text_emb.device)
        txt_random_noise = txt_random_noise/txt_random_noise.norm(dim=-1, keepdim=True)
        clipaligned_text_emb_perturbed = clipaligned_text_emb*(1-aug_level_1) + txt_random_noise*aug_level_1
        clipaligned_text_emb_perturbed = clipaligned_text_emb_perturbed/clipaligned_text_emb_perturbed.norm(dim=-1, keepdim=True)
        
        img_random_noise = torch.randn(clipaligned_img_emb.shape).to(clipaligned_img_emb.device) 
        img_random_noise = img_random_noise/img_random_noise.norm(dim=-1, keepdim=True)
        clipaligned_img_emb_perturbed = clipaligned_img_emb*(1-aug_level_1) + img_random_noise*clipaligned_img_emb
        clipaligned_img_emb_perturbed = clipaligned_img_emb_perturbed/clipaligned_img_emb_perturbed.norm(dim=-1, keepdim=True)
        return clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed
    
    def get_styles(self, clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed):
        z = torch.zeros([self.train_params.batch_size,self.Generator.z_dim], device=self.device)
        c = None
        ws = self.Generator.mapping(z, c)

        if self.train_params.style_mixing_prob > 0:
            new_ws = self.Generator.mapping(torch.randn_like(z), c, skip_w_avg_update=True)
            
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.train_params.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = new_ws[:, cutoff:]
        return ws        
    
    
    def run_discriminator(self, img, fts):
        c = None
        img = self.augment_pipeline(img)
        logits, d_fts, d_fts2 = self.Discriminator(img, c, fts=fts)
        return logits, d_fts, d_fts2
    
    def train(self):
        pbar = tqdm.tqdm(range(self.train_params.num_train_epochs),ncols=250)
        for epoch in pbar:
            for train_i, (voxel, img_input, cap_id) in enumerate(self.train_dl):
                voxel = voxel.to(self.device).float()
                img_emb = self.clip_extractor.embed_image(img_input).to(self.device).float()
                #print("img_emb", img_emb.shape)
                text_emb = self.clip_extractor.embed_curated_annotations(self.subj01_annots[cap_id]).float()
                #print("self.fMRI_to_image_mapper", self.fMRI_to_image_mapper)
                
                # Map fmri to clip-aligned image and text features
                clipaligned_img_emb = self.fMRI_to_image_mapper(voxel)
                clipaligned_text_emb = self.fMRI_to_text_mapper(voxel)
                clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed = self.get_perturbed_embeddings(clipaligned_img_emb, clipaligned_text_emb)
                fts = torch.cat((clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed),-1) # concat fmri mapped features
                
                # Get styles
                ws = self.get_styles(clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed)
                
                # Generate image
                # The synthesizer automatically maps the fmri-mapped clip features to condition codes
                # as in page 3 of MindReader
                gen_img = self.Generator.synthesis(ws, fts=fts)
                
                # Discriminator
                real_or_fake_logits, discriminator_img_emb, discriminator_txt_emb = self.run_discriminator(gen_img, fts=fts)
                
                # GAN-Losses
                loss_GAN_G = F.softplus(-real_or_fake_logits)
                # Process generated image
                print(f'gen img shape: {gen_img.shape}, range: {gen_img.min(), gen_img.max()}')
                normed_gen_full_img = train_utils.full_preprocess(gen_img)
                normed_gen_full_img_clip = self.clip_extractor.embed_image(normed_gen_full_img)
                normed_gen_full_img_clip = normed_gen_full_img_clip/normed_gen_full_img_clip.norm(dim=-1, keepdim=True)
                
                # Lc2
                Lc2_img_img = train_utils.contra_loss(self.train_params.contrastive_params.temp, normed_gen_full_img_clip, clipaligned_img_emb_perturbed, self.train_params.contrastive_params.lam) # align clip-embedding of generated image with clip-aligned image-fmri
                Lc2_img_txt = train_utils.contra_loss(self.train_params.contrastive_params.temp, normed_gen_full_img_clip, clipaligned_text_emb_perturbed, self.train_params.contrastive_params.lam)
                Lc2 = -Lc2_img_img.mean() -Lc2_img_txt.mean()
                
                # Lc1
                Lc1_discrim_img = train_utils.contra_loss(self.train_params.contrastive_params.temp, discriminator_img_emb, clipaligned_img_emb_perturbed, self.train_params.contrastive_params.lam)
                Lc1_discrim_txt = train_utils.contra_loss(self.train_params.contrastive_params.temp, discriminator_txt_emb, clipaligned_text_emb_perturbed, self.train_params.contrastive_params.lam)
                Lc1 = -Lc1_discrim_img.mean() -Lc1_discrim_txt.mean()
                
                print(f'normed gen img shape: {normed_gen_full_img.shape}, range: {normed_gen_full_img.min(), normed_gen_full_img.max()}')
                
                print("clipaligned_embs_perturbed", clipaligned_embs_perturbed.shape)
                #gen_img, _ = self.run_G(gen_z, gen_c, txt_fts=txt_fts_, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                
                #with torch.cuda.amp.autocast():
                print("hi")
                if np.random.rand() < 1:
                    print(wa)
   
def initialize(rank, train_params):
    # Initialize.
    print("Rank", rank)
    start_time = time.time()
    if torch.cuda.is_available():
        device = torch.device('cuda', rank)
    else:
        device = torch.device('cpu')
    np.random.seed(train_params.random_seed * train_params.num_gpus + rank)
    torch.manual_seed(train_params.random_seed * train_params.num_gpus + rank)
    torch.backends.cudnn.benchmark = train_params.cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = train_params.allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = train_params.allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    torch_utils.ops.conv2d_gradfix.enabled = True                       # Improves training speed.
    #torch_utils.ops.conv2d_gradfix.enabled = False                       # Speed is worse but we avoid using their custom op ?
    torch_utils.ops.grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.
    return device

def initialize_record(train_params, data_params, model_optim_params, save_params):
    record = dnnlib.EasyDict(
            curr_epoch = -1,
            metrics = dnnlib.EasyDict(
            ),    
            data = dnnlib.EasyDict(
                inputs = None,
                labels = None
            ),
            print_interval = 5000 // train_params['batch_size'],
            data_params = data_params,
            train_params = train_params,
            model_optim_params = model_optim_params,
            save_params = save_params,
            success = False
        
            
        )
    
    record_model = dnnlib.EasyDict(
                state_dict = None, 
                optimizer = None,
                scheduler = None,
                best_model = None,
            )
    
    return record, record_model
def load_training_set(rank, train_params, data_params, device):
    # Load training set.
    num_samples = 24983
    num_workers = train_params.num_gpus
    global_batch_size = train_params.batch_size * train_params.num_gpus
    print("global_batch_size", global_batch_size)
    num_batches = math.floor(num_samples / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)


    if rank == 0:
        print('Loading training set...')

    train_data = wds.DataPipeline([wds.ResampledShards(data_params.train_url),
                wds.tarfile_to_samples(),
                wds.shuffle(500,initial=500),
                wds.decode("torch"),
                # wds.rename(images="jpg;png", voxels="nsdgeneral.npy", embs="sgxl_emb.npy", trial="trial.npy"),
                wds.rename(images="jpg;png", voxels="nsdgeneral.npy", trial="trial.npy"),
                wds.to_tuple("voxels", 'images', 'trial'),
                wds.batched(train_params.batch_size, partial=True),
            ]).with_epoch(num_worker_batches)
    train_dl = wds.WebLoader(train_data, num_workers=num_workers,
                     batch_size=None, shuffle=False, persistent_workers=True)

    val_data = wds.DataPipeline([wds.SimpleShardList(data_params.val_url),
                        wds.tarfile_to_samples(),
                        wds.decode("torch"),
                        wds.rename(images="jpg;png", voxels="nsdgeneral.npy", 
                                    embs="sgxl_emb.npy", trial="trial.npy"),
                        wds.to_tuple("voxels", 'images', 'trial'),
                        wds.batched(train_params.batch_size, partial=True),
                    ]).with_epoch(num_worker_batches)
    val_dl = wds.WebLoader(val_data, num_workers=num_workers,
                           batch_size=None, shuffle=False, persistent_workers=True)

    # Load all text annotations and select the annotations for subject 1
    f = h5py.File('/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_subj_indices.hdf5', 'r')
    subj01_order = f['subj01'][:]
    f.close()
    annots = np.load('/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_annots_curated.npy',allow_pickle=True)
    subj01_annots = annots[subj01_order]

    if rank == 0:
        batch = next(iter(train_dl))
        print('fMRI shape:', batch[0].shape) 
        print(f'Image shape:', batch[1].shape)
        print(f'Text shape:', batch[2].shape) # note that we only load the text label annotations, not the text itself !
        print()

    # Build data augmentation pipeline
    augment_pipeline = None
    if (data_params.augment_kwargs is not None) and (data_params.augment_p > 0 or data_params.ada_target is not None):
        augment_pipeline = training.augment.AugmentPipe(**data_params.augment_kwargs).to(device)
        augment_pipeline.p.copy_(torch.as_tensor(data_params.augment_p))
    
    return train_dl, val_dl, augment_pipeline, subj01_annots

def load_models(rank, train_params, model_optim_params, augment_pipeline, device):
    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    # fMRI mappers are frozen at this stage
    fMRI_to_image_mapper = networks.BrainNetwork(model_optim_params.emb_shape).requires_grad_(False).to(device)
    ckpt_path = 'checkpoints/clip_image_vitB_conv_subj01_epoch35.pth'
    checkpoint = torch.load(ckpt_path, map_location=device)
    fMRI_to_image_mapper.load_state_dict(checkpoint['model_state_dict'])
    fMRI_to_image_mapper.eval()

    fMRI_to_text_mapper = networks.BrainNetwork(model_optim_params.emb_shape).requires_grad_(False).to(device) 
    ckpt_path_txt = 'checkpoints/clip_text_vitB_conv_subj01_epoch25.pth'
    checkpoint_txt = torch.load(ckpt_path_txt, map_location=device)
    fMRI_to_text_mapper.load_state_dict(checkpoint_txt['model_state_dict'])
    fMRI_to_text_mapper.eval()

    # CLIP model
    clip_extractor = networks.Clipper(model_optim_params.clip_model_name, device)

    # Load in Lafite Generator
    Generator = networks.Generator(**model_optim_params.G_kwargs).train().requires_grad_(False).to(device)
    ckpt_path_lafite = 'checkpoints/COCO2014_Language-free_Gaussian.pkl'
    checkpoint_lafite = load_file_pickle(ckpt_path_lafite)
    Generator.load_state_dict(checkpoint_lafite["G"].state_dict(), strict=False)

    # Exponential moving average of Generator 
    Generator_ema = copy.deepcopy(Generator).eval()
    
    # Load in Lafite Discriminator
    Discriminator = networks.Discriminator(**model_optim_params.D_kwargs).train().requires_grad_(False).to(device)
    Discriminator.load_state_dict(checkpoint_lafite["D"].state_dict(), strict=False)
    
    # Resnet
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet_preprocess = ResNet50_Weights.DEFAULT.transforms()


    
    # DDP
    if (train_params.num_gpus > 1):
        for module in [fMRI_to_image_mapper, fMRI_to_text_mapper, clip_extractor, Generator, Discriminator, augment_pipeline]:
            if (module is None) or len(list(module.parameters())) == 0:
                continue
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
   

    return fMRI_to_image_mapper, fMRI_to_text_mapper, clip_extractor, Generator, Generator_ema, Discriminator, resnet, resnet_preprocess

def Print_network_summary_tables(rank, G, D, model_optim_params, device):
    if rank == 0:
        z = torch.empty([16, G.z_dim], device=device)
        c = torch.empty([16, G.c_dim], device=device)
        if model_optim_params.G_kwargs.synthesis_kwargs.structure == 4:
            fts = torch.empty([16, model_optim_params.G_kwargs.synthesis_kwargs.f_dim+model_optim_params.G_kwargs.synthesis_kwargs.f_dim2], device=device)
        else:
            fts = torch.empty([16, model_optim_params.G_kwargs.synthesis_kwargs.f_dim], device=device)
        img = torch_utils.misc.print_module_summary(G, [z, c])
        torch_utils.misc.print_module_summary(D, [img, c, fts])
        

       
def load_optimizer(Generator, Discriminator, model_optim_params):
    # Generator main and 
    mb_ratio = model_optim_params.opt_kwargs.G_reg_interval / (model_optim_params.opt_kwargs.G_reg_interval + 1)
    G_lrate = model_optim_params.opt_kwargs.G_lrate * mb_ratio
    betas = [beta ** mb_ratio for beta in model_optim_params.opt_kwargs.betas]
    Generator_opt = torch.optim.Adam(Generator.parameters(), lr= G_lrate, betas=betas, eps=model_optim_params.opt_kwargs.eps)
    
    # Discriminator optimizer
    mb_ratio = model_optim_params.opt_kwargs.D_reg_interval / (model_optim_params.opt_kwargs.D_reg_interval + 1)
    D_lrate = model_optim_params.opt_kwargs.D_lrate * mb_ratio
    betas = [beta ** mb_ratio for beta in model_optim_params.opt_kwargs.betas]
    Discriminator_opt = torch.optim.Adam(Discriminator.parameters(), lr= D_lrate, betas= betas, eps=model_optim_params.opt_kwargs.eps)
    
    
    return Generator_opt, Discriminator_opt
    
def load_training_phases(Generator, Discriminator, Generator_opt, Discriminator_opt, model_optim_params):
    phases = []
    phase_params = [("Generator", Generator, Generator_opt, model_optim_params.opt_kwargs.G_reg_interval), 
     ("Discriminator", Discriminator, Discriminator_opt, model_optim_params.opt_kwargs.D_reg_interval)]
    
    for name, module, opt, reg_interval in phase_params:
        phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
        phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
        
    return phases    

def get_perturbed_embeddings(clipaligned_img_emb, clipaligned_text_emb):
    # augmentation levels
    aug_level_1 = 0.1
    aug_level_2 = 0.75 # not really used ?
    txt_random_noise = torch.randn(clipaligned_text_emb.shape).to(clipaligned_text_emb.device)
    txt_random_noise = txt_random_noise/txt_random_noise.norm(dim=-1, keepdim=True)
    clipaligned_text_emb_perturbed = clipaligned_text_emb*(1-aug_level_1) + txt_random_noise*aug_level_1
    clipaligned_text_emb_perturbed = clipaligned_text_emb_perturbed/clipaligned_text_emb_perturbed.norm(dim=-1, keepdim=True)

    img_random_noise = torch.randn(clipaligned_img_emb.shape).to(clipaligned_img_emb.device) 
    img_random_noise = img_random_noise/img_random_noise.norm(dim=-1, keepdim=True)
    clipaligned_img_emb_perturbed = clipaligned_img_emb*(1-aug_level_1) + img_random_noise*clipaligned_img_emb
    clipaligned_img_emb_perturbed = clipaligned_img_emb_perturbed/clipaligned_img_emb_perturbed.norm(dim=-1, keepdim=True)
    return clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed
    
def get_styles(clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed, Generator, train_params, device):
    batch_size = clipaligned_text_emb_perturbed.shape[0]
    z = torch.zeros([batch_size,Generator.z_dim], device=device) # fix z
    c = None
    ws = Generator.mapping(z, c)

    if train_params.style_mixing_prob > 0:
        new_ws = Generator.mapping(torch.randn_like(z), c, skip_w_avg_update=True)

        with torch.autograd.profiler.record_function('style_mixing'):
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(torch.rand([], device=ws.device) < train_params.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = new_ws[:, cutoff:]
    return ws  

def run_discriminator(img, fts, augment_pipeline, Discriminator):
    c = None
    if augment_pipeline is not None:
        img = augment_pipeline(img)
    logits, d_fts, d_fts2 = Discriminator(img, c, fts=fts, force_fp32=not torch.cuda.is_available())
    return logits, d_fts, d_fts2

def run_res(img, resnet):
    img = train_utils.full_preprocess(img)
    resnet_fts = resnet(img, chan_avg=False, return_layer= 2)
    max_vec = resnet_fts.max(1)[0]
    mean_vec = resnet_fts.mean(1)
    resnet_fts = max_vec / max_vec.max() + 2 * mean_vec / mean_vec.max()
    return resnet_fts.flatten(start_dim=1)

def perform_phase(phase, clipaligned_img_emb, clipaligned_text_emb, augment_pipeline, clip_extractor, Generator, Discriminator, resnet, pl_mean, train_params, img_input, device):
    phase.opt.zero_grad()
    phase.module.requires_grad_(True)
    clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed = get_perturbed_embeddings(clipaligned_img_emb, clipaligned_text_emb)
    fts = torch.cat((clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed),-1) # concat fmri mapped features
    #print("clipaligned_img_emb_perturbed", clipaligned_img_emb_perturbed.shape, "clipaligned_text_emb_perturbed", clipaligned_text_emb_perturbed.shape)
    metrics = dnnlib.EasyDict()
    if phase.name == "Generatormain":
        

        # Get styles
        ws = get_styles(clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed, Generator, train_params, device)

        # Generate image
        # The synthesizer automatically maps the fmri-mapped clip features to condition codes
        # as in page 3 of MindReader
        gen_img = Generator.synthesis(ws, fts=fts, force_fp32=not torch.cuda.is_available())

        # Discriminator
        real_or_fake_logits, discriminator_img_emb, discriminator_txt_emb = run_discriminator(gen_img, fts, augment_pipeline, Discriminator)

        # GAN-Losses
        loss_GAN_Generator = F.softplus(-real_or_fake_logits)

        # Process generated image
        # print(f'gen img shape: {gen_img.shape}, range: {gen_img.min(), gen_img.max()}')
        normed_gen_full_img = train_utils.full_preprocess(gen_img)
        normed_gen_full_img_clip = clip_extractor.embed_image(normed_gen_full_img) #h_img^gen
        normed_gen_full_img_clip = F.normalize(normed_gen_full_img_clip,dim=-1) 

        # Lc2
        Lc2_img_img = train_utils.contra_loss(train_params.contrastive_params.temp, normed_gen_full_img_clip, clipaligned_img_emb_perturbed, train_params.contrastive_params.lam) # align clip-embedding of generated image with clip-aligned image-fmri
        Lc2_img_txt = train_utils.contra_loss(train_params.contrastive_params.temp, normed_gen_full_img_clip, clipaligned_text_emb_perturbed, train_params.contrastive_params.lam)
        Lc2 = -Lc2_img_img.mean() -Lc2_img_txt.mean()

        # Lc1
        Lc1_img_img = train_utils.contra_loss(train_params.contrastive_params.temp, discriminator_img_emb, clipaligned_img_emb_perturbed, train_params.contrastive_params.lam)
        Lc1_txt_txt = train_utils.contra_loss(train_params.contrastive_params.temp, discriminator_txt_emb, clipaligned_text_emb_perturbed, train_params.contrastive_params.lam)
        Lc1 = -Lc1_img_img.mean() -Lc1_txt_txt.mean()

        # Lc3
        # run_res(gen_img, resnet)

        
        # Backward
        loss_Generator = loss_GAN_Generator + train_params.lambda_1 * Lc1 + train_params.lambda_2 * Lc2 # + train_params.lambda_3 * Lc3
        loss_Generator = loss_Generator.mean().mul(phase.interval)
        # print("loss_Generator", loss_Generator)
        loss_Generator.backward()
        metrics.loss_GAN_Generator = loss_GAN_Generator.detach()
        metrics.Lc1 = Lc1.detach()
        metrics.Lc2 = Lc2.detach()
        
    elif phase.name == "Generatorreg": # path length regularization for generator
        batch_size = train_params.batch_size // train_params.pl_batch_shrink
        fts = fts[:batch_size]
        fts.requires_grad_()
        
        # Get styles
        gen_ws = get_styles(clipaligned_img_emb_perturbed[:batch_size], clipaligned_text_emb_perturbed[:batch_size], Generator, train_params, device)

        # Generate image
        # The synthesizer automatically maps the fmri-mapped clip features to condition codes
        # as in page 3 of MindReader
        gen_img = Generator.synthesis(gen_ws, fts=fts, force_fp32=not torch.cuda.is_available())
        
        pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
        with torch.autograd.profiler.record_function('pl_grads'), torch_utils.ops.conv2d_gradfix.no_weight_gradients():
            pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws, fts], create_graph=True, only_inputs=True)[0]
        pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        _tmp = pl_mean.lerp(pl_lengths.mean(), train_params.pl_decay)
        pl_mean.copy_(_tmp.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        loss_Gpl = pl_penalty * train_params.pl_weight
        
        #with torch.autograd.profiler.record_function('Gpl_backward'):
        #    print("gen_img[:, 0, 0, 0] * 0", gen_img[:, 0, 0, 0] * 0)
            #(gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(phase.interval).backward() # ? why gen_img[:, 0, 0, 0] * 0
            
        # Backward
        (loss_Gpl).mean().mul(phase.interval).backward() # ? why gen_img[:, 0, 0, 0] * 0
        
        metrics.pl_penalty = pl_penalty.detach()
        # print("loss_Gpl", loss_Gpl)
    elif phase.name == "Discriminatormain": 
        # Dmain: Minimize logits for generated images.
        # Get styles
        ws = get_styles(clipaligned_img_emb_perturbed, clipaligned_text_emb_perturbed, Generator, train_params, device)

        # Generate image
        # The synthesizer automatically maps the fmri-mapped clip features to condition codes
        # as in page 3 of MindReader
        gen_img = Generator.synthesis(ws, fts=fts, force_fp32=not torch.cuda.is_available())

        # Discriminator
        real_or_fake_logits, discriminator_img_emb, discriminator_txt_emb = run_discriminator(gen_img, fts, augment_pipeline, Discriminator)
        
        loss_Dgen = torch.nn.functional.softplus(real_or_fake_logits) # -log(1 - sigmoid(gen_logits))
        loss_Dgen.mean().mul(phase.interval).backward()
        
        # Dmain: Maximize logits for real images 
        real_logits, discriminator_img_emb, discriminator_txt_emb = run_discriminator(img_input.to(device).detach(), fts, augment_pipeline, Discriminator)
        loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
        Lc1_txt_txt = train_utils.contra_loss(train_params.contrastive_params.temp, discriminator_txt_emb, clipaligned_text_emb_perturbed, train_params.contrastive_params.lam)
        Lc1_img_img = train_utils.contra_loss(train_params.contrastive_params.temp, discriminator_img_emb, clipaligned_img_emb_perturbed, train_params.contrastive_params.lam)
        loss_Dreal = loss_Dreal - train_params.lambda_2 * (Lc1_txt_txt + Lc1_img_img)
        
        # Backward
        #(real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward() ?? real_logits * 0 
        (loss_Dreal).mean().mul(phase.interval).backward()
        

        metrics.loss_Dreal = loss_Dreal.detach()
        metrics.Lc1_txt_txt = Lc1_txt_txt.detach()
        metrics.Lc1_img_img = Lc1_img_img.detach()
        # print("loss_Dreal", loss_Dreal)
    elif phase.name == "Discriminatorreg": # and apply R1 regularization.
        real_img_tmp = img_input.to(device).detach().requires_grad_()
        real_logits, discriminator_img_emb, discriminator_txt_emb = run_discriminator(real_img_tmp, fts, augment_pipeline, Discriminator)
        with torch.autograd.profiler.record_function('r1_grads'), torch_utils.ops.conv2d_gradfix.no_weight_gradients():
            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
        r1_penalty = r1_grads.square().sum([1,2,3])
        loss_Dr1 = r1_penalty * (train_params.r1_gamma / 2)
        
        # Backward
        (loss_Dr1).mean().mul(phase.interval).backward()
        
        metrics.r1_penalty = r1_penalty.detach()
       
        # print("loss_Dr1", loss_Dr1, flush=True)
    phase.module.requires_grad_(False)
    
    return pl_mean, metrics
    
    
def after_trainiter_callback(metrics, temp_metrics_tracker, phase, epoch, batch_idx, batch_size, tracker_init = [[],0]):
    for metric in metrics:
        try:
            temp_metrics_tracker[metric][0].append(metrics[metric].sum().detach()) 
            temp_metrics_tracker[metric][1] += batch_size
        except:
            temp_metrics_tracker[metric] = copy.deepcopy(tracker_init) # save sum and count for average
            
        
    
    # Update weights
    with torch.autograd.profiler.record_function(phase.name + '_opt'):
        for param in phase.module.parameters():
            if param.grad is not None:
                torch_utils.misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        phase.opt.step()
            
    return temp_metrics_tracker

def after_batch_callback(train_params, Generator, Generator_ema, Discriminator, epoch, batch_idx, cur_nimg, temp_metrics_tracker, record, n_iters = 100, tracker_init = [[],0]):
    if batch_idx % n_iters == n_iters - 1:
        # track moving average of metrics every n_iters iters
        for metric in temp_metrics_tracker:
            sums, count = torch.sum(torch.stack(temp_metrics_tracker[metric][0])), temp_metrics_tracker[metric][1]
            print(metric, "sums count", sums, count )
            record.metrics.setdefault(metric,{}).setdefault(epoch,[]).append(sums / count)
        temp_metrics_tracker = {} # [metric] = tracker_init
        print("batch_idx" , batch_idx, "temp_metrics_tracker", temp_metrics_tracker)
        
    cur_nimg += train_params.batch_size
    
    # Update Generator_ema
    with torch.autograd.profiler.record_function('Gema'):
        ema_nimg = train_params.ema_kwargs.ema_kimg * 1000
        if train_params.ema_kwargs.ema_rampup is not None:
            ema_nimg = min(ema_nimg, cur_nimg * train_params.ema_kwargs.ema_rampup)
        ema_beta = 0.5 ** (train_params.batch_size / max(ema_nimg, 1e-8))
        for p_ema, p in zip(Generator_ema.parameters(), Generator.parameters()):
            p_ema.copy_(p.lerp(p_ema, ema_beta))
        for b_ema, b in zip(Generator_ema.buffers(), Generator.buffers()):
            b_ema.copy_(b)

    # Not done in Lafite code but I include it here for completeness
    # Execute ADA heuristic.
    #if (ada_stats is not None) and (batch_idx % ada_interval == 0):
    #    ada_stats.update()
    #    adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
    #    augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

    # Compute Metric
    
    return temp_metrics_tracker, record

def after_epoch_callback(epoch, rank, record, record_model,
                         train_params, save_params,
                         train_dl, val_dl,
                        augment_pipeline, Generator, Generator_ema, Discriminator, resnet, Generator_opt, Discriminator_opt
                        ):
    # compute fid metrics
    result_dict = metric_main.calc_metric(metric='fid50k_full', G=Generator_ema, D=Discriminator,
                dataset_kwargs=training_set_kwargs, testset_kwargs=testing_set_kwargs, num_gpus=num_gpus, rank=rank,
                device=device, txt_recon=True, img_recon=False, metric_only_test=metric_only_test, use_fmri=use_fmri,
                fmri_vec=fmri_vec_eval, fmri_vec2=fmri_vec2_eval, structure=structure)
    
    if rank == 0:
        record.curr_epoch = epoch


        #if val_acc1 > record.best_val_acc1:
        #    record.best_model = copy.deepcopy(model.state_dict())
        #    record.best_val_acc1 = max(val_acc1, record.best_val_acc1)

        #record.metrics.train_losses[epoch] = train_losses
        #record.metrics.train_acc5[epoch] = train_acc5
        #record.metrics.train_acc1[epoch] = train_acc1
        #record.metrics.val_losses[epoch] = val_losses
        #record.metrics.val_acc5[epoch] = val_acc5
        #record.metrics.val_acc1[epoch] = val_acc1

        # Save models
        name_module_pairs = [('Generator', Generator), ('Discriminator', Discriminator), ('Generator_ema', Generator_ema), ('augment_pipeline', augment_pipeline)]
        for name, module in name_module_pairs:
            if module is not None:
                if train_params.num_gpus > 1:
                    torch_utils.misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                record_model[name] = copy.deepcopy(module).eval().requires_grad_(False).cpu().state_dict()

        # Save optimizers
        record_model.Generator_opt = Generator_opt.state_dict()
        record_model.Discriminator_opt = Discriminator_opt.state_dict()

        train_utils.save_checkpoint(record, save_dir = save_params.save_dir, filename = save_params.exp_name)
        train_utils.save_checkpoint(record_model, save_dir = save_params.save_dir, filename = f"weights_{save_params.exp_name}")

def train(rank, phases, train_params, save_params, train_dl, val_dl, augment_pipeline, resnet_preprocess, clip_extractor, fMRI_to_image_mapper, fMRI_to_text_mapper, 
          Generator, Generator_ema, Discriminator, resnet, Generator_opt, Discriminator_opt, subj01_annots, record, 
          record_model, device):
    pbar = tqdm.tqdm(range(train_params.num_train_epochs),ncols=250)
    pl_mean = torch.zeros([], device=device)
    cur_nimg = 0
    temp_metrics_tracker = {}
    for epoch in pbar:
        for batch_idx, (voxel, img_input, cap_id) in enumerate(train_dl):
            
            voxel = voxel.to(device).float()
            img_emb = clip_extractor.embed_image(img_input).to(device).float()
            #print("img_emb", img_emb.shape)
            text_emb = clip_extractor.embed_curated_annotations(subj01_annots[cap_id]).float()
            batch_size = voxel.size(0)
            #print("self.fMRI_to_image_mapper", self.fMRI_to_image_mapper)

             # Map fmri to clip-aligned image and text features
            clipaligned_img_emb = fMRI_to_image_mapper(voxel)
            clipaligned_text_emb = fMRI_to_text_mapper(voxel)

            # page 3 of MindReader: clamp and then normalize
            clipaligned_img_emb = F.normalize(torch.clamp(clipaligned_img_emb, -1.5, 1.5), dim=1)
            clipaligned_text_emb = F.normalize(torch.clamp(clipaligned_text_emb, -1.5, 1.5), dim=1)
            
            for phase in phases:
                if batch_idx % phase.interval != 0:
                    continue
                pl_mean, metrics = perform_phase(phase, clipaligned_img_emb, clipaligned_text_emb, augment_pipeline, clip_extractor, Generator, Discriminator, resnet, pl_mean, train_params, img_input, device)
                
                    
                temp_metrics_tracker = after_trainiter_callback(metrics, temp_metrics_tracker, phase, epoch, batch_idx, batch_size)


             
            
                    
            
            temp_metrics_tracker, record = after_batch_callback(train_params, Generator, Generator_ema, Discriminator, epoch, batch_idx, cur_nimg, temp_metrics_tracker, record)
            break
        # After epoch, run
        after_epoch_callback(epoch, rank, record, record_model,
                         train_params, save_params,
                             train_dl, val_dl,
                        augment_pipeline, Generator, Generator_ema, Discriminator, resnet, Generator_opt, Discriminator_opt
                        )    
        

def training_loop(rank, train_params, data_params, model_optim_params, save_params):
    print("Train params", train_params)
    device = initialize(rank, train_params)
    record, record_model = initialize_record(train_params, data_params, model_optim_params, save_params)
    
    train_dl, val_dl, augment_pipeline, subj01_annots = load_training_set(rank, train_params, data_params, device)
    
    fMRI_to_image_mapper, fMRI_to_text_mapper, clip_extractor, Generator, Generator_ema, Discriminator, resnet, resnet_preprocess = load_models(rank, train_params, model_optim_params, augment_pipeline, device)
    Generator_opt, Discriminator_opt = load_optimizer(Generator, Discriminator, model_optim_params)
    phases = load_training_phases(Generator, Discriminator, Generator_opt, Discriminator_opt, model_optim_params)
    print("phases", [p.name for p in phases])
    train(rank, phases, train_params, save_params, train_dl, val_dl, augment_pipeline, resnet_preprocess, clip_extractor, fMRI_to_image_mapper, fMRI_to_text_mapper, 
          Generator, Generator_ema, Discriminator, resnet, Generator_opt, Discriminator_opt, subj01_annots, record, record_model, device)
    

    #Print_network_summary_tables(rank, Generator, Discriminator, model_optim_params, device)
    
    # Distribute across GPUs
    
#     pbar = tqdm.tqdm(range(train_params.num_train_epochs),ncols=250)
#     for epoch in pbar:
#         for train_i, (voxel, img_input, cap_id) in enumerate(train_dl):
#             pass