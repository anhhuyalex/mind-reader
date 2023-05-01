import torch
print("Torch version", torch.__version__)
from torch import nn

from Lafite import dnnlib
from Lafite import torch_utils
from Lafite import training
import sys
sys.path.append('./Lafite')
import utils
import tempfile
import argparse
import os
import datetime

def get_parser():
    """
    Possible save_dir
        "/gpfs/scratch60/turk-browne/an633/aha"
        "/gpfs/milgram/scratch60/turk-browne/an633/aha"
    """
    parser = argparse.ArgumentParser(
            description='Pytorch training framework for general dist training')
    parser.add_argument(
            '--run_dir', 
            default="/gpfs/milgram/scratch60/turk-browne/an633/mind_reader", type=str, 
            # default="/gpfs/milgram/scratch60/turk-browne/an633/mind_reader", type=str, 
            action='store')
    parser.add_argument(
            '--model_name', 
            default="Lafite", type=str, 
            action='store')
    
    parser.add_argument(
            '--batch_size', 
            default=16, type=int, 
            action='store')
    parser.add_argument(
            '--num_gpus', 
            default=1, type=int, 
            action='store')
    parser.add_argument(
            '--num_train_epochs', 
            default=90, type=int, 
            action='store')
    parser.add_argument(
            '--structure', 
            default=4, type=int, 
            action='store')
    parser.add_argument(
            '--train_url', 
            default="/scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_avg_split/train/train_subj01_{0..17}.tar", type=str, 
            #default="/gpfs/milgram/scratch60/turk-browne/an633/nsd/train_subj01_{0..49}.tar", type=str, 
            action='store')
    parser.add_argument(
            '--train_indices_batchidx_dict_url', 
            default="/scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_avg_split/train/train_subj01_{0..17}.tar", type=str, 
            #default="/gpfs/milgram/scratch60/turk-browne/an633/nsd/train_subj01_{0..49}.tar", type=str, 
            action='store')
    parser.add_argument(
            '--val_url', 
            default="/scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_avg_split/val/val_subj01_0.tar", type=str, 
            #default="/gpfs/milgram/scratch60/turk-browne/an633/nsd/val/val_subj01_0.tar", type=str, 
            action='store') 
    parser.add_argument(
            '--val_indices_batchidx_dict_url', 
            default="/scratch/gpfs/KNORMAN/webdataset_nsd/webdataset_avg_split/val/val_subj01_0.tar", type=str, 
            #default="/gpfs/milgram/scratch60/turk-browne/an633/nsd/val/val_subj01_0.tar", type=str, 
            action='store') 
            
    parser.add_argument(
            '--subjectorder_url', 
            default="/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_subj_indices.hdf5", type=str,
            #default="/gpfs/milgram/scratch60/turk-browne/an633/nsd/COCO_73k_subj_indices.hdf5", type=str, 
            action='store')
    parser.add_argument(
            '--annotations_url', 
            default="/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_annots_curated.npy", type=str, 
            #default="/gpfs/milgram/scratch60/turk-browne/an633/nsd/COCO_73k_annots_curated.npy", type=str, 
            action='store')
    parser.add_argument(
            '--subj01_vitb32text_train_pred_clips_url', 
            default="/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_annots_curated.npy", type=str, 
            #default="/gpfs/milgram/scratch60/turk-browne/an633/nsd/COCO_73k_annots_curated.npy", type=str, 
            action='store')
    parser.add_argument(
            '--subj01_vitb32image_train_pred_clips_url', 
            default="/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_annots_curated.npy", type=str, 
            #default="/gpfs/milgram/scratch60/turk-browne/an633/nsd/COCO_73k_annots_curated.npy", type=str, 
            action='store')
    parser.add_argument(
            '--subj01_vitb32text_test_pred_clips_url', 
            default="/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_annots_curated.npy", type=str, 
            #default="/gpfs/milgram/scratch60/turk-browne/an633/nsd/COCO_73k_annots_curated.npy", type=str, 
            action='store')
    parser.add_argument(
            '--subj01_vitb32image_test_pred_clips_url', 
            default="/scratch/gpfs/KNORMAN/nsdgeneral_hdf5/COCO_73k_annots_curated.npy", type=str, 
            #default="/gpfs/milgram/scratch60/turk-browne/an633/nsd/COCO_73k_annots_curated.npy", type=str, 
            action='store')
    
    
#     parser.add_argument(
#             '--random_coefs', 
#             default=False, 
#             type=lambda x: (str(x).lower() in ['true','1', 'yes']))
   
    
    parser.add_argument('-d', '--debug', help="in debug mode or not", 
                        action='store_true')

    return parser


    
def subprocess_fn(rank, args, temp_dir, train_params, data_params, model_optim_params, save_params):
    print("args.run_dir", args.run_dir)
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        print("os.name", os.name)
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    torch_utils.training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    # training.training_loop.training_loop(rank=rank, **vars(args))
    training.trainer.training_loop(rank=rank, train_params=train_params, data_params=data_params, 
                                   model_optim_params=model_optim_params, save_params=save_params)
    
#     train_params.rank = rank
#     trainer = training.trainer.MindReader_Trainer(train_params = train_params,
#                                                  data_params = data_params,
#                                                  model_optim_params = model_optim_params)
#     trainer.train()
    
def main(args, **config_kwargs):
    """Train a GAN using the techniques described in the paper
    "Training Generative Adversarial Networks with Limited Data".
    Examples:
    \b
    # Train with custom dataset using 1 GPU.
    python train.py --outdir=~/training-runs --data=~/mydataset.zip --gpus=1
    \b
    # Train class-conditional CIFAR-10 using 2 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/cifar10.zip \\
        --gpus=2 --cfg=cifar --cond=1
    \b
    # Transfer learn MetFaces from FFHQ using 4 GPUs.
    python train.py --outdir=~/training-runs --data=~/datasets/metfaces.zip \\
        --gpus=4 --cfg=paper1024 --mirror=1 --resume=ffhq1024 --snap=10
    \b
    # Reproduce original StyleGAN2 config F.
    python train.py --outdir=~/training-runs --data=~/datasets/ffhq.zip \\
        --gpus=8 --cfg=stylegan2 --mirror=1 --aug=noaug
    \b
    Base configs (--cfg):
      auto       Automatically select reasonable defaults based on resolution
                 and GPU count. Good starting point for new datasets.
      stylegan2  Reproduce results for StyleGAN2 config F at 1024x1024.
      paper256   Reproduce results for FFHQ and LSUN Cat at 256x256.
      paper512   Reproduce results for BreCaHAD and AFHQ at 512x512.
      paper1024  Reproduce results for MetFaces at 1024x1024.
      cifar      Reproduce results for CIFAR-10 at 32x32.
    \b
    Transfer learning source networks (--resume):
      ffhq256        FFHQ trained at 256x256 resolution.
      ffhq512        FFHQ trained at 512x512 resolution.
      ffhq1024       FFHQ trained at 1024x1024 resolution.
      celebahq256    CelebA-HQ trained at 256x256 resolution.
      lsundog256     LSUN Dog trained at 256x256 resolution.
      <PATH or URL>  Custom network pickle.
    """
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
#     try:
#         run_desc, args = utils.setup_training_loop_kwargs(**config_kwargs)
#     except utils.UserError as err:
#         ctx.fail(err)

#     # Pick output directory.
#     prev_run_dirs = []
#     if os.path.isdir(outdir):
#         prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
#     prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
#     prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
#     cur_run_id = max(prev_run_ids, default=-1) + 1
#     args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
#     assert not os.path.exists(args.run_dir)

#     # Print options.
#     print()
#     print('Training options:')
#     print(json.dumps(args, indent=2))
#     print()
#     print(f'Output directory:   {args.run_dir}')
#     print(f'Training data:      {args.training_set_kwargs.path}')
#     print(f'Training duration:  {args.total_kimg} kimg')
#     print(f'Number of GPUs:     {args.num_gpus}')
#     print(f'Number of images:   {args.training_set_kwargs.max_size}')
#     print(f'Image resolution:   {args.training_set_kwargs.resolution}')
#     print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
#     print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
#     print(f'Discriminator use normalization:  {args.d_use_norm}')
#     print(f'Discriminator use fts: {args.d_use_fts}')

#     # Dry run?
#     if dry_run:
#         print('Dry run; exiting.')
#         return

#     # Create output directory.
#     print('Creating output directory...')
#     os.makedirs(args.run_dir)
#     with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
#         json.dump(args, f, indent=2)
    
    # default params
    defaults = dnnlib.EasyDict(
        fmaps = 1,
        m_layer_features = 1024,
        mb = 16 * args.num_gpus, # max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay 
        img_resolution = 256
    )
    # Get train parameters
    train_params = dnnlib.EasyDict(batch_size = args.batch_size, 
                        num_train_epochs = args.num_train_epochs,
                        num_gpus = args.num_gpus,
                        random_seed = 42,
                        cudnn_benchmark = True,
                        allow_tf32 = False,
                        style_mixing_prob = 0.9,
                        contrastive_params = dnnlib.EasyDict(
                            temp = 0.5,
                            lam = 0.0
                        ),
                        lambda_1 = 5,
                        lambda_2 = 10,
                        lambda_3 = 10,
                        pl_batch_shrink = 2,
                        pl_decay = 0.01,
                        pl_weight = 2,
                        r1_gamma = 0.0002 * (defaults.img_resolution ** 2) / defaults.mb, # heuristic formula,
                        ema_kwargs = dnnlib.EasyDict(
                           ema_kimg = defaults.mb * 10 / 32,
                           ema_rampup = 0.05,
                        ),
                        metrics = ['fid50k_full']
                    )
    
    # Get data parameters
    data_params = dnnlib.EasyDict(
                        dataset_type = "avg_split",
                        train_url = args.train_url,
                        train_indices_batchidx_dict_url = args.train_indices_batchidx_dict_url,
                        val_url = args.val_url,
                        val_indices_batchidx_dict_url = args.val_indices_batchidx_dict_url, 
                        subj01_vitb32text_train_pred_clips_url = args.subj01_vitb32text_train_pred_clips_url,
                        subj01_vitb32image_train_pred_clips_url = args.subj01_vitb32image_train_pred_clips_url,
                        subj01_vitb32text_test_pred_clips_url = args.subj01_vitb32text_test_pred_clips_url,
                        subj01_vitb32image_test_pred_clips_url = args.subj01_vitb32image_test_pred_clips_url,
                        subjectorder_url = args.subjectorder_url,
                        annotations_url = args.annotations_url,
                        # default augmentation pipeline is 'bgc'
                        # look at augpipe_specs in utils
                        augment_kwargs = dnnlib.EasyDict(
                            xflip=1, 
                            rotate90=1, 
                            xint=1, 
                            scale=1, 
                            rotate=1, 
                            aniso=1, 
                            xfrac=1, 
                            brightness=1, 
                            contrast=1, 
                            lumaflip=1,
                            hue=1, 
                            saturation=1
                        ),
                        augment_p = 0.0, # AFAICT, no augmentations are done in conditional image generation
                        ada_target = None,
    )
    
    # Get model and optimizer parameters
    
    model_optim_params = dnnlib.EasyDict(
                        emb_shape = 512,
                        clip_model_name = "ViT-B/32",
                        G_kwargs = dnnlib.EasyDict(
                            z_dim=512, 
                            c_dim=0,
                            w_dim=512,
                            img_resolution=defaults.img_resolution,
                            img_channels=3,
                            m_layer_features=defaults.m_layer_features, 
                            mapping_kwargs=dnnlib.EasyDict(
                                num_layers = 8
                            ),
                            synthesis_kwargs=dnnlib.EasyDict(
                                structure=args.structure,
                                channel_base = int(defaults.fmaps * 32768),
                                channel_max = 512,
                                num_fp16_res = 4,
                                conv_clamp = 256,
                                change = 256,
                                f_dim = 512,
                                f_dim2 = 512
                            ),
                             
                        ),
                        D_kwargs = dnnlib.EasyDict(
                            c_dim=0,
                            img_resolution=defaults.img_resolution,
                            img_channels=3,
                            use_norm=False,
                            use_fts=True, 
                            block_kwargs=dnnlib.EasyDict(), 
                            mapping_kwargs=dnnlib.EasyDict(),
                            channel_base = int(defaults.fmaps * 32768),
                            num_fp16_res = 4,
                            conv_clamp = 256,
                            epilogue_kwargs = dnnlib.EasyDict(
                                structure=args.structure,
                                mbstd_group_size = min(defaults.mb // args.num_gpus, 4),
                                f_dim = 512,
                                f_dim2 = 512
                            ),
                             
                        ),
                        opt_kwargs = dnnlib.EasyDict(
                                G_reg_interval = 4,
                                D_reg_interval = 16,
                                G_lrate = 0.0025,
                                D_lrate = 0.0025,
                                betas = [0,0.99], 
                                eps=1e-8
                            )
                        
    )

    # 
    save_params = dnnlib.EasyDict(
        save_dir = args.run_dir,
        exp_name = f"{args.model_name}" \
                + f"_rep_{datetime.datetime.now().timestamp()}.pth.tar"
    )
    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            print("sub_process_fn run directly")
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir, train_params=train_params, 
                          data_params=data_params, model_optim_params=model_optim_params, save_params=save_params)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir, train_params, data_params, model_optim_params, save_params), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    main(args) # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
