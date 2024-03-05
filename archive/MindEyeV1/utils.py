from Lafite import dnnlib
from Lafite.metrics import metric_main

#----------------------------------------------------------------------------
class UserError(Exception):
    pass
    
#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    f_dim      = None,
    f_dim2     = None,
    cond_vec   = None,
    use_fmri   = False, # if True, carry out end to end training that trains both mapper and GAN
    fmri_len   = 15744,
    structure  = 2,
    enabled_forced_map = False, # whether to copy the same weights to different pre_0 and pre_1 branches or not
    resloss    = False,
    d_use_norm = None, # normalize the feature extracted by discriminator or not
    d_use_fts  = None, # discriminator extract semantic feature or not
    mixing_prob= None, # mixing probability of ground-truth and language-free generated pairs, mixing_prob=0 means only use ground-truth, mixing_prob=1. means using only pseudo pairs(language-free)
    lam        = None, # hyper-parameter for contrastive loss
    temp       = None, # hyper-parameter for contrastive loss
    change     = None, # hyper-parameter for architecture
    map_num    = None, # hyper-parameter for architecture
    gather     = None, # hyper-parameter for contrastive loss
    itd        = None, # hyper-parameter for contrastive loss
    itc        = None, # hyper-parameter for contrastive loss
    iid        = None, # hyper-parameter for contrastive loss
    iic        = None, # hyper-parameter for contrastive loss
    ires       = None,   # hyper-parameter for resnet vector contrastive loss
    metric_only_test = None, # hyper-parameter for computing metrics
    fmap       = None, # hyper-parameter for architecture, related to channel number
    ratio      = None,
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    seed       = None, # Random seed: <int>, default = 0
    # Dataset.
    data       = None, # Training dataset (required): <path>
    test_data  = None, # Testing dataset for metrics, if not use training dataset
    cond       = None, # Train conditional model based on dataset labels: <bool>, default = False
    subset     = None, # Train with only N images: <int>, default = all
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
    gamma      = None, # Override R1 gamma: <float>
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>

    # Discriminator augmentation.
    aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
    p          = None, # Specify p for 'fixed' (required): <float>
    target     = None, # Override ADA target for 'ada': <float>, default = depends on aug
    augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------
    if f_dim is None:
        f_dim = 512
    assert isinstance(f_dim, int)
    args.f_dim = f_dim

    if f_dim2 is None:
        f_dim2 = 0
    assert isinstance(f_dim2, int)
    args.f_dim2 = f_dim2

    if resloss is None:
        resloss = False
    assert isinstance(resloss, bool)
    args.resloss = resloss

    if structure is None:
        structure = 2
    assert isinstance(structure, int)
    args.structure = structure

    if use_fmri is None:
        use_fmri = False
    assert isinstance(use_fmri, bool)
    args.use_fmri = use_fmri

    if fmri_len is None:
        fmri_len = 15744
    assert isinstance(fmri_len, int)

    if enabled_forced_map is None:
        enabled_forced_map = False
    assert isinstance(enabled_forced_map, bool)
    args.enabled_forced_map = enabled_forced_map

    if ratio is None:
        ratio = 1.0
    args.ratio = ratio
    
    if mixing_prob is None:
        mixing_prob = 0.
    args.mixing_prob = mixing_prob
    
    if fmap is None:
        fmap = 1.
    
    if metric_only_test is None:
        metric_only_test = False
    args.metric_only_test = metric_only_test
    
    if map_num is None:
        map_num = 8
        
    if lam is None:
        lam = 0.
    args.lam = lam
    
    if temp is None:
        temp = 0.5
    args.temp = temp
    
    if itd is None:
        itd = 10.
    args.itd = itd
    if itc is None:
        itc = 10.
    args.itc = itc
    
    if iid is None:
        iid = 0.
    args.iid = iid
    if iic is None:
        iic = 0.
    args.iic = iic
    if ires is None:
        ires = 10.
    args.ires = ires

    if change is None:
        change = 256
    
    if d_use_norm is None:
        d_use_norm = False
    assert isinstance(d_use_norm, bool)
    args.d_use_norm = d_use_norm
    
    if d_use_fts is None:
        d_use_fts = True
    args.d_use_fts = d_use_fts
    
    if gather is None:
        gather = False
    args.gather = gather
    
    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    if not all(metric_main.is_valid_metric(metric) for metric in metrics):
        raise UserError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    args.metrics = metrics

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    assert data is not None
    assert isinstance(data, str)
    print('using data: ', data, 'testing data: ', test_data)
    if test_data is None:
        test_data = data
    # args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False, use_clip=True, ratio=args.ratio)
    # args.testing_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=test_data, use_labels=True, max_size=None, xflip=False, use_clip=True, ratio=1.0)
    args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.NsdClipDataset', path=data, use_mapped=cond_vec, use_fmri=args.use_fmri, fmri_pad=15744, use_clip=True, threshold=1.5, normalize_clip=True, use_labels=True, max_size=None, xflip=False, ratio=args.ratio)
    args.testing_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.NsdClipDataset', path=test_data, use_mapped=cond_vec, use_fmri=args.use_fmri, fmri_pad=15744, use_clip=True, threshold=1.5, normalize_clip=True, use_labels=True, max_size=None, xflip=False, ratio=1.0)

    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=False, num_workers=1, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        desc = training_set.name
        args.testing_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.testing_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        del training_set # conserve memory

    except IOError as err:
        raise UserError(f'--data: {err}')

    if cond is None:
        cond = False
    assert isinstance(cond, bool)
    if cond:
        if not args.training_set_kwargs.use_labels:
            raise UserError('--cond=True requires labels specified in dataset.json')
        desc += '-cond'
    else:
        args.training_set_kwargs.use_labels = False
        args.testing_set_kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= args.training_set_kwargs.max_size:
            raise UserError(f'--subset must be between 1 and {args.training_set_kwargs.max_size}')
        desc += f'-subset{subset}'
        if subset < args.training_set_kwargs.max_size:
            args.training_set_kwargs.max_size = subset
            args.training_set_kwargs.random_seed = args.random_seed

    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True
        args.testing_set_kwargs.xflip = True

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}-lam{lam:g}-temp{temp:g}-map_num{map_num:g}'

    cfg_specs = {# Populated dynamically based on resolution and GPU count.
        'auto': dict(ref_gpus=-1, kimg=25000, mb=-1, mbstd=-1, fmaps=-1, lrate=-1, gamma=1., ema=-1, ramp=0.05, map=map_num),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'-gpus{gpus:d}'
        spec.ref_gpus = gpus
        res = args.training_set_kwargs.resolution
        # spec.mb = 8 * gpus
        spec.mb = 16 * gpus # max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else fmap
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        # spec.lrate = spec.lrate * 0.9
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32
        
    # args.M_kwargs = dnnlib.EasyDict(class_name='training.networks.ManiNetwork', z_dim=args.f_dim,  layer_features=args.f_dim, w_dim=512, num_layers=8)
    m_layer_features = args.f_dim + args.f_dim2 if args.structure == 4 else args.f_dim
    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512,
        m_layer_features=m_layer_features, m_num_layers=8, mapping_kwargs=dnnlib.EasyDict(),
        synthesis_kwargs=dnnlib.EasyDict(structure=args.structure))
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator', use_norm=args.d_use_norm,
        use_fts=args.d_use_fts, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(),
        epilogue_kwargs=dnnlib.EasyDict(structure=args.structure))
    if args.use_fmri:
        args.mapper_kwargs = dnnlib.EasyDict(class_name='training.networks.FmriVecMapper', fmri_len=fmri_len, f_dim=args.f_dim)
        if args.structure in [4, 5]:
            # used for resnet vec, hardcoded params
            args.mapper2_kwargs = dnnlib.EasyDict(class_name='training.networks.FmriVecMapper', fmri_len=fmri_len, f_dim=args.f_dim2) # fmri_clip
            # args.mapper2_kwargs = dnnlib.EasyDict(class_name='training.networks.FmriVecMapper', fmri_len=fmri_len, f_dim=args.f_dim2, num_layers=2, fc_hdim=256, last_activation=True) # resnet

    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.G_kwargs.synthesis_kwargs.change = change
    args.G_kwargs.synthesis_kwargs.f_dim = args.f_dim
    args.G_kwargs.synthesis_kwargs.f_dim2 = args.f_dim2
    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
    args.D_kwargs.epilogue_kwargs.f_dim = args.f_dim
    args.D_kwargs.epilogue_kwargs.f_dim2 = args.f_dim2
    
    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    if args.use_fmri:
        args.fmri_vec_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
        if args.structure in [4, 5]:
            args.fmri_vec2_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', r1_gamma=spec.gamma, use_fmri=args.use_fmri,
        resloss=args.resloss, ires=args.ires)

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0 # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0 # disable style mixing
        args.D_kwargs.architecture = 'orig' # disable residual skip connections

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug is None:
        aug = 'noaug' # no augmentation is used in our experiments
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.ada_target = 0.6

    elif aug == 'noaug':
        pass

    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')

    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        assert isinstance(target, float)
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, imgfilter=1, noise=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    elif resume in resume_specs:
        desc += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume] # predefined url
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args

#----------------------------------------------------------------------------
augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, imgfilter=1, noise=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }