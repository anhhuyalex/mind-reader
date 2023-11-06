# from huggingface_hub import snapshot_download
# # snapshot_download(repo_id="pscotti/mindeyev2", repo_type = "dataset", revision="main", cache_dir = "./cache" ,
#     # local_dir= "/scratch/gpfs/KNORMAN/mindeyev2", local_dir_use_symlinks = False, resume_download = True)
# snapshot_download(repo_id="shi-labs/versatile-diffusion", repo_type = "model", revision="main", cache_dir = "./cache" ,
#     local_dir= "/scratch/gpfs/KNORMAN/versatile-diffusion", local_dir_use_symlinks = False, resume_download = True)

# from models import Clipper
import torch
# local_rank = 0
# eva02_model = Clipper("ViT-L/14", device=torch.device(f"cpu"), hidden_state=True, norm_embs=True)

from diffusers import AutoencoderKL
device=torch.device(f"cpu")
autoenc = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, cache_dir="./cache")
# autoenc.load_state_dict(torch.load('../train_logs/sdxl_vae_normed/best.pth')["model_state_dict"])
autoenc.eval()
autoenc.requires_grad_(False)
autoenc.to(device)
utils.count_params(autoenc)