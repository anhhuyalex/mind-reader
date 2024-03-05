from huggingface_hub import snapshot_download
import torch
from diffusers import VersatileDiffusionPipeline, UniPCMultistepScheduler
import diffusers
print (diffusers.__version__)
# snapshot_download(repo_id="pscotti/mindeyev2", repo_type = "dataset", revision="main", cache_dir = "./cache" ,
#     local_dir= "/scratch/gpfs/KNORMAN/mindeyev2", local_dir_use_symlinks = False, resume_download = True)
# vd_pipe = VersatileDiffusionPipeline.from_pretrained("shi-labs/versatile-diffusion", torch_dtype=torch.float16, cache_dir="./cache")
# vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained("shi-labs/versatile-diffusion", subfolder="scheduler", cache_dir="./cache")

snapshot_download(repo_id="shi-labs/versatile-diffusion", repo_type = "model", revision="main", cache_dir = "./cache" ,
    local_dir= "/fsx/proj-fmri/alexnguyen/mind-reader/MindEyeV2/src/versatile-diffusion", local_dir_use_symlinks = False, resume_download = True)

# # from models import Clipper
# import torch
# # local_rank = 0
# # eva02_model = Clipper("ViT-L/14", device=torch.device(f"cpu"), hidden_state=True, norm_embs=True)

# from diffusers import AutoencoderKL
# device=torch.device(f"cpu")
# autoenc = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, cache_dir="./cache")
# # autoenc.load_state_dict(torch.load('../train_logs/sdxl_vae_normed/best.pth')["model_state_dict"])
# autoenc.eval()
# autoenc.requires_grad_(False)
# autoenc.to(device)
# utils.count_params(autoenc)

# def pca_metrics(A_pre, A_post, B_post): 
#     B_pre = B_post
#     normalized_a_pre = A_pre / np.linalg.norm(A_pre)
#     normalized_b_pre = B_pre / np.linalg.norm(B_pre)
#     normalized_a_post = A_post / np.linalg.norm(A_post)
#     normalized_b_post = B_post / np.linalg.norm(B_post)
     
#     # compute normalized bisector: shared direction
#     bisector = (normalized_a_pre + normalized_b_pre)
#     bisector = bisector / np.linalg.norm(bisector)
#     shared_metric = np.dot(normalized_a_post, bisector) 

#     # compute normalized perpendicular: direction in the plane perpendicular to bisector
#     perpendicular = normalized_a_pre - normalized_b_pre
#     perpendicular = perpendicular / np.linalg.norm(perpendicular)
#     perpendicular_metric = np.dot(normalized_a_post, perpendicular) 

#     # compute normalized normal: direction normal to bisector and perpendicular but closest to the changed vector   
#     normal = normalized_a_post - get_projection_vector(bisector, normalized_a_post) - get_projection_vector(perpendicular, normalized_a_post)
#     assert np.linalg.norm(normal) > 1e-3, "normal is too small"
#     normal = normal / np.linalg.norm(normal)
#     normal_metric = np.dot(normalized_a_post, normal) 

#     # assert that the three directions are orthonormal
#     assert np.allclose(np.dot(bisector, perpendicular), 0), "bisector and perpendicular are not orthogonal"
#     assert np.allclose(np.dot(bisector, normal), 0), "bisector and normal are not orthogonal"
#     assert np.allclose(np.dot(perpendicular, normal), 0), "perpendicular and normal are not orthogonal" 
#     assert np.allclose(np.linalg.norm(bisector), 1), "bisector is not normalized" 
#     assert np.allclose(np.linalg.norm(perpendicular), 1), "perpendicular is not normalized" 
#     assert np.allclose(np.linalg.norm(normal), 1), "normal is not normalized" 
    
#     return shared_metric, perpendicular_metric, normal_metric