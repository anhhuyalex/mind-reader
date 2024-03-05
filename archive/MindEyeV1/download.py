from huggingface_hub import snapshot_download
snapshot_download(repo_id="pscotti/mindeyev2", repo_type = "dataset", revision="main", cache_dir = "./cache" ,
    local_dir= "/scratch/gpfs/KNORMAN/mindeyev2", local_dir_use_symlinks = False, resume_download = True)