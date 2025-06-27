# !pip install huggingface_hub hf_transfer
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

print("Starting download...")

# snapshot_download(
#     repo_id = "hustvl/Vim-tiny-midclstok",
#     local_dir = "hustvl/Vim-tiny-midclstok",
# )

snapshot_download(
    repo_id = "hustvl/Vim-small-midclstok",
    local_dir = "hustvl/Vim-small-midclstok",
)

print("Download finished.")