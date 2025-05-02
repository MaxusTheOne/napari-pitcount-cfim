import torch
import importlib.resources as pkg_resources

pkg_root = pkg_resources.files("src")
# IMPORTANT: Use map_location='cpu' to avoid GPU loading issues
checkpoint_path = pkg_root.joinpath("resources","models",'Fluor_Pit_v1_1_checkpoint.pth.tar')

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint loaded successfully.")
    print(f"Keys: {list(checkpoint.keys())}")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
