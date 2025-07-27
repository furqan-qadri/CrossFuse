# # Create targeted download script
# from huggingface_hub import snapshot_download
# import os

# print("Downloading KAIST set00/V000 folder specifically...")

# # Download only the V000 folder
# snapshot_download(
#     repo_id="richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset",
#     repo_type="dataset",
#     local_dir="./kaist_dataset",
#     allow_patterns=["kaist_train/set00/V000/**"],  # Only V000 folder
#     cache_dir="./hf_cache"
# )

# print("Download completed!")
# print("Check: ./kaist_dataset/kaist_train/set00/V000/")


"""
Simple script to download VOT-RGBT dataset from Hugging Face
"""

from datasets import load_dataset

print("Downloading VOT-RGBT dataset from Hugging Face...")

# Download the dataset
dataset = load_dataset("langutang/vot-rgbt-2019", 
                      download_mode="force_redownload")


print("Dataset downloaded successfully!")
print("Dataset info:")
print(dataset)

# Show basic structure
if hasattr(dataset, 'keys'):
    print(f"Available splits: {list(dataset.keys())}")

# Show a sample
if 'train' in dataset:
    sample = dataset['train'][0]
    print(f"Sample keys: {sample.keys()}")
    print("Dataset is ready to use!")
else:
    # Check what splits exist
    first_split = list(dataset.keys())[0]
    sample = dataset[first_split][0]
    print(f"Sample keys: {sample.keys()}")
    print("Dataset is ready to use!")