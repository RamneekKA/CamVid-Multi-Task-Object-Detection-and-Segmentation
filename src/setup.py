import os
import urllib.request

# Specify the target directory directly as pytorch_helper
target_dir = "pytorch_helper"  # This will be created in your current directory

# Create the directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"Created directory: {target_dir}")

required_files = [
    "engine.py",
    "utils.py",
    "coco_utils.py",
    "coco_eval.py",
    "transforms.py"
]

for file in required_files:
    # Construct the full file path with the target directory
    file_path = os.path.join(target_dir, file)
    if not os.path.exists(file_path):
        url = f"https://raw.githubusercontent.com/pytorch/vision/main/references/detection/{file}"
        print(f"Downloading {file} to {target_dir}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded {file} successfully.")
    else:
        print(f"File {file} already exists in {target_dir}")