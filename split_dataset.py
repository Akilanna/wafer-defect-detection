import os
import shutil
import random

source_path = os.path.expanduser("~/Workspace/Hackathons/Wafer_Inspection/data/processed")
target_path = os.path.expanduser("~/Workspace/Hackathons/Wafer_Inspection/data/final_dataset")

# Define the splits
splits = ['train', 'val', 'test']

# Filter: Only get directories from the source path
classes = [d for d in os.listdir(source_path) 
           if os.path.isdir(os.path.join(source_path, d))]

print(f"Detected classes: {classes}")

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(target_path, split, cls), exist_ok=True)

for cls in classes:
    class_dir = os.path.join(source_path, cls)
    files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
    random.shuffle(files)
    
    # Calculate split indices (80/10/10)
    train_end = int(len(files) * 0.8)
    val_end = train_end + int(len(files) * 0.1)
    
    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]
    
    # Move files to final_dataset structure
    for f in train_files:
        shutil.copy(os.path.join(class_dir, f), os.path.join(target_path, 'train', cls, f))
    for f in val_files:
        shutil.copy(os.path.join(class_dir, f), os.path.join(target_path, 'val', cls, f))
    for f in test_files:
        shutil.copy(os.path.join(class_dir, f), os.path.join(target_path, 'test', cls, f))

print(f"\nSuccess! Dataset split complete in {target_path}")
