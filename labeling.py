import os
from pathlib import Path
from PIL import Image
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

# Paths
dataset_root = Path("C:/Users/20235050/Downloads/BDS_Y3/DC3/coral_bleaching/reef_support")
images_dir = dataset_root / "images"
masks_b_dir = dataset_root / "masks_bleached"
masks_nb_dir = dataset_root / "masks_non_bleached"

# Output
out_root = dataset_root / "yolo_classification"
train_dir = out_root / "train"
val_dir = out_root / "val"

for sub in ["CORAL", "CORAL_BL"]:
    (train_dir / sub).mkdir(parents=True, exist_ok=True)
    (val_dir / sub).mkdir(parents=True, exist_ok=True)

# Collect all samples
samples = []
for img_file in images_dir.glob("*.jpg"):
    base = img_file.stem
    mask_b_file = masks_b_dir / f"{base}_bleached.png"
    mask_nb_file = masks_nb_dir / f"{base}_non_bleached.png"
    if mask_b_file.exists() and mask_nb_file.exists():
        samples.append((img_file, mask_b_file, mask_nb_file))

# Train/val split
train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

def process_and_copy(samples, target_root):
    for img_file, mask_b_file, mask_nb_file in samples:
        base = img_file.stem

        # Load masks
        mask_b = np.array(Image.open(mask_b_file).convert("L")) > 127
        mask_nb = np.array(Image.open(mask_nb_file).convert("L")) > 127

        frac_b = mask_b.mean()
        frac_nb = mask_nb.mean()

        if frac_b > 0.2:
            label = "CORAL_BL"
        else:
            label = "CORAL"

        # Copy masked image we already generated
        masked_img_file = dataset_root / "images_preprocessed" / f"{base}_preprocessed.jpg"
        if masked_img_file.exists():
            shutil.copy(masked_img_file, target_root / label / masked_img_file.name)
        else:
            print(f"Missing masked image for {base}")

process_and_copy(train_samples, train_dir)
process_and_copy(val_samples, val_dir)

print("Dataset prepared at:", out_root)

