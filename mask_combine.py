import os
from pathlib import Path
from PIL import Image
import numpy as np

# Paths
dataset_root = Path("C:/Users/20235050/Downloads/BDS_Y3/DC3/coral_bleaching/reef_support")
images_dir = dataset_root / "images"
masks_b_dir = dataset_root / "masks_bleached"
masks_nb_dir = dataset_root / "masks_non_bleached"
out_dir = dataset_root / "images_preprocessed"
out_dir.mkdir(exist_ok=True)

# Process each image
for img_file in images_dir.glob("*.jpg"):
    base = img_file.stem  # e.g. i_1

    # Load image
    image = Image.open(img_file).convert("RGB")
    img_arr = np.array(image)

    # Load masks
    mask_b_file = masks_b_dir / f"{base}_bleached.png"
    mask_nb_file = masks_nb_dir / f"{base}_non_bleached.png"

    if not mask_b_file.exists() or not mask_nb_file.exists():
        print(f"⚠️ Missing masks for {base}, skipping")
        continue

    mask_b = np.array(Image.open(mask_b_file).convert("L")) > 127
    mask_nb = np.array(Image.open(mask_nb_file).convert("L")) > 127

    # Combine masks (coral = bleached OR non-bleached)
    coral_mask = mask_b | mask_nb

    # Apply mask (everything else black)
    masked_img = np.zeros_like(img_arr)
    masked_img[coral_mask] = img_arr[coral_mask]

    # Save result
    out_path = out_dir / f"{base}_preprocessed.jpg"
    Image.fromarray(masked_img).save(out_path)

    print(f"Saved {out_path}")
