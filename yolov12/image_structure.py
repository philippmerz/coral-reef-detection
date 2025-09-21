import random
from pathlib import Path
import shutil

# ==== CONFIGURATION (change if needed) ====
ROOT = Path(r"C:/Users/20235050/Downloads/BDS_Y3/DC3/coral_bleaching/reef_support")
VAL_RATIO = 0.2
SEED = 42
COPY = True 
# ==========================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    images_root = ROOT / "images"
    assert images_root.exists(), f"Missing folder: {images_root}"

    # Collect images (flat only, not recursive)
    imgs = sorted([p for p in images_root.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    assert imgs, f"No images found in {images_root}"

    random.Random(SEED).shuffle(imgs)
    n_val = int(len(imgs) * VAL_RATIO)
    val_imgs = set(imgs[:n_val])

    out_train = ROOT / "yolo_seg/images/train"
    out_val = ROOT / "yolo_seg/images/val"
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    op = shutil.copy2 if COPY else shutil.move

    for p in imgs:
        dst_dir = out_val if p in val_imgs else out_train
        dst = dst_dir / p.name
        if dst.exists():
            continue
        op(str(p), str(dst))

    print(f"Split complete. Train: {len(imgs)-n_val}, Val: {n_val}")
    print(f"Train dir: {out_train}")
    print(f"Val dir:   {out_val}")


if __name__ == "__main__":
    main()
