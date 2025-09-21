import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from skimage import measure

# Class order -> class IDs
CLASS_NAMES = ["bleached", "non_bleached"]

# Default naming: image IMG_001.jpg -> masks in:
# masks_bleached/IMG_001_bleached.png
# masks_non_bleached/IMG_001_non_bleached.png
DEFAULT_SUFFIX_BLEACHED = "_bleached"
DEFAULT_SUFFIX_NON = "_non_bleached"

ALLOWED_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# ---------------- Default configuration ----------------
ROOT = Path("C:/Users/20235050/Downloads/BDS_Y3/DC3/coral_bleaching/reef_support")  # change to your dataset path
SPLITS = ["train", "val"]
BLEACHED_DIR = ROOT / "masks_bleached"
NONBLEACHED_DIR = ROOT / "masks_non_bleached"
SUFFIX_BLEACHED = DEFAULT_SUFFIX_BLEACHED
SUFFIX_NON = DEFAULT_SUFFIX_NON
MIN_AREA_PX = 20 # minimum area of polygon to keep (in pixels)
EPS_FRAC = 0.001 # (fraction of contour perimeter, smaller = more accurate)
THRESH = 127
RESOLVE_OVERLAP = True
INVERT = False
VERBOSE = True
# -------------------------------------------------------

def load_mask_any(path: Path) -> np.ndarray | None:
    if not path or not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 4:
            return img[:, :, 3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return None

def find_mask_by_suffix(msk_dir: Path, stem: str, suffix: str) -> Path | None:
    for ext in ALLOWED_EXTS:
        p = msk_dir / f"{stem}{suffix}{ext}"
        if p.exists():
            return p
    cands = sorted(msk_dir.glob(f"{stem}{suffix}*"))
    for c in cands:
        if c.is_file() and c.suffix.lower() in ALLOWED_EXTS:
            return c
    return None

def mask_to_polygons(mask01: np.ndarray, min_area_px: int, eps_frac: float) -> list[np.ndarray]:
    H, W = mask01.shape[:2]
    m = (mask01 > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px:
            continue
        if eps_frac > 0:
            eps = eps_frac * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, eps, True)
        if len(cnt) >= 3:
            pts = cnt.reshape(-1, 2).astype(np.float32)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            polys.append(pts)
    return polys

def watershed_instance_mask(mask: np.ndarray) -> np.ndarray:
    """
    Applies watershed to a binary mask to separate overlapping objects.
    Args:
        mask: np.ndarray, binary mask (H, W), 1=object, 0=background
    Returns:
        instance_mask: np.ndarray, label image where 0=background, 1,2,...=instances
    """
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Sure foreground
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 1, 0)
    sure_fg = np.uint8(sure_fg)
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labeling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 1] = 0
    # Watershed
    img3 = np.stack([mask*255]*3, axis=-1).astype(np.uint8)
    cv2.watershed(img3, markers)
    instance_mask = np.where(markers > 1, markers-1, 0).astype(np.uint16)
    return instance_mask

def write_yolo_label(label_path: Path, polys_by_class: list[list[np.ndarray]], H: int, W: int, save_empty=True):
    lines = []
    for cls_id, polys in enumerate(polys_by_class):
        for pts in polys:
            norm = pts.copy()
            norm[:, 0] /= W
            norm[:, 1] /= H
            flat = norm.flatten().tolist()
            if len(flat) >= 6:
                lines.append(str(cls_id) + " " + " ".join(f"{v:.6f}" for v in flat))
    if lines or save_empty:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")

def process_split(root: Path, split: str, bleached_dir: Path, non_dir: Path,
                  suffix_bleached: str, suffix_non: str,
                  min_area_px: int, eps_frac: float, thresh: int,
                  resolve_overlap: bool, invert: bool, verbose: bool):
    img_dir = root / "yolo_seg/images" / split
    lbl_dir = root / "yolo_seg/labels" / split
    assert img_dir.exists(), f"Missing images dir: {img_dir}"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    assert imgs, f"No images found in {img_dir}"

    found_b = found_nb = 0
    wrote = 0

    for img_path in tqdm(imgs, desc=f"{split}: images"):
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            if verbose:
                print(f"Cannot read image {img_path}, skipping")
            continue
        H, W = img.shape[:2]

        p_b = find_mask_by_suffix(bleached_dir, stem, suffix_bleached)
        p_nb = find_mask_by_suffix(non_dir, stem, suffix_non)

        m_b = load_mask_any(p_b) if p_b else None
        m_nb = load_mask_any(p_nb) if p_nb else None

        if m_b is not None:
            found_b += 1
            if m_b.shape[:2] != (H, W):
                m_b = cv2.resize(m_b, (W, H), interpolation=cv2.INTER_NEAREST)
            m_b = (m_b > thresh).astype(np.uint8)
            if invert:
                m_b = 1 - m_b

        if m_nb is not None:
            found_nb += 1
            if m_nb.shape[:2] != (H, W):
                m_nb = cv2.resize(m_nb, (W, H), interpolation=cv2.INTER_NEAREST)
            m_nb = (m_nb > thresh).astype(np.uint8)
            if invert:
                m_nb = 1 - m_nb

        if resolve_overlap and (m_b is not None) and (m_nb is not None):
            m_nb = (m_nb & (1 - m_b)).astype(np.uint8)

        # --- Watershed instance separation for each mask ---
        polys_by_class: list[list[np.ndarray]] = [[] for _ in CLASS_NAMES]
        for mask, class_idx in zip([m_b, m_nb], [0, 1]):
            if mask is not None and mask.any():
                instance_mask = watershed_instance_mask(mask)
                for inst_id in np.unique(instance_mask):
                    if inst_id == 0:
                        continue # background
                    polylist = mask_to_polygons((instance_mask==inst_id).astype(np.uint8), min_area_px, eps_frac)
                    polys_by_class[class_idx].extend(polylist)

        label_path = lbl_dir / f"{stem}.txt"
        write_yolo_label(label_path, polys_by_class, H, W, save_empty=True)
        wrote += 1

        if verbose:
            n0 = len(polys_by_class[0])
            n1 = len(polys_by_class[1])
            print(f"{img_path.name}: "
                  f"bleached_mask={'OK' if m_b is not None else 'MISSING'} "
                  f"non_bleached_mask={'OK' if m_nb is not None else 'MISSING'} "
                  f"| polygons: bleached={n0}, non_bleached={n1}")

    print(f"{split}: wrote {wrote} label files to {lbl_dir}")
    print(f"{split}: found masks — bleached: {found_b}/{len(imgs)}, non_bleached: {found_nb}/{len(imgs)}")

def visualize_polygons_on_image(img_path: Path, bleached_mask_path: Path, non_bleached_mask_path: Path, min_area_px: int, eps_frac: float, thresh: int, invert: bool = False, save_path: Path = None):
    """
    Overlay polygons extracted from masks on the original image and display/save the result.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot read image {img_path}")
        return
    H, W = img.shape[:2]
    m_b = load_mask_any(bleached_mask_path) if bleached_mask_path else None
    m_nb = load_mask_any(non_bleached_mask_path) if non_bleached_mask_path else None
    if m_b is not None:
        if m_b.shape[:2] != (H, W):
            m_b = cv2.resize(m_b, (W, H), interpolation=cv2.INTER_NEAREST)
        m_b = (m_b > thresh).astype(np.uint8)
        if invert:
            m_b = 1 - m_b
    if m_nb is not None:
        if m_nb.shape[:2] != (H, W):
            m_nb = cv2.resize(m_nb, (W, H), interpolation=cv2.INTER_NEAREST)
        m_nb = (m_nb > thresh).astype(np.uint8)
        if invert:
            m_nb = 1 - m_nb
    # --- Watershed instance separation for visualization ---
    polys_by_class = [[] for _ in CLASS_NAMES]
    for mask, class_idx in zip([m_b, m_nb], [0, 1]):
        if mask is not None and mask.any():
            instance_mask = watershed_instance_mask(mask)
            for inst_id in np.unique(instance_mask):
                if inst_id == 0:
                    continue
                polylist = mask_to_polygons((instance_mask==inst_id).astype(np.uint8), min_area_px, eps_frac)
                polys_by_class[class_idx].extend(polylist)
    # Draw polygons
    overlay = img.copy()
    colors = [(0, 0, 255), (0, 255, 0)]  # BGR: red for bleached, green for non-bleached
    for cls_id, polys in enumerate(polys_by_class):
        for pts in polys:
            pts_int = np.round(pts).astype(np.int32)
            cv2.polylines(overlay, [pts_int], isClosed=True, color=colors[cls_id], thickness=2)
            cv2.fillPoly(overlay, [pts_int], color=(*colors[cls_id], 60))  # semi-transparent fill
    # Blend overlay with original image
    vis = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Polygons overlay: {img_path.name}")
    plt.axis('off')
    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert masks to YOLO polygons and optionally visualize.")
    parser.add_argument('--visualize', action='store_true', help='Visualize polygons overlay for a single image')
    parser.add_argument('--img', type=str, help='Path to image to visualize')
    parser.add_argument('--bleached_mask', type=str, help='Path to bleached mask')
    parser.add_argument('--nonbleached_mask', type=str, help='Path to non-bleached mask')
    parser.add_argument('--save', type=str, help='Path to save visualization (optional)')
    args = parser.parse_args()

    if args.visualize:
        assert args.img, "--img is required for visualization"
        visualize_polygons_on_image(
            Path(args.img),
            Path(args.bleached_mask) if args.bleached_mask else None,
            Path(args.nonbleached_mask) if args.nonbleached_mask else None,
            MIN_AREA_PX, EPS_FRAC, THRESH, INVERT,
            Path(args.save) if args.save else None
        )
        return

    assert (ROOT / "images").exists(), f"Missing images folder: {ROOT/'images'}"
    assert BLEACHED_DIR.exists(), f"Missing masks folder: {BLEACHED_DIR}"
    assert NONBLEACHED_DIR.exists(), f"Missing masks folder: {NONBLEACHED_DIR}"

    for split in SPLITS:
        process_split(ROOT, split, BLEACHED_DIR, NONBLEACHED_DIR,
                      SUFFIX_BLEACHED, SUFFIX_NON,
                      MIN_AREA_PX, EPS_FRAC, THRESH,
                      RESOLVE_OVERLAP, INVERT, VERBOSE)


if __name__ == "__main__":
    main()