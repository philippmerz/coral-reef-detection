from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import pandas as pd
from skimage.draw import polygon as sk_polygon
from tqdm import tqdm

# ==========================
# Configuration
# ==========================
CONFIG: Dict[str, Any] = {
    # Base CV results directory containing the three pipelines
    "base_cv_dir": "C:/Users/20235050/Downloads/BDS_Y3/DC3/yolov12/runs/cv",

    # Pipelines and folds to evaluate
    "pipelines": ["watershed+aug", "no_watershed+no_aug"],
    "folds": [1, 2, 3, 4, 5],


    # Validation dataset
    "images_dir": "C:/Users/20235050/Downloads/BDS_Y3/DC3/yolov12/dataset/images/val",
    "gt_dir":     "C:/Users/20235050/Downloads/BDS_Y3/DC3/yolov12/dataset/labels/val",

    # Where per-image prediction .txt files live for each fold
    # Path pattern (formatted with pipeline and fold)
    "pred_dir_pattern": "{base}/{pipeline}/fold_{fold}/predict_val/labels",

    # Class mapping and overlap priority
    # Your classes: 1 = bleached, 0 = non-bleached
    "class_map": {"1": "bleached", "0": "non-bleached"},
    # Priority high->low: if pixels overlap, higher priority wins
    "class_priority": ["bleached", "non-bleached"],  # equivalent to "1,0"

    # Outputs
    "output_dir": "C:/Users/20235050/Downloads/BDS_Y3/DC3/yolov12/runs/cv/pixelwise_eval_outputs",
    # Filenames
    "cv_summary_csv": "cv_summary.csv",
    "pipeline_summary_csv": "pipeline_summary.csv",
    # Per-image CSV filename pattern per (pipeline, fold)
    "per_image_csv_pattern": "{pipeline}_fold_{fold}_per_image.csv",
}


# ==========================
# Utilities
# ==========================

@dataclass
class Instance:
    polygon: np.ndarray  # shape (N, 2) in pixel coords
    class_id: str        # mapped class label string (e.g., "bleached", "non-bleached")
    source: str          # "gt" or "pred"


IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def ensure_class_string(cls: Any) -> str:
    return str(cls)


def find_image_by_stem(images_dir: str, stem: str) -> Optional[str]:
    if not images_dir or not os.path.isdir(images_dir):
        return None
    # Exact case match first
    for ext in IMG_EXTS:
        p = os.path.join(images_dir, stem + ext)
        if os.path.exists(p):
            return p
    # Case-insensitive fallback
    stem_lower = stem.lower()
    for fname in os.listdir(images_dir):
        if os.path.splitext(fname)[0].lower() == stem_lower:
            return os.path.join(images_dir, fname)
    return None


def get_image_size(path: str) -> Tuple[int, int]:
    """Return (W, H)."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    h, w = img.shape[:2]
    return (w, h)


def rasterize_polygon(poly_xy: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    """
    Rasterize an Nx2 polygon (x,y) into a binary mask of shape (H, W).
    Coordinates are clipped to image bounds.
    """
    h, w = shape_hw
    if poly_xy.size == 0:
        return np.zeros((h, w), dtype=bool)
    x = np.clip(poly_xy[:, 0], 0, w - 1)
    y = np.clip(poly_xy[:, 1], 0, h - 1)
    rr, cc = sk_polygon(y, x, shape=(h, w))
    mask = np.zeros((h, w), dtype=bool)
    mask[rr, cc] = True
    return mask


def build_class_mask(
    instances: List[Instance],
    image_shape_hw: Tuple[int, int],
    class_index_map: Dict[str, int],
    class_priority_high_to_low: List[str],
    background_value: int = -1,
) -> np.ndarray:
    """
    Build a per-pixel class mask from instance polygons with class priority.
    - class_index_map: maps class label string -> integer class id (e.g., {"non-bleached":0, "bleached":1})
    - class_priority_high_to_low: list of class labels, highest priority first (e.g., ["bleached","non-bleached"])
    Overlaps are resolved so that higher-priority classes overwrite lower-priority where they overlap.
    """
    H, W = image_shape_hw
    mask = np.full((H, W), fill_value=background_value, dtype=np.int16)

    # Process classes from lowest to highest priority so that highest overwrites last
    for cls_label in reversed(class_priority_high_to_low):
        if cls_label not in class_index_map:
            continue
        cls_id = class_index_map[cls_label]
        cls_union = np.zeros((H, W), dtype=bool)
        for inst in instances:
            if inst.class_id != cls_label:
                continue
            poly_mask = rasterize_polygon(inst.polygon, image_shape_hw)
            cls_union |= poly_mask
        mask[cls_union] = cls_id

    return mask


def _parse_polygon_line(parts: List[str], image_size: Tuple[int, int] | None) -> Tuple[str, np.ndarray]:
    """
    Accepts both:
      - cls conf x1 y1 x2 y2 ... (normalized or pixel)
      - cls x1 y1 x2 y2 ...
    Auto-detects normalized vs pixel coordinates by checking max value <= 1.5.
    If normalized, requires image_size (W,H) to convert to pixels.
    Returns: (raw_cls, points Nx2 in pixels)
    """
    if not parts:
        raise ValueError("Empty line")
    raw_cls = ensure_class_string(parts[0])

    # Detect optional conf: if parts[1] is float and remaining count even >= 6
    coord_start = 1
    if len(parts) >= 3:
        try:
            test = float(parts[1])
            if (len(parts) - 2) >= 6 and ((len(parts) - 2) % 2 == 0):
                coord_start = 2  # treat as conf present; we ignore conf in semantic eval
        except ValueError:
            pass

    coords = list(map(float, parts[coord_start:]))
    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError("Invalid polygon coordinate count")

    xs = np.array(coords[0::2], dtype=float)
    ys = np.array(coords[1::2], dtype=float)

    # Detect normalized vs pixel
    maxv = max(float(xs.max(initial=0.0)), float(ys.max(initial=0.0)))
    normalized = maxv <= 1.5
    if normalized:
        if image_size is None:
            raise ValueError("Normalized coordinates provided but image_size is None")
        W, H = image_size
        xs = xs * W
        ys = ys * H

    pts = np.stack([xs, ys], axis=1)
    return raw_cls, pts


def parse_txt_instances(
    txt_path: str,
    image_size: Tuple[int, int] | None,
    source: str,
    class_map: Dict[str, str],
) -> List[Instance]:
    """
    Parser for YOLO-seg style .txt lines for both GT and predictions.
    - Normalized coords are converted using image_size (W,H).
    - class_map maps raw labels (e.g., "1" or "0") to names (e.g., "bleached", "non-bleached").
    """
    instances: List[Instance] = []
    if not txt_path or not os.path.exists(txt_path):
        return instances
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                raw_cls, pts = _parse_polygon_line(parts, image_size=image_size)
                mapped_cls = class_map.get(raw_cls, raw_cls)  # default to raw if not in map
                instances.append(Instance(
                    polygon=pts,
                    class_id=mapped_cls,
                    source=source
                ))
            except Exception:
                # Skip malformed lines
                continue
    return instances


def collect_image_pairs(
    gt_dir: str,
    pred_dir: str,
    gt_ext: str = ".txt",
    pred_ext: str = ".txt",
) -> List[Tuple[str, str, str]]:
    """
    Pair GT and prediction files by stem.
    Returns list of (stem, gt_path_or_empty, pred_path_or_empty)
    """
    gt_files = {os.path.splitext(f)[0]: os.path.join(gt_dir, f)
                for f in os.listdir(gt_dir) if f.endswith(gt_ext)}
    pred_files = {os.path.splitext(f)[0]: os.path.join(pred_dir, f)
                  for f in os.listdir(pred_dir) if f.endswith(pred_ext)}
    stems = sorted(set(gt_files.keys()) | set(pred_files.keys()))
    return [(s, gt_files.get(s, ""), pred_files.get(s, "")) for s in stems]


# ==========================
# Metrics (semantic)
# ==========================

def compute_metrics_per_class(gt_mask: np.ndarray, pred_mask: np.ndarray, class_id: int) -> Dict[str, float]:
    gt = (gt_mask == class_id).astype(np.uint8)
    pred = (pred_mask == class_id).astype(np.uint8)

    tp = int(np.logical_and(pred == 1, gt == 1).sum())
    fp = int(np.logical_and(pred == 1, gt == 0).sum())
    fn = int(np.logical_and(pred == 0, gt == 1).sum())
    tn = int(np.logical_and(pred == 0, gt == 0).sum())

    eps = 1e-8

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    class_accuracy = tp / (tp + fn + eps)

    return {
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-Score": float(f1),
        "Dice": float(dice),
        "Pixel Accuracy": float(pixel_accuracy),
        "Class Accuracy": float(class_accuracy),
    }


def evaluate_semantic_image(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    class_ids: List[int],
) -> Dict[str, Any]:
    per_class = {}
    rows = []
    for cid in class_ids:
        m = compute_metrics_per_class(gt_mask, pred_mask, cid)
        per_class[cid] = m
        rows.append(m)

    keys = list(rows[0].keys()) if rows else []
    mean_metrics = {k: float(np.mean([r[k] for r in rows])) for k in keys} if rows else {}
    return {"per_class": per_class, "mean": mean_metrics}


# ==========================
# Evaluation runners
# ==========================

def derive_class_index_map(class_map: Dict[str, str]) -> Dict[str, int]:
    """
    Map class names -> indices. We ensure:
      "non-bleached" -> 0
      "bleached"     -> 1
    Any other classes get assigned next indices deterministically.
    """
    names = sorted(set(class_map.values()), key=lambda x: (x != "non-bleached", x))
    name_to_idx: Dict[str, int] = {}
    if "non-bleached" in names:
        name_to_idx["non-bleached"] = 0
    if "bleached" in names:
        name_to_idx["bleached"] = 1
    for n in names:
        if n not in name_to_idx:
            name_to_idx[n] = len(name_to_idx)
    return name_to_idx


def evaluate_image_semantic(
    stem: str,
    gt_instances: List[Instance],
    pred_instances: List[Instance],
    image_shape_hw: Tuple[int, int],
    class_index_map: Dict[str, int],
    class_priority_high_to_low: List[str],
) -> Dict[str, Any]:
    gt_mask = build_class_mask(
        instances=gt_instances,
        image_shape_hw=image_shape_hw,
        class_index_map=class_index_map,
        class_priority_high_to_low=class_priority_high_to_low,
        background_value=-1,
    )
    pred_mask = build_class_mask(
        instances=pred_instances,
        image_shape_hw=image_shape_hw,
        class_index_map=class_index_map,
        class_priority_high_to_low=class_priority_high_to_low,
        background_value=-1,
    )

    class_ids = sorted(set(class_index_map.values()))
    m = evaluate_semantic_image(gt_mask, pred_mask, class_ids=class_ids)

    return {
        "image_id": stem,
        "per_class_metrics": m["per_class"],
        "mean_metrics": m["mean"],
        "per_image_mean_iou": float(m["mean"].get("IoU", 0.0)),
    }


def run_fold(
    gt_dir: str,
    pred_dir: str,
    images_dir: str,
    class_map: Dict[str, str],
    class_priority: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run semantic evaluation for one fold.
    Returns:
      - per_image_df: per-image metrics dataframe
      - fold_summary: dict with summary (includes mean IoU over images)
    """
    pairs = collect_image_pairs(gt_dir, pred_dir, gt_ext=".txt", pred_ext=".txt")

    name_to_idx = derive_class_index_map(class_map)
    priority_names_high_to_low = class_priority

    rows = []

    for stem, gt_path, pred_path in tqdm(pairs, desc=f"Evaluating fold dir: {os.path.basename(os.path.dirname(pred_dir))}"):
        img_path = find_image_by_stem(images_dir, stem)
        if not img_path:
            # If image missing, skip
            continue
        image_size = get_image_size(img_path)  # (W,H)
        image_shape_hw = (image_size[1], image_size[0])

        gt_instances = parse_txt_instances(gt_path, image_size=image_size, source="gt", class_map=class_map)
        pred_instances = parse_txt_instances(pred_path, image_size=image_size, source="pred", class_map=class_map)

        res = evaluate_image_semantic(
            stem=stem,
            gt_instances=gt_instances,
            pred_instances=pred_instances,
            image_shape_hw=image_shape_hw,
            class_index_map=name_to_idx,
            class_priority_high_to_low=priority_names_high_to_low,
        )
        row = {
            "image_id": res["image_id"],
            "mean_IoU": res["mean_metrics"].get("IoU", 0.0),
            "mean_Dice": res["mean_metrics"].get("Dice", 0.0),
            "mean_Precision": res["mean_metrics"].get("Precision", 0.0),
            "mean_Recall": res["mean_metrics"].get("Recall", 0.0),
            "mean_F1": res["mean_metrics"].get("F1-Score", 0.0),
            "mean_Pixel_Accuracy": res["mean_metrics"].get("Pixel Accuracy", 0.0),
            "mean_Class_Accuracy": res["mean_metrics"].get("Class Accuracy", 0.0),
        }
        # Per-class IoU/Dice (0=non-bleached, 1=bleached by our mapping)
        for cid, md in res["per_class_metrics"].items():
            row[f"IoU_c{cid}"] = md.get("IoU", 0.0)
            row[f"Dice_c{cid}"] = md.get("Dice", 0.0)

        rows.append(row)

    per_image_df = pd.DataFrame(rows).sort_values("image_id")
    mean_iou_over_images = float(per_image_df["mean_IoU"].mean()) if len(per_image_df) else 0.0
    fold_summary = {
        "mode": "semantic",
        "num_images": int(len(per_image_df)),
        "mean_iou_over_images": mean_iou_over_images,
    }
    return per_image_df, fold_summary


def main():
    cfg = CONFIG.copy()
    os.makedirs(cfg["output_dir"], exist_ok=True)

    cv_rows = []
    pipeline_to_fold_means: Dict[str, List[float]] = {}

    for pipeline in cfg["pipelines"]:
        for fold in cfg["folds"]:
            pred_dir = cfg["pred_dir_pattern"].format(
                base=cfg["base_cv_dir"], pipeline=pipeline, fold=fold
            )
            if not os.path.isdir(pred_dir):
                print(f"[WARN] Prediction directory not found for {pipeline} fold {fold}: {pred_dir}")
                continue

            per_image_df, fold_summary = run_fold(
                gt_dir=cfg["gt_dir"],
                pred_dir=pred_dir,
                images_dir=cfg["images_dir"],
                class_map=cfg["class_map"],
                class_priority=cfg["class_priority"],
            )

            # Save per-image CSV for this pipeline/fold
            per_image_csv = os.path.join(
                cfg["output_dir"],
                cfg["per_image_csv_pattern"].format(pipeline=pipeline.replace("+", "_"), fold=fold)
            )
            per_image_df.to_csv(per_image_csv, index=False)

            mean_iou = fold_summary["mean_iou_over_images"]
            n_images = fold_summary["num_images"]

            cv_rows.append({
                "pipeline": pipeline,
                "fold": str(fold),
                "num_images": n_images,
                "mean_iou": mean_iou,
                "pred_dir": pred_dir,
            })
            pipeline_to_fold_means.setdefault(pipeline, []).append(mean_iou)

    # Save CV per-fold summary
    cv_df = pd.DataFrame(cv_rows).sort_values(["pipeline", "fold"])
    cv_summary_path = os.path.join(cfg["output_dir"], cfg["cv_summary_csv"])
    cv_df.to_csv(cv_summary_path, index=False)

    # Save per-pipeline unweighted mean across folds
    rows = []
    for pipeline, vals in pipeline_to_fold_means.items():
        arr = np.array(vals, dtype=float)
        rows.append({
            "pipeline": pipeline,
            "folds": int(len(arr)),
            "mean_iou_across_folds": float(arr.mean()) if len(arr) else 0.0,
            "std_iou_across_folds": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min_iou": float(arr.min()) if len(arr) else 0.0,
            "max_iou": float(arr.max()) if len(arr) else 0.0,
        })
    pipeline_df = pd.DataFrame(rows).sort_values("pipeline")
    pipeline_summary_path = os.path.join(cfg["output_dir"], cfg["pipeline_summary_csv"])
    pipeline_df.to_csv(pipeline_summary_path, index=False)

    print(f"[DONE] Saved CV per-fold summary to: {cv_summary_path}")
    print(f"[DONE] Saved per-pipeline summary to: {pipeline_summary_path}")
    print(f"[INFO] Per-image CSVs are in: {cfg['output_dir']}")


if __name__ == "__main__":
    main()
