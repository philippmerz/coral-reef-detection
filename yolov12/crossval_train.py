"""
k-Fold Stratified Cross-Validation driver for Ultralytics YOLO models.

What it does
- Uses the dataset's existing train set as the pool and ignores the existing val split.
- Stratifies a k-fold split by balancing class presence across folds using YOLO label files
    (multi-label aware: if an image has multiple classes, all are considered during balancing).
- For each fold i, creates two image list files (train/val) and a fold-specific data.yaml that
    points to those lists, then runs training with the provided model.
- After each fold finishes, it reads the results.csv to collect final metrics and prints/exports a summary.

How to run (PowerShell):
    python scripts/crossval_train.py

Notes
- This script requires the ultralytics package available in this repo.
- For segmentation/detection, labels are resolved by Ultralytics via images->labels path mapping.
"""

from __future__ import annotations

import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError as e:
    print("Missing dependency: pyyaml is required. Please install it (pip install pyyaml).", file=sys.stderr)
    raise


# Supported image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class FoldFiles:
    train_list: Path
    val_list: Path
    data_yaml: Path
    fold_id: int


def read_dataset_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Basic validation
    for key in ("path", "train", "names"):
        if key not in data:
            raise ValueError(f"Dataset YAML missing required key: {key}")
    return data


def collect_train_images(dataset: dict) -> List[Path]:
    base = Path(dataset["path"])  # absolute or relative path root
    train_rel = Path(dataset["train"])  # e.g., images/train or a list file

    # We only support directory for CV source (images/train), not list files
    if train_rel.suffix.lower() in {".txt", ".csv"}:
        # If train is a list file, read it directly
        list_path = train_rel if train_rel.is_absolute() else base / train_rel
        if not list_path.exists():
            raise FileNotFoundError(f"Train list file not found: {list_path}")
        images = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                p = Path(line.strip())
                if p.suffix.lower() in IMG_EXTS and p.exists():
                    images.append(p.resolve())
        if not images:
            raise RuntimeError("No images found from train list file.")
        return images

    # Else, assume directory relative to base
    images_dir = train_rel if train_rel.is_absolute() else base / train_rel
    if not images_dir.exists():
        raise FileNotFoundError(f"Train images directory not found: {images_dir}")

    images: List[Path] = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            images.append(p.resolve())

    if not images:
        raise RuntimeError(f"No images found in {images_dir}")
    return images


def kfold_split(items: List[Path], k: int, seed: int) -> List[List[Path]]:
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(items) < k:
        raise ValueError(f"Number of items ({len(items)}) is smaller than k ({k}).")

    rnd = random.Random(seed)
    idx = list(range(len(items)))
    rnd.shuffle(idx)

    fold_sizes = [len(items) // k] * k
    for i in range(len(items) % k):
        fold_sizes[i] += 1

    folds: List[List[Path]] = []
    start = 0
    for size in fold_sizes:
        end = start + size
        fold = [items[i] for i in idx[start:end]]
        folds.append(fold)
        start = end
    return folds


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def _normalize_folds_selection(folds_to_run, total_folds: int) -> List[int]:
    """Normalize folds_to_run into a list of 1-based fold indices within [1, total_folds].
    Accepts int, str (e.g., '2'), iterable of ints/strs, or None/empty.
    """
    if folds_to_run is None:
        return []
    # Single int
    if isinstance(folds_to_run, int):
        val = folds_to_run
        return [val] if 1 <= val <= total_folds else []
    # String: allow comma/space separated values like "2" or "1,3" or "1 3"
    if isinstance(folds_to_run, str):
        s = folds_to_run.strip()
        if not s:
            return []
        # Split on commas or whitespace
        tokens = [t for chunk in s.split(',') for t in chunk.split()]
        out: List[int] = []
        for t in tokens:
            try:
                v = int(t)
            except Exception:
                continue
            if 1 <= v <= total_folds and v not in out:
                out.append(v)
        return out
    # Iterable
    try:
        seq = list(folds_to_run)
    except TypeError:
        return []
    out: List[int] = []
    for x in seq:
        try:
            v = int(x)
        except Exception:
            continue
        if 1 <= v <= total_folds and v not in out:
            out.append(v)
    return out


def _image_to_label_path(img_path: Path, base_path: Path) -> Path | None:
    """Map image path under base .../images/... to YOLO label .txt under .../labels/...."""
    try:
        rel = img_path.relative_to(base_path)
    except Exception:
        # Best-effort replacement on absolute path
        parts = list(img_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return Path(*parts).with_suffix(".txt")
        return None

    parts = list(rel.parts)
    if "images" not in parts:
        return None
    parts[parts.index("images")] = "labels"
    label_rel = Path(*parts).with_suffix(".txt")
    return (base_path / label_rel).resolve()


def _parse_label_classes(label_path: Path) -> List[int]:
    """Return sorted list of class ids present in a YOLO label file. Empty if missing."""
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            classes = set()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tok = line.split()[0]
                try:
                    c = int(float(tok))
                    classes.add(c)
                except Exception:
                    continue
            return sorted(classes)
    except FileNotFoundError:
        return []


def stratified_kfold_split(items: List[Path], base_path: Path, k: int, seed: int) -> List[List[Path]]:
    """Greedy multi-label stratified split balancing class presence across folds."""
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(items) < k:
        raise ValueError(f"Number of items ({len(items)}) is smaller than k ({k}).")

    item_classes: List[List[int]] = []
    all_classes: set[int] = set()
    for p in items:
        lp = _image_to_label_path(p, base_path)
        cls = _parse_label_classes(lp) if lp else []
        item_classes.append(cls)
        all_classes.update(cls)

    folds: List[List[Path]] = [[] for _ in range(k)]
    fold_class_counts: List[Dict[int, int]] = [dict() for _ in range(k)]
    for d in fold_class_counts:
        for c in all_classes:
            d[c] = 0

    rnd = random.Random(seed)
    order = list(range(len(items)))
    rnd.shuffle(order)

    for idx in order:
        cls = item_classes[idx]
        best_fold = 0
        best_score = None
        for j in range(k):
            score = sum(fold_class_counts[j].get(c, 0) for c in cls)
            tie = len(folds[j])
            cand = (score, tie)
            if best_score is None or cand < best_score:
                best_score = cand
                best_fold = j
        folds[best_fold].append(items[idx])
        for c in cls:
            fold_class_counts[best_fold][c] = fold_class_counts[best_fold].get(c, 0) + 1

    return folds


def write_list_file(path: Path, items: List[Path]) -> None:
    ensure_dirs(path)
    with open(path, "w", encoding="utf-8") as f:
        for p in items:
            f.write(str(p) + "\n")


def make_fold_yaml(base_dataset: dict, train_list: Path, val_list: Path, out_path: Path) -> None:
    ensure_dirs(out_path)
    fold_ds = dict(base_dataset)
    # Replace train/val with list file paths
    fold_ds["train"] = str(train_list)
    fold_ds["val"] = str(val_list)
    # Keep other fields (path, names, test if present)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(fold_ds, f, sort_keys=False)


def prepare_fold_files(dataset_yaml: Path, work_root: Path, k: int, seed: int, force_regenerate: bool = False) -> Tuple[List[FoldFiles], List[Path]]:
    dataset = read_dataset_yaml(dataset_yaml)
    images = collect_train_images(dataset)
    base_path_cfg = Path(dataset["path"]) if isinstance(dataset.get("path"), str) else Path(work_root)
    base_path = (base_path_cfg if base_path_cfg.is_absolute() else (work_root / base_path_cfg)).resolve()
    splits_dir = work_root / "splits_kfold"
    data_dir = work_root / "folds"
    data_dir.mkdir(parents=True, exist_ok=True)

    fold_files: List[FoldFiles] = []

    # 1) Prefer using existing folds from splits_kfold if available
    existing_ids: List[int] = []
    if splits_dir.exists() and not force_regenerate:
        # find fold_?_train.txt and matching val
        for p in sorted(splits_dir.glob("fold_*_train.txt")):
            name = p.stem  # fold_X_train
            try:
                idx = int(name.split("_")[1])
            except Exception:
                continue
            val_file = splits_dir / f"fold_{idx}_val.txt"
            if val_file.exists():
                existing_ids.append(idx)
        existing_ids = sorted(set(existing_ids))

    if existing_ids and not force_regenerate:
        print(f"Found existing folds in {splits_dir}: {existing_ids}")
        for idx in existing_ids:
            train_list = splits_dir / f"fold_{idx}_train.txt"
            val_list = splits_dir / f"fold_{idx}_val.txt"
            data_yaml = data_dir / f"fold_{idx}.yaml"
            if not data_yaml.exists():
                make_fold_yaml(dataset, train_list, val_list, data_yaml)
            fold_files.append(FoldFiles(train_list=train_list, val_list=val_list, data_yaml=data_yaml, fold_id=idx))
        return fold_files, images

    # 2) Else, create new stratified folds and write lists
    splits_dir.mkdir(parents=True, exist_ok=True)
    # Adjust k if dataset is too small
    k_eff = max(1, min(k, len(images)))
    if k_eff != k:
        print(f"Warning: Requested k={k} but only {len(images)} images available. Using k={k_eff}.")
    if k_eff == 1:
        folds = [sorted(images)]
    else:
        folds = stratified_kfold_split(images, base_path, k=k_eff, seed=seed)

    # For consistency, keep a stable order in each fold list
    for fold in folds:
        fold.sort()

    for i in range(len(folds)):
        val_items = folds[i]
        train_items = [p for j, fold in enumerate(folds) if j != i for p in fold]

        train_list = splits_dir / f"fold_{i+1}_train.txt"
        val_list = splits_dir / f"fold_{i+1}_val.txt"
        write_list_file(train_list, train_items)
        write_list_file(val_list, val_items)

        data_yaml = data_dir / f"fold_{i+1}.yaml"
        make_fold_yaml(dataset, train_list, val_list, data_yaml)

        fold_files.append(FoldFiles(train_list=train_list, val_list=val_list, data_yaml=data_yaml, fold_id=i+1))

    return fold_files, images


def run_training_over_folds(
    fold_files: List[FoldFiles],
    model_path: Path,
    epochs: int,
    imgsz: int,
    project: Path,
    device: str,
    seed: int,
    batch: int | None = None,
    workers: int | None = None,
    extra_args: Dict | None = None,
) -> List[Path]:
    from ultralytics import YOLO  # local package
    
    def try_train_with_backoff(model, base_args: Dict, max_retries: int = 3) -> Tuple[bool, Dict]:
        """Attempt training with decreasing batch and image size on CUDA OOM.
        Returns (success, final_args)."""
        # Local copy we can mutate
        args = dict(base_args)
        # Ensure AMP is on if supported (saves VRAM)
        args.setdefault("amp", True)
        # Provide safe defaults if missing
        cur_batch = int(args.get("batch", 4))
        cur_imgsz = int(args.get("imgsz", 640))
        args["batch"] = cur_batch
        args["imgsz"] = cur_imgsz

        for attempt in range(max_retries + 1):
            try:
                _ = model.train(**args)
                return True, args
            except RuntimeError as e:
                msg = str(e)
                is_oom = "out of memory" in msg.lower() or "CUDA error" in msg or "CUBLAS" in msg
                if not is_oom or attempt >= max_retries:
                    # Not an OOM, or retries exhausted
                    raise
                # Backoff strategy: halve batch until 1, then reduce imgsz by 128 down to 320
                if cur_batch > 1:
                    cur_batch = max(1, cur_batch // 2)
                    args["batch"] = cur_batch
                    print(f"CUDA OOM detected. Retrying with smaller batch: {cur_batch}")
                elif cur_imgsz > 320:
                    cur_imgsz = max(320, cur_imgsz - 128)
                    args["imgsz"] = cur_imgsz
                    print(f"CUDA OOM persists. Retrying with smaller imgsz: {cur_imgsz}")
                else:
                    # Nothing else to reduce
                    raise
                # Optional: clear CUDA cache before retry
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        return False, args

    run_dirs: List[Path] = []
    for ff in fold_files:
        fold_name = f"fold_{ff.fold_id}"
        print(f"\n=== Training {fold_name} ===")
        try:
            print(f"Using lists: {ff.train_list.name} (train), {ff.val_list.name} (val)")
        except Exception:
            pass
        model = YOLO(str(model_path))

        train_args = dict(
            data=str(ff.data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            project=str(project),
            name=fold_name,
            device=device,
            seed=seed,
            verbose=True,
        )
        if batch is not None:
            train_args["batch"] = batch
        if workers is not None:
            train_args["workers"] = workers
        if extra_args:
            train_args.update(extra_args)

        # Run training with OOM backoff retries
        _ok, final_args = try_train_with_backoff(model, train_args, max_retries=3)

        # Determine run directory from trainer if available (more robust than assuming path)
        run_dir = None
        try:
            save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
            if save_dir:
                run_dir = Path(save_dir)
        except Exception:
            pass
        if run_dir is None:
            # Search under project for results.csv paths containing this fold name
            try:
                candidates = []
                for csv_path in project.rglob("results.csv"):
                    pstr = str(csv_path.parent)
                    if fold_name in pstr:
                        candidates.append(csv_path.parent)
                if not candidates:
                    # Pick latest results.csv anywhere under project
                    any_csv = list(project.rglob("results.csv"))
                    if any_csv:
                        candidates = [max(any_csv, key=lambda p: p.stat().st_mtime).parent]
                if candidates:
                    # Choose the most recently modified candidate
                    run_dir = max(candidates, key=lambda d: (d / "results.csv").stat().st_mtime)
            except Exception:
                run_dir = None
        if run_dir is None:
            # Final fallback: assume {project}/{name}
            run_dir = project / fold_name

        results_file = run_dir / "results.csv"
        print(f"Resolved run directory for {fold_name}: {run_dir}")
        print(f"  results.csv exists: {results_file.exists()}")
        run_dirs.append(run_dir)

    return run_dirs


def read_last_row(csv_path: Path) -> Tuple[List[str], List[str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if len(rows) < 2:
            raise RuntimeError(f"results.csv appears incomplete: {csv_path}")
        header = rows[0]
        last = rows[-1]
        return header, last


def summarize_metrics(run_dirs: List[Path]) -> Dict[str, float]:
    # Aggregate any columns that look like metrics across folds by simple average
    agg: Dict[str, List[float]] = {}
    for rd in run_dirs:
        csv_path = rd / "results.csv"
        if not csv_path.exists():
            print(f"Warning: results.csv not found for {rd}; skipping in summary.")
            continue
        header, last = read_last_row(csv_path)
        for k, v in zip(header, last):
            if not k or k.lower().startswith("epoch"):
                continue
            try:
                val = float(v)
            except ValueError:
                continue
            agg.setdefault(k, []).append(val)

    means: Dict[str, float] = {}
    for k, vals in agg.items():
        if not vals:
            continue
        means[k] = sum(vals) / len(vals)
    return means


def save_summary(project: Path, summary: Dict[str, float]) -> None:
    out = project / "cv_summary.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=True)
    print(f"\nSaved cross-validation summary to: {out}")


def main() -> None:
    # parameters
    dataset_yaml = Path("dataset/dataset.yaml").resolve()
    model_path = Path("yolov12m-seg.pt").resolve() 
    folds = 5  # total number of folds to GENERATE (stratified K)
    # To run a specific fold or subset manually, set this to a 1-based list, e.g., [2] or [1,3].
    # Set to None or [] to run all folds.
    folds_to_run = [5]  # (optional) run only selected fold
    # If True, ignore any existing splits in dataset/splits_kfold and rebuild exactly 'folds' splits.
    force_regenerate = False
    seed = 20251006
    epochs = 5
    imgsz = 640
    project_dir = (Path("runs") / "cv").resolve()
    device = "0"  # "0" for GPU0, "cpu" for CPU, empty string lets Ultralytics choose
    batch = 4
    workers = 2
    # Lighter, more stable training profile (good for Windows/limited CPU)
    extra_args: Dict = {
        "plots": True,     
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "erasing": 0.0,
        "multi_scale": False,        
    }

    # Derive workspace root from dataset.yaml location
    work_root = dataset_yaml.parent  # e.g., dataset/

    print("Preparing stratified K-fold split and fold-specific data files...")
    fold_files, all_images = prepare_fold_files(dataset_yaml, work_root, k=folds, seed=seed, force_regenerate=force_regenerate)
    print(f"Total images in pool: {len(all_images)}")
    print(f"Total folds requested to generate: {folds}")
    for i, ff in enumerate(fold_files):
        print(f"Fold {i+1}: train list -> {ff.train_list.name} | val list -> {ff.val_list.name} | data -> {ff.data_yaml.name}")

    # Select specific folds if requested (robust parsing of int/str/list)
    print(f"\nRequested folds_to_run: {folds_to_run!r} (available: 1..{len(fold_files)})")
    valid_idx = _normalize_folds_selection(folds_to_run, total_folds=len(fold_files))
    if valid_idx:
        selected_folds = [fold_files[i - 1] for i in valid_idx]
        print(f"Parsed fold indices: {valid_idx}\nWill run only folds: {valid_idx}")
    else:
        selected_folds = fold_files
        if folds_to_run:
            print(f"Parsed fold indices: []\nNo valid fold indices parsed from {folds_to_run!r}; will run all folds 1..{len(fold_files)}.")
        else:
            print("\nWill run all folds.")

    project_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting cross-validation training...")
    run_dirs = run_training_over_folds(
        fold_files=selected_folds,
        model_path=model_path,
        epochs=epochs,
        imgsz=imgsz,
        project=project_dir,
        device=device,
        seed=seed,
        batch=batch,
        workers=workers,
        extra_args=extra_args,
    )

    print("\nCollecting and summarizing metrics across folds...")
    summary = summarize_metrics(run_dirs)
    if summary:
        # Print a small, commonly interesting subset first if present
        keys_of_interest = [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
            "train/box_loss",
            "train/seg_loss",
            "val/box_loss",
            "val/seg_loss",
        ]
        print("\nCross-Validation Means (available metrics):")
        for k in keys_of_interest:
            if k in summary:
                print(f"  {k}: {summary[k]:.5f}")

        # Print any other metrics
        other = {k: v for k, v in summary.items() if k not in keys_of_interest}
        if other:
            print("\nOther averaged metrics:")
            for k in sorted(other.keys()):
                print(f"  {k}: {other[k]:.5f}")

        save_summary(project_dir, summary)
    else:
        print("No results.csv files found; cannot summarize.")


if __name__ == "__main__":
    main()
