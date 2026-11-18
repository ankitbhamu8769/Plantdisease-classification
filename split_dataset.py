"""
split_dataset.py
-----------------
Splits a dataset that currently looks like:

    raw_data/
        Apple___Scab/*.jpg
        Apple___Healthy/*.jpg
        Tomato___Leaf_Mold/*.jpg
        ...

into the train/val/test layout expected by dataset.py:

    data/
        train/Apple___Scab/*.jpg
        val/Apple___Scab/*.jpg
        test/Apple___Scab/*.jpg
        ...

Usage:
    python split_dataset.py --source /path/to/raw_data --dest ../data \
        --train 0.8 --val 0.1 --test 0.1

By default it COPIES files (safe, keeps your original download intact).
Pass --move if you'd rather move them (saves disk space, but is irreversible).
"""

import argparse
import os
import shutil
import random
from pathlib import Path

from sklearn.model_selection import train_test_split


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(class_dir: Path):
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]


def split_dataset(source_dir: str, dest_dir: str,
                   train_frac: float, val_frac: float, test_frac: float,
                   move: bool, seed: int):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "train + val + test fractions must sum to 1.0"

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subfolders found in {source_dir}. "
                          f"Expected structure: {source_dir}/<class_name>/*.jpg")

    random.seed(seed)
    file_op = shutil.move if move else shutil.copy2

    summary = []

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = list_images(class_dir)

        if len(images) == 0:
            print(f"  [skip] {class_name}: no images found")
            continue

        train_files, remaining_files = train_test_split(
            images, train_size=train_frac, random_state=seed
        )

        relative_val_frac = val_frac / (val_frac + test_frac)
        val_files, test_files = train_test_split(
            remaining_files, train_size=relative_val_frac, random_state=seed
        )

        for split_name, files in [("train", train_files),
                                   ("val", val_files),
                                   ("test", test_files)]:
            out_dir = dest_dir / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                file_op(str(f), str(out_dir / f.name))

        summary.append((class_name, len(train_files), len(val_files), len(test_files)))
        print(f"  {class_name}: train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    print("\nDone.")
    print(f"{'Class':<40}{'Train':>8}{'Val':>8}{'Test':>8}")
    for class_name, n_train, n_val, n_test in summary:
        print(f"{class_name:<40}{n_train:>8}{n_val:>8}{n_test:>8}")

    total_train = sum(s[1] for s in summary)
    total_val = sum(s[2] for s in summary)
    total_test = sum(s[3] for s in summary)
    print(f"\nTotal classes: {len(summary)}")
    print(f"Total images -> train={total_train}, val={total_val}, test={total_test}")
    print(f"\n>>> Set NUM_CLASSES = {len(summary)} in config.py <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                         help="Path to the raw dataset (one folder per class)")
    parser.add_argument("--dest", default="../data",
                         help="Destination root that will contain train/val/test (default: ../data)")
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction (default 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Val fraction (default 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Test fraction (default 0.1)")
    parser.add_argument("--move", action="store_true",
                         help="Move files instead of copying (saves disk space, irreversible)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    split_dataset(args.source, args.dest, args.train, args.val, args.test,
                  args.move, args.seed)