"""
Converts the CHoiCe dataset into this project's canonical layout:

    data/processed/choice/images/000001.png, ...
    data/processed/choice/labels.csv

This version derives each character's label DIRECTLY FROM ITS FILENAME,
since the actual distributed dataset (confirmed against a real download)
does not ship a label.txt -- instead each class folder contains images
named like "I (18).png" or "5 (1).png", where everything before " (" is
the ground-truth character. This is simpler and more robust than trusting
folder-name-to-class-index conventions, which vary between releases.

If your copy of CHoiCe *does* include a label.txt and you'd rather use
folder-index-based labeling instead, pass --label_txt explicitly.
"""
import argparse
import os
import re
import csv
from PIL import Image

from labels import CLASS_TO_IDX


FILENAME_LABEL_RE = re.compile(r"^(.*?)\s*\(")


def label_from_filename(filename: str):
    """'I (18).png' -> 'I'; '5 (1).png' -> '5'. Returns None if no match."""
    stem = os.path.splitext(filename)[0]
    m = FILENAME_LABEL_RE.match(stem)
    if not m:
        return None
    return m.group(1).strip()


def load_label_map(label_txt_path: str) -> dict:
    """Optional: folder_index -> character, if a label.txt is provided."""
    mapping = {}
    with open(label_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            mapping[parts[0]] = parts[1]
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--choice_root", required=True,
                         help="Path to the CHoiCe data folder (contains per-class subfolders, e.g. .../data)")
    parser.add_argument("--label_txt", default=None,
                         help="Optional: use folder-index labels from this file instead of filename parsing")
    parser.add_argument("--out_dir", default="data/processed/choice")
    parser.add_argument("--target_size", type=int, default=32)
    args = parser.parse_args()

    label_map = None
    if args.label_txt:
        label_map = load_label_map(args.label_txt)
        print(f"Using folder-index labels from {args.label_txt}")
    else:
        print("No --label_txt given; deriving labels directly from filenames (e.g. 'I (18).png' -> 'I')")

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    count = 0
    skipped = 0
    skipped_examples = []

    for root, _, files in os.walk(args.choice_root):
        folder_name = os.path.basename(root)
        for fname in sorted(files):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            if label_map is not None:
                char = label_map.get(folder_name)
            else:
                char = label_from_filename(fname)

            if char is None or char not in CLASS_TO_IDX:
                skipped += 1
                if len(skipped_examples) < 5:
                    skipped_examples.append((os.path.join(root, fname), char))
                continue

            src_path = os.path.join(root, fname)
            img = Image.open(src_path).convert("L")
            if img.size != (args.target_size, args.target_size):
                img = img.resize((args.target_size, args.target_size), Image.BILINEAR)

            out_name = f"{count:06d}.png"
            img.save(os.path.join(img_dir, out_name))
            rows.append((out_name, char))
            count += 1

    with open(os.path.join(args.out_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"Converted {count} characters -> {args.out_dir} ({skipped} skipped)")
    if skipped_examples:
        print("Examples of skipped files (unrecognized label):")
        for path, char in skipped_examples:
            print(f"    {path} -> parsed label: {char!r}")


if __name__ == "__main__":
    main()