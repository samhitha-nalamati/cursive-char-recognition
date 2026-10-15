"""
Unified character dataset. Reads from one or more canonical
`data/processed/<name>/{images/, labels.csv}` folders (produced by
convert_ccube.py, convert_choice.py, convert_emnist.py) so you can train on
any single dataset or any combination of them.
"""
import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from labels import CLASS_TO_IDX


class CharacterDataset(Dataset):
    def __init__(self, processed_dirs, img_size: int = 32, augment: bool = False):
        """
        Args:
            processed_dirs: str or list[str], each pointing at a
                data/processed/<name> folder containing images/ and labels.csv.
            augment: enable light augmentation (helps generalize across the
                different handwriting styles/sources you combine).
        """
        if isinstance(processed_dirs, str):
            processed_dirs = [processed_dirs]

        self.samples = []  # list of (image_path, label_idx)
        for d in processed_dirs:
            labels_csv = os.path.join(d, "labels.csv")
            img_dir = os.path.join(d, "images")
            with open(labels_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row["label"]
                    if label not in CLASS_TO_IDX:
                        continue
                    self.samples.append((os.path.join(img_dir, row["filename"]), CLASS_TO_IDX[label]))

        aug = []
        if augment:
            aug = [T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=6, fill=255)]

        self.transform = T.Compose(aug + [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), label

    def class_counts(self):
        """Returns {label_idx: count} -- useful for checking imbalance across combined sources."""
        counts = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts
