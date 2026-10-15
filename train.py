"""
Train the character classifier on one or more processed datasets.

Examples:
    # Train on cursive-only sources
    python train.py --datasets data/processed/ccube data/processed/choice --epochs 30

    # Compare: add EMNIST into the mix and see how it shifts accuracy
    python train.py --datasets data/processed/ccube data/processed/choice data/processed/emnist --epochs 30

    # Train on just one source, to compare against the combined run
    python train.py --datasets data/processed/ccube --epochs 30
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

from dataset import CharacterDataset
from model import CharCNN
from labels import NUM_CLASSES, IDX_TO_CLASS


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    per_class_correct = {}
    per_class_total = {}
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(preds.tolist(), labels.tolist()):
                per_class_total[l] = per_class_total.get(l, 0) + 1
                if p == l:
                    per_class_correct[l] = per_class_correct.get(l, 0) + 1
    acc = correct / max(total, 1)
    per_class_acc = {
        IDX_TO_CLASS[l]: per_class_correct.get(l, 0) / per_class_total[l]
        for l in per_class_total
    }
    return acc, per_class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True,
                         help="One or more data/processed/<name> directories to combine")
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num_workers", type=int, default=0,
                         help="DataLoader worker processes. Default 0 (main process only) -- "
                              "safest on Windows, where worker spawning can hang or stall. "
                              "Try 2-4 only if you've confirmed multiprocessing works reliably.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    full_ds = CharacterDataset(args.datasets, img_size=args.img_size, augment=True)
    print(f"Loaded {len(full_ds)} samples from: {', '.join(args.datasets)}")

    n_val = int(len(full_ds) * args.val_fraction)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    # Validation shouldn't use training-time augmentation.
    val_ds.dataset = CharacterDataset(args.datasets, img_size=args.img_size, augment=False)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)

    model = CharCNN(num_classes=NUM_CLASSES, img_size=args.img_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("val_acc", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (val_acc={best_acc:.4f})")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_acc, per_class_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

        ckpt = {
            "model": model.state_dict(),
            "img_size": args.img_size,
            "epoch": epoch,
            "val_acc": val_acc,
            "datasets": args.datasets,
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pt"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"    -> new best model saved (val_acc={val_acc:.4f})")

    # Report the worst-performing classes at the end -- useful for spotting
    # confusable cursive letter pairs (e.g. 'n'/'u', 'a'/'o', 'C'/'c').
    _, final_per_class = evaluate(model, val_loader, device)
    worst = sorted(final_per_class.items(), key=lambda kv: kv[1])[:10]
    print("\nWorst-performing classes (likely confusable cursive letters):")
    for cls, acc in worst:
        print(f"    {cls!r}: {acc:.3f}")


if __name__ == "__main__":
    main()