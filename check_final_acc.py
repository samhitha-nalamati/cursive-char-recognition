"""
Reads the final results saved inside a checkpoint file.

Usage:
    python check_final_acc.py
"""
import torch

ckpt = torch.load("checkpoints/best.pt", map_location="cpu")

print(f"Best validation accuracy: {ckpt['val_acc']:.4f}  ({ckpt['val_acc'] * 100:.2f}%)")
print(f"Achieved at epoch:        {ckpt['epoch'] + 1}")
print(f"Trained on:                {', '.join(ckpt['datasets'])}")
print(f"Image size:                {ckpt['img_size']}x{ckpt['img_size']}")